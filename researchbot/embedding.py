"""Embedding client for generating text embeddings.

Supports阿里云 dashscope embedding models and any OpenAI-compatible endpoint.
Designed to gracefully degrade when embedding is unavailable.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import httpx
from loguru import logger

from researchbot.config.schema import SemanticSearchConfig


class EmbeddingClient:
    """Client for generating text embeddings.

    Supports阿里云 dashscope (text-embedding-v1, text-embedding-v2, text-embedding-v4)
    and any OpenAI-compatible embedding endpoint.

    Usage:
        client = EmbeddingClient(config=semantic_search_config)
        vector = await client.embed("Hello world")
        vectors = await client.embed_batch(["Hello", "World"])
    """

    # Supported embedding providers
    _SUPPORTED_PROVIDERS = {"dashscope", "openai", "openai-compatible"}

    def __init__(self, config: SemanticSearchConfig | None = None):
        self._config = config or SemanticSearchConfig()
        self._client: httpx.AsyncClient | None = None
        self._embedding_dimension: int | None = None
        self._available: bool = self._is_configured()
        self._lock = asyncio.Lock()

    def _is_configured(self) -> bool:
        """Check if embedding is properly configured (not just available after a request)."""
        if not self._config.embedding_api_key:
            return False
        provider = self._config.embedding_provider.lower()
        if provider and provider not in self._SUPPORTED_PROVIDERS:
            return False
        return True

    def is_enabled(self) -> bool:
        """Whether embedding is enabled based on configuration.

        Unlike `available`, this reflects static config (api key, provider)
        rather than runtime availability after requests.
        """
        return self._is_configured()

    @property
    def available(self) -> bool:
        """Whether embedding service is available (runtime state, may change after failures)."""
        return self._available and self._is_configured()

    @property
    def dimension(self) -> int | None:
        """Embedding vector dimension (None if unknown)."""
        return self._embedding_dimension

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _resolve_api_base(self) -> str:
        """Resolve embedding API base URL from config or provider defaults."""
        if self._config.embedding_api_base:
            return self._config.embedding_api_base

        # dashscope 阿里云 dashscope API base
        if self._config.embedding_provider == "dashscope":
            return "https://dashscope.aliyuncs.com/api/v1/services/embeddings"
        # 默认 OpenAI-compatible
        return "https://api.openai.com/v1"

    def _build_dashscope_payload(self, texts: list[str], model: str) -> dict[str, Any]:
        """Build阿里云 dashscope embedding request payload."""
        return {
            "model": model,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"},
        }

    def _parse_dashscope_response(self, response: dict[str, Any]) -> tuple[list[list[float]], int]:
        """Parse阿里云 dashscope embedding response.

        Returns (embeddings, dimension).
        """
        output = response.get("output", {})
        embeddings_data = output.get("embeddings", [])

        vectors: list[list[float]] = []
        dimension = 0
        for item in embeddings_data:
            vector = item.get("embedding", [])
            if dimension == 0:
                dimension = len(vector)
            vectors.append(vector)

        return vectors, dimension

    def _build_openai_compat_payload(self, texts: list[str], model: str) -> dict[str, Any]:
        """Build OpenAI-compatible embedding request payload."""
        return {"model": model, "input": texts}

    def _parse_openai_compat_response(
        self, response: dict[str, Any]
    ) -> tuple[list[list[float]], int]:
        """Parse OpenAI-compatible embedding response.

        Returns (embeddings, dimension).
        """
        data = response.get("data", [])
        vectors: list[list[float]] = []
        dimension = 0

        # Sort by index to maintain order
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        for item in sorted_data:
            embedding = item.get("embedding", [])
            if dimension == 0:
                dimension = len(embedding)
            vectors.append(embedding)

        return vectors, dimension

    async def embed(self, text: str) -> list[float] | None:
        """Generate embedding for a single text.

        Returns None if embedding service is unavailable.
        """
        results = await self.embed_batch([text])
        return results[0] if results else None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings for multiple texts.

        Returns None if embedding service is unavailable.
        """
        if not texts:
            return []

        if not self._available:
            return None

        try:
            api_base = self._resolve_api_base()
            model = self._config.embedding_model
            client = await self._get_client()

            # Determine payload format based on provider
            # If embedding_api_base is explicitly set to compatible-mode URL, treat as OpenAI-compatible
            is_compatible_mode = (
                self._config.embedding_api_base
                and "compatible-mode" in self._config.embedding_api_base
            )
            if self._config.embedding_provider == "dashscope" and not is_compatible_mode:
                payload = self._build_dashscope_payload(texts, model)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.embedding_api_key}",
                }
                url = api_base
                parse_fn = self._parse_dashscope_response
            else:
                # OpenAI-compatible (including dashscope compatible-mode)
                payload = self._build_openai_compat_payload(texts, model)
                api_key = self._config.embedding_api_key or "dummy"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                # embedding API path - append /embeddings to base URL
                if api_base.endswith("/v1") or api_base.endswith("/v1/"):
                    url = api_base.rstrip("/") + "/embeddings"
                else:
                    url = api_base.rstrip("/") + "/v1/embeddings"
                parse_fn = self._parse_openai_compat_response

            async with self._lock:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    content = response.text[:200]
                    logger.warning(
                        "Embedding API error ({}): {}", response.status_code, content
                    )
                    self._available = False
                    return None

                data = response.json()
                vectors, dimension = parse_fn(data)

                if self._embedding_dimension is None and dimension > 0:
                    self._embedding_dimension = dimension

                return vectors

        except Exception as e:
            logger.warning("Embedding service unavailable: {}", e)
            self._available = False
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def mark_unavailable(self) -> None:
        """Mark embedding service as unavailable (for fallback)."""
        self._available = False

    def mark_available(self) -> None:
        """Mark embedding service as available."""
        self._available = True


def vector_to_bytes(vector: list[float]) -> bytes:
    """Serialize a float vector to bytes for SQLite storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def bytes_to_vector(data: bytes) -> list[float]:
    """Deserialize a float vector from bytes."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))
