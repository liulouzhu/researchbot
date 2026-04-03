"""Local semantic search tool for papers.

Provides hybrid search (FTS5 + vector) over local literature storage.
Supports filtering by topic, tags, year, categories, and source.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.config.schema import SemanticSearchConfig
from researchbot.search_index import SearchIndex


class PaperSearchLocalTool(Tool):
    """Search local papers using semantic search.

    Supports hybrid search combining keyword matching (FTS5) and
    vector similarity (sqlite-vec). Falls back to FTS5-only
    when vector search is unavailable.

    Filtering:
    - topic: substring match on topic/tags
    - tags: any of the specified tags match
    - year: exact year match
    - year_from/year_to: year range
    - categories: any of the categories match
    - source: exact source match (arxiv, crossref, openalex)
    """

    name = "paper_search_local"
    description = (
        "Search local papers using semantic search. "
        "Combines keyword and vector similarity for best results. "
        "Supports filtering by topic, tags, year, categories, and source. "
        "Uses hybrid search with reranking when enabled."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g. 'machine learning for graph analysis')",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "topic": {
                "type": "string",
                "description": "Filter by topic (substring match)",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by tags (any match)",
            },
            "year": {
                "type": "string",
                "description": "Filter by exact publication year",
            },
            "year_from": {
                "type": "integer",
                "description": "Filter by year >= this value",
            },
            "year_to": {
                "type": "integer",
                "description": "Filter by year <= this value",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by categories (any match)",
            },
            "source": {
                "type": "string",
                "description": "Filter by source (arxiv, crossref, openalex)",
            },
            "rerank": {
                "type": "boolean",
                "description": "Apply LLM reranking to top results",
                "default": True,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._index: SearchIndex | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_index(self) -> SearchIndex:
        """Get or create the search index."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        return self._index

    async def _ensure_initialized(self) -> None:
        """Ensure index is initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    index = self._get_index()
                    await index.initialize()
                    self._initialized = True

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        topic: str | None = None,
        tags: list[str] | None = None,
        year: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
        source: str | None = None,
        rerank: bool = True,
        **kwargs: Any,
    ) -> str:
        """Search local papers."""
        if not self._workspace:
            return "Error: workspace not configured"

        try:
            await self._ensure_initialized()
            index = self._get_index()

            results = await index.search(
                query=query,
                top_k=top_k,
                topic=topic,
                tags=tags,
                year=year,
                year_from=year_from,
                year_to=year_to,
                categories=categories,
                source=source,
                rerank=rerank,
            )

            if not results:
                return f"No papers found for query: {query}\n\nTry broadening your search or removing filters."

            # Build output
            lines = [
                f"Search results for: {query}\n",
                f"Found {len(results)} result(s)\n",
                f"Search mode: {'hybrid (FTS5 + vector)' if index.sqlite_vec_available and index.embedding_client.available else 'FTS5 keyword only'}\n",
                "-" * 60,
            ]

            for i, r in enumerate(results, 1):
                paper_id = r.get("paper_id", "?")
                title = r.get("title", "?")
                year = r.get("year", "?")
                source = r.get("source", "")
                final_score = r.get("final_score", 0)
                lex_score = r.get("lexical_score", 0)
                vec_score = r.get("vector_score", 0)
                rerank_score = r.get("rerank_score")

                lines.append(f"\n[{i}] {title}")
                lines.append(f"    Paper ID: {paper_id}")
                lines.append(f"    Year: {year} | Source: {source}")
                lines.append(f"    Scores: final={final_score:.3f} lexical={lex_score:.3f} vector={vec_score:.3f}")
                if rerank_score is not None:
                    lines.append(f"    Rerank score: {rerank_score}")

                # Show matched filters if any
                filters = r.get("matched_filters", {})
                if filters.get("topic"):
                    lines.append(f"    Topic: {filters['topic']}")
                if filters.get("tags"):
                    lines.append(f"    Tags: {', '.join(filters['tags'][:5])}")

                # Show abstract snippet if available
                snippet = r.get("abstract_snippet", "")
                if snippet:
                    lines.append(f"    ...{snippet}...")

            return "\n".join(lines)

        except Exception as e:
            return f"Search error: {e}"

    def close(self) -> None:
        """Close the search index."""
        if self._index:
            self._index.close()
            self._index = None
            self._initialized = False


class PaperIndexTool(Tool):
    """Update the local paper search index.

    Indexes or re-indexes papers in local storage for semantic search.
    Normally called automatically, but can be invoked manually.
    """

    name = "paper_index"
    description = (
        "Update the local paper search index. "
        "Normally happens automatically when papers are saved, "
        "but can be invoked manually to rebuild the index."
    )

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Index a specific paper by ID (optional)",
            },
            "rebuild": {
                "type": "boolean",
                "description": "Rebuild the entire index from all local papers",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._index: SearchIndex | None = None

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_index(self) -> SearchIndex:
        """Get or create the search index."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        return self._index

    def _load_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load a paper from local storage."""
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_all_papers(self) -> list[dict[str, Any]]:
        """Load all papers from local storage."""
        papers_dir = self._resolve_path("literature/papers")
        if not papers_dir.exists():
            return []

        papers = []
        import json as json_module
        for fp in papers_dir.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    papers.append(json_module.load(f))
            except Exception:
                continue
        return papers

    async def execute(
        self,
        paper_id: str | None = None,
        rebuild: bool = False,
        **kwargs: Any,
    ) -> str:
        """Update the search index."""
        if not self._workspace:
            return "Error: workspace not configured"

        index = self._get_index()
        await index.initialize()

        if rebuild:
            papers = self._load_all_papers()
            updated = 0
            for paper in papers:
                pid = paper.get("paper_id", "")
                if pid:
                    if await index.upsert_paper(paper):
                        updated += 1
            index.close()
            return f"Index rebuilt: {updated}/{len(papers)} papers updated"

        if paper_id:
            paper = self._load_paper(paper_id)
            if not paper:
                return f"Paper not found: {paper_id}"
            updated = await index.upsert_paper(paper)
            index.close()
            if updated:
                return f"Indexed paper: {paper_id}"
            else:
                return f"Paper unchanged (skipped): {paper_id}"

        return "Error: specify paper_id or rebuild=True"
