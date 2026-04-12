"""Method extraction and search tools for reusable techniques."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from researchbot.agent.tools.base import Tool

logger = logging.getLogger(__name__)
from researchbot.config.schema import SemanticSearchConfig
from researchbot.search_index import SearchIndex
from researchbot.utils.helpers import compute_short_id, extract_json_array


# =============================================================================
# Prompt templates for method extraction
# =============================================================================

EXTRACT_METHODS_PROMPT = """You are a research paper analyst specializing in reusable technical methods.

Given a research paper abstract and/or full text, extract all reusable technical methods and modules
that could be applied to other research projects.

## Paper Abstract
{abstract}

## Output Format
Extract methods as a JSON array. Return ONLY the JSON array, no additional text.
[
  {{
    "method_name": "Name of the method (e.g., 'Attention Gate', 'Feature Pyramid Network')",
    "task_type": "The primary task type (e.g., 'classification', 'detection', 'segmentation', 'regression', 'generation', 'embedding', 'optimization')",
    "description": "Brief description of what the method does and how it works (2-3 sentences)",
    "module_interface": "API/interface description if applicable (e.g., 'input: feature maps, output: refined features')",
    "dependencies": ["list of dependent methods or libraries if mentioned"]
  }}
]

## Guidelines
- Focus on methods that are technically reusable (architectures, layers, loss functions, optimization techniques)
- task_type should be one of: classification, detection, segmentation, regression, generation, embedding, optimization, augmentation, pooling, attention, fusion, encoding, decoding, regularization, sampling, clustering, ranking, retrieval, nlp, cv, rl, other
- Include only methods actually described in the paper, not generic techniques
- If no specific reusable methods are found, return: []
- Return ONLY the JSON array, no additional text"""

EXTRACT_METHODS_FULLTEXT_PROMPT = """You are a research paper analyst specializing in reusable technical methods.

Given a research paper full text, extract all reusable technical methods and modules
that could be applied to other research projects.

## Paper Title: {title}

## Full Text
{full_text}

## Output Format
Extract methods as a JSON array. Return ONLY the JSON array, no additional text.
[
  {{
    "method_name": "Name of the method (e.g., 'Squeeze-and-Excitation', 'Spatial Pyramid Pooling')",
    "task_type": "The primary task type (e.g., 'classification', 'detection', 'segmentation', 'regression', 'generation', 'embedding', 'optimization')",
    "description": "Brief description of what the method does and how it works (2-3 sentences)",
    "module_interface": "API/interface description if applicable (e.g., 'input: feature maps, output: refined features')",
    "dependencies": ["list of dependent methods or libraries if mentioned"]
  }}
]

## Guidelines
- Focus on methods that are technically reusable (architectures, layers, loss functions, optimization techniques)
- task_type should be one of: classification, detection, segmentation, regression, generation, embedding, optimization, augmentation, pooling, attention, fusion, encoding, decoding, regularization, sampling, clustering, ranking, retrieval, nlp, cv, rl, other
- Include only methods actually described in the paper, not generic techniques
- Look for method names in section titles, figures, and equations
- If no specific reusable methods are found, return: []
- Return ONLY the JSON array, no additional text"""


def _format_methods_result(methods: list[dict[str, Any]], paper_id: str) -> str:
    """Format extracted methods as a readable string."""
    if not methods:
        return f"No methods extracted from paper: {paper_id}"

    lines = [f"Extracted {len(methods)} method(s) from paper: {paper_id}\n"]

    for i, m in enumerate(methods, 1):
        lines.append(f"[{i}] {m.get('method_name', 'Unknown')}")
        lines.append(f"    Task Type: {m.get('task_type', 'N/A')}")
        lines.append(f"    Description: {m.get('description', 'N/A')}")
        if m.get('module_interface'):
            lines.append(f"    Interface: {m['module_interface']}")
        deps = m.get('dependencies', [])
        if deps:
            lines.append(f"    Dependencies: {', '.join(deps)}")
        lines.append("")

    return "\n".join(lines)


def _format_search_results(results: list[dict[str, Any]], query: str | None) -> str:
    """Format search results as a readable string."""
    if not results:
        return "No methods found" + (f" for query: {query}" if query else "")

    lines = [f"Found {len(results)} method(s)" + (f" for query: {query}" if query else "") + "\n"]

    for i, m in enumerate(results, 1):
        lines.append(f"[{i}] {m.get('method_name', 'Unknown')}")
        lines.append(f"    Task Type: {m.get('task_type', 'N/A')}")
        lines.append(f"    Description: {m.get('description', 'N/A')}")
        if m.get('module_interface'):
            lines.append(f"    Interface: {m['module_interface']}")
        deps = m.get('dependencies', [])
        if deps:
            lines.append(f"    Dependencies: {', '.join(deps)}")
        lines.append(f"    Paper ID: {m.get('paper_id', 'N/A')}")
        if m.get('vec_distance') is not None:
            lines.append(f"    Similarity: {1.0 / (1.0 + m['vec_distance']):.4f}")
        lines.append("")

    return "\n".join(lines)


class MethodExtractionTool(Tool):
    """Extract reusable technical methods from a research paper."""

    name = "method_extract"
    description = "Extract reusable technical methods and modules from a paper"

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID (e.g. '2401.12345' or '2401.12345v2')",
            },
            "fulltext": {
                "type": "string",
                "description": "Full text content of the paper (optional, will use abstract if not provided)",
            },
        },
        "required": ["paper_id"],
    }

    def __init__(
        self,
        provider: Any = None,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._provider = provider
        self._workspace = workspace
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._semantic_config is None:
            return None
        if self._search_index is None:
            self._search_index = SearchIndex(
                self._semantic_config.sqlite_db_path,
                self._semantic_config,
            )
        return self._search_index

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Try to load paper from local storage."""
        if not self._workspace:
            return None
        from pathlib import Path
        json_path = Path(self._workspace) / f"literature/papers/{paper_id}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    async def _extract_with_llm(
        self,
        abstract: str,
        fulltext: str | None = None,
        title: str = "",
    ) -> list[dict[str, Any]]:
        """Call LLM to extract methods from paper text."""
        if not self._provider:
            return []

        if fulltext:
            prompt = EXTRACT_METHODS_FULLTEXT_PROMPT.format(
                title=title or "Unknown",
                full_text=fulltext[:8000] if fulltext else "",
            )
        else:
            prompt = EXTRACT_METHODS_PROMPT.format(
                abstract=abstract[:2000] if abstract else "",
            )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._provider.chat_with_retry(messages)
            content = response.content or "[]"

            # Parse JSON array from response
            methods = extract_json_array(content)
            if methods is None:
                logger.warning("Failed to parse JSON from LLM response")
                return []

            # Validate and normalize each method
            validated = []
            for m in methods:
                if isinstance(m, dict) and m.get("method_name"):
                    validated.append({
                        "method_name": m.get("method_name", ""),
                        "task_type": m.get("task_type", "other"),
                        "description": m.get("description", ""),
                        "module_interface": m.get("module_interface", ""),
                        "dependencies": m.get("dependencies", []) if isinstance(m.get("dependencies"), list) else [],
                    })
            return validated
        except Exception as e:
            logger.warning("LLM method extraction failed: %s", e)

        return []

    async def execute(
        self,
        paper_id: str,
        fulltext: str | None = None,
        abstract: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not paper_id:
            return "Error: paper_id is required"

        # Try to load from local storage first
        local_paper = self._load_local_paper(paper_id)

        # Determine text sources
        use_fulltext = bool(fulltext)
        if not use_fulltext and local_paper:
            # Check for extracted text
            if local_paper.get("extracted_text_path"):
                from pathlib import Path
                extracted_path = Path(local_paper["extracted_text_path"])
                if extracted_path.exists():
                    with open(extracted_path, "r", encoding="utf-8") as f:
                        fulltext = f.read()
                    use_fulltext = True

        # Get abstract and title
        if local_paper:
            abstract = abstract or local_paper.get("abstract", "")
            title = local_paper.get("title", "")
        elif not abstract:
            return f"Error: Could not load paper {paper_id} locally and no abstract provided"
        else:
            title = ""

        # Extract methods via LLM
        methods = await self._extract_with_llm(
            abstract=abstract or "",
            fulltext=fulltext,
            title=title,
        )

        if not methods:
            return f"No methods extracted from paper: {paper_id}"

        # Store methods in search index
        search_index = self._get_search_index()
        stored_count = 0
        if search_index is not None:
            try:
                await search_index.initialize()
                now = datetime.now(timezone.utc).isoformat()

                for m in methods:
                    method_id = compute_short_id(f"{paper_id}:{m['method_name']}")
                    method_record = {
                        "id": method_id,
                        "paper_id": paper_id,
                        "method_name": m["method_name"],
                        "task_type": m["task_type"],
                        "description": m["description"],
                        "module_interface": m.get("module_interface", ""),
                        "dependencies": m.get("dependencies", []),
                        "extracted_at": now,
                    }
                    await search_index.upsert_method_with_vector(method_record)
                    stored_count += 1

                search_index.close()
            except Exception as e:
                logger.warning("Failed to store methods in search index: %s", e)

        result = _format_methods_result(methods, paper_id)
        if stored_count > 0:
            result += f"\n(Stored {stored_count} method(s) in search index)"

        return result


class MethodSearchTool(Tool):
    """Search the methods library for reusable techniques."""

    name = "method_search"
    description = "Search the methods library for reusable techniques"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query (e.g., 'attention mechanism for image classification')",
            },
            "task_type": {
                "type": "string",
                "description": "Filter by task type (e.g., 'classification', 'detection', 'segmentation', 'embedding')",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = workspace
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._semantic_config is None:
            return None
        if self._search_index is None:
            self._search_index = SearchIndex(
                self._semantic_config.sqlite_db_path,
                self._semantic_config,
            )
        return self._search_index

    async def execute(
        self,
        query: str | None = None,
        task_type: str | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> str:
        search_index = self._get_search_index()

        if search_index is None:
            return "Error: search index not configured"

        try:
            await search_index.initialize()
            results = search_index.search_methods(
                query=query,
                task_type=task_type,
                top_k=top_k,
            )
            search_index.close()
            return _format_search_results(results, query)
        except Exception as e:
            return f"Error searching methods: {e}"