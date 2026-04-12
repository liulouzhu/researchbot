"""Paper recommendation tool.

Provides two recommendation modes:
- collection: recommends papers based on co-citation analysis of user's collection
- topic: recommends papers by searching Semantic Scholar and OpenAlex by research topic
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.openalex_client import search_openalex
from researchbot.agent.tools.semantic_scholar_client import search_semantic_scholar
from researchbot.config.schema import RecommendationConfig, SemanticSearchConfig
from researchbot.search_index import SearchIndex


class PaperRecommendTool(Tool):
    """Recommend papers based on co-citation analysis or research topic.

    Mode 'collection': Given papers in your collection, finds foundational papers
        frequently cited together by your collection — via co-citation analysis.

    Mode 'topic': Searches Semantic Scholar and OpenAlex by research topic,
        merges and deduplicates results, and filters out papers already collected.
    """

    name = "paper_recommend"
    description = (
        "Recommend papers based on co-citation analysis of your collection or by research topic. "
        "Use 'collection' mode to find foundational papers cited by your collection. "
        "Use 'topic' mode to search external sources by research topic."
    )

    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["collection", "topic"],
                "description": "Recommendation mode: collection (based on your papers) or topic (based on search query)",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paper IDs in your collection (for collection mode)",
            },
            "topic": {
                "type": "string",
                "description": "Research topic for recommendation (for topic mode)",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of recommendations",
                "default": 10,
            },
            "year_from": {"type": "integer"},
            "year_to": {"type": "integer"},
            "exclude_collected": {
                "type": "boolean",
                "description": "Exclude papers already in your collection",
                "default": True,
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
        recommendation_config: RecommendationConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._recommendation_config = recommendation_config or RecommendationConfig()
        self._index: SearchIndex | None = None

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to workspace."""
        if self._workspace is None:
            return Path(relative_path)
        return self._workspace / relative_path

    def _get_graph(self):
        """Get or create the knowledge graph."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        self._index._ensure_graph_initialized()
        from researchbot.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(self._index)

    def _get_collected_paper_ids(self) -> set[str]:
        """Return IDs of all papers already in the local collection."""
        if self._index is None:
            return set()
        conn = self._index._get_conn()
        rows = conn.execute("SELECT paper_id FROM papers").fetchall()
        return {r["paper_id"] for r in rows}

    async def execute(
        self,
        mode: str,
        paper_ids: list[str] | None = None,
        topic: str | None = None,
        max_results: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        exclude_collected: bool = True,
        **kwargs: Any,
    ) -> str:
        """Execute paper recommendation."""
        if mode == "collection":
            if not paper_ids:
                return "Error: paper_ids is required for collection mode"
            return self._recommend_by_collection(paper_ids, max_results, year_from, year_to, exclude_collected)
        elif mode == "topic":
            if not topic:
                return "Error: topic is required for topic mode"
            return await self._recommend_by_topic(topic, max_results, year_from, year_to, exclude_collected)
        else:
            return f"Error: unknown mode '{mode}'. Use 'collection' or 'topic'."

    def _recommend_by_collection(
        self,
        paper_ids: list[str],
        max_results: int,
        year_from: int | None,
        year_to: int | None,
        exclude_collected: bool,
    ) -> str:
        """Recommend papers via co-citation analysis of user's collection."""
        try:
            kg = self._get_graph()
            recommendations = kg.recommend_cocited_papers(
                paper_ids,
                min_count=1,
                year_from=year_from,
                year_to=year_to,
                top_k=max_results * 2,
            )
        except Exception as e:
            return f"Co-citation recommendation error: {e}"

        if not recommendations:
            return (
                f"No co-citation recommendations found for the given collection (paper_ids={paper_ids}).\n"
                "Try providing more papers from your collection, or check if citation edges "
                "exist for these papers in the knowledge graph."
            )

        # Filter out collected papers if requested
        collected_ids: set[str] = set()
        if exclude_collected:
            collected_ids = self._get_collected_paper_ids()
            recommendations = [r for r in recommendations if r["paper_id"] not in collected_ids]

        recommendations = recommendations[:max_results]

        header = f"推荐论文（基于你的收藏）：\n共找到 {len(recommendations)} 篇推荐论文\n" if recommendations else "推荐论文（基于你的收藏）：\n"
        return header + self._format_recommendations(recommendations, "collection", collected_ids)

    async def _recommend_by_topic(
        self,
        topic: str,
        max_results: int,
        year_from: int | None,
        year_to: int | None,
        exclude_collected: bool,
    ) -> str:
        """Recommend papers by searching external sources via research topic."""
        # Parallel search
        ss_task = self._search_semantic_scholar(topic, max_results, year_from)
        oa_task = self._search_openalex(topic, max_results, year_from)

        ss_results, oa_results = await asyncio.gather(ss_task, oa_task)

        # Merge and deduplicate
        merged = self._merge_external_results(ss_results, oa_results)

        # Filter by year range
        if year_from is not None:
            merged = [r for r in merged if r.get("year", 0) and r["year"] >= year_from]
        if year_to is not None:
            merged = [r for r in merged if r.get("year", 0) and r["year"] <= year_to]

        # Filter out collected papers
        collected_ids: set[str] = set()
        if exclude_collected:
            collected_ids = self._get_collected_paper_ids()
            merged = [r for r in merged if r.get("paper_id", "") not in collected_ids]

        merged = merged[:max_results]

        header = f"推荐论文（基于主题「{topic}」）：\n共找到 {len(merged)} 篇推荐论文\n" if merged else f"推荐论文（基于主题「{topic}」）：\n"
        return header + self._format_recommendations(merged, "topic", collected_ids)

    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
        year: int | None,
    ) -> list[dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        try:
            results = await search_semantic_scholar(
                query=query,
                max_results=max_results,
                year=year,
            )
            return [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "authors": r.authors,
                    "year": r.year or 0,
                    "citation_count": r.citation_count,
                    "venue": r.venue,
                    "abstract": r.abstract,
                    "doi": r.doi,
                    "source": "Semantic Scholar",
                }
                for r in results
            ]
        except Exception:
            return []

    async def _search_openalex(
        self,
        query: str,
        max_results: int,
        year: int | None,
    ) -> list[dict[str, Any]]:
        """Search OpenAlex for papers."""
        try:
            results = await search_openalex(
                query=query,
                max_results=max_results,
                year=year,
            )
            return [
                {
                    "paper_id": r.id,
                    "title": r.title,
                    "authors": r.authors,
                    "year": int(r.year) if r.year else 0,
                    "citation_count": r.cited_by_count,
                    "venue": r.journal,
                    "abstract": r.abstract,
                    "doi": r.doi,
                    "source": "OpenAlex",
                }
                for r in results
            ]
        except Exception:
            return []

    def _merge_external_results(
        self,
        ss_results: list[dict[str, Any]],
        oa_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate results from Semantic Scholar and OpenAlex.

        Deduplication is done by DOI, then by title similarity.
        """
        merged: list[dict[str, Any]] = []
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()

        all_results = ss_results + oa_results
        for r in all_results:
            doi = r.get("doi", "").strip()
            title = r.get("title", "").strip().lower()

            # Skip if we've already seen this DOI (non-empty)
            if doi and doi in seen_dois:
                continue
            # Skip exact title duplicates
            if title and title in seen_titles:
                continue

            seen_dois.add(doi)
            seen_titles.add(title)
            merged.append(r)

        return merged

    def _format_recommendations(
        self,
        recommendations: list[dict[str, Any]],
        mode: str,
        collected_ids: set[str],
    ) -> str:
        """Format recommendation results for display."""
        if not recommendations:
            return "未找到符合条件的推荐论文。\n"

        lines = ["-" * 60]

        for i, rec in enumerate(recommendations, 1):
            paper_id = rec.get("paper_id", "?")
            title = rec.get("title", "?")
            authors = rec.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += f" et al. ({len(authors)} total)"
            else:
                authors_str = str(authors)
            year = rec.get("year", "?")
            citation_count = rec.get("citation_count", rec.get("cited_by_count", "?"))
            source = rec.get("source", "")

            lines.append(f"\n[{i}] {title}")
            lines.append(f"    {authors_str}")
            lines.append(f"    理由：{rec.get('explanation', '基于引用关系分析')}")
            lines.append(f"    年份：{year} | 引用数：{citation_count} | 来源：{source}")

        return "\n".join(lines)
