"""Extract evidence chains from papers for gap discovery."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from researchbot.agent.tools.gap import Evidence
from researchbot.agent.tools.openalex_client import search_openalex
from researchbot.agent.tools.semantic_scholar_client import search_semantic_scholar
from researchbot.config.schema import SemanticSearchConfig
from researchbot.search_index import SearchIndex

logger = logging.getLogger(__name__)


class EvidenceChain:
    """Extract evidence chains from papers."""

    def __init__(self, workspace: str | None = None):
        self._workspace = Path(workspace) if workspace else None
        self._index: SearchIndex | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Ensure index is initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    index = self._get_index()
                    await index.initialize()
                    self._initialized = True

    def _get_index(self) -> SearchIndex:
        """Get or create the search index."""
        if self._index is None:
            db_path = self._resolve_path(SemanticSearchConfig().sqlite_db_path)
            self._index = SearchIndex(db_path, SemanticSearchConfig())
        return self._index

    async def extract_from_papers(self, paper_ids: list[str]) -> list[Evidence]:
        """Extract evidence from a list of papers."""
        await self._ensure_initialized()
        index = self._get_index()

        evidence = []

        for paper_id in paper_ids:
            paper = index.get_paper(paper_id)
            if not paper:
                continue

            # 从 paper 的 abstract, intro, conclusion, future_work 字段提取
            text_fields = [
                paper.get("abstract", ""),
                paper.get("introduction", ""),
                paper.get("conclusion", ""),
                paper.get("future_work", ""),
            ]
            combined_text = " ".join(text_fields)

            indicators = self._extract_gap_indicators(combined_text)
            if indicators:
                evidence.append(
                    Evidence(
                        source=paper.get("title", "Unknown"),
                        source_id=paper_id,
                        quote=f"检测到指示词: {', '.join(indicators)}",
                        context="从论文引言/结论中提取",
                    )
                )

        index.close()
        return evidence

    async def extract_from_topic(self, topic: str, max_papers: int = 20) -> list[Evidence]:
        """Extract evidence from papers related to a topic."""
        evidence = []
        tasks = []

        tasks.append(self._search_and_extract("semantic_scholar", topic, max_papers))
        tasks.append(self._search_and_extract("openalex", topic, max_papers))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                evidence.extend(result)

        return evidence

    async def _search_and_extract(self, source: str, topic: str, max_papers: int) -> list[Evidence]:
        """Search one source and extract evidence."""
        try:
            if source == "semantic_scholar":
                papers = await search_semantic_scholar(query=topic, max_results=max_papers)
            elif source == "openalex":
                papers = await search_openalex(query=topic, max_results=max_papers)
            else:
                return []

            evidence = []
            for paper in papers:
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                indicators = self._extract_gap_indicators(text)
                if indicators:
                    evidence.append(
                        Evidence(
                            source=paper.get("title", "Unknown"),
                            source_id=paper.get("paper_id") or paper.get("id", ""),
                            quote=f"检测到指示词: {', '.join(indicators)}",
                            context=f"从 {source} 搜索结果提取",
                        )
                    )
            return evidence
        except Exception as e:
            logger.warning(f"Error searching {source} for topic '{topic}': {e}")
            return []

    def _extract_gap_indicators(self, text: str) -> list[str]:
        """从文本中提取研究空白指示词。"""
        indicators = [
            "未被解决",
            "尚未研究",
            "没有被探索",
            "没有方法",
            "现有方法无法",
            "挑战",
            "limitation",
            "future work",
            "remains open",
            "untouched",
            "underexplored",
            "understudied",
            "lack of",
            "insufficient",
            "no existing method",
            "unresolved",
        ]
        return [ind for ind in indicators if ind.lower() in text.lower()]

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve path relative to workspace."""
        p = Path(relative_path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p