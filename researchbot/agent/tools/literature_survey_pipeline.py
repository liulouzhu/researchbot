"""Literature survey pipeline - orchestrates search, analysis, and reporting."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from researchbot.agent.tools.gap import ResearchGap
from researchbot.agent.tools.paper import PaperSearchTool
from researchbot.agent.tools.arxiv_client import search_arxiv
from researchbot.agent.tools.openalex_client import search_openalex
from researchbot.agent.tools.semantic_scholar_client import search_semantic_scholar
from researchbot.config.schema import LiteratureSurveyConfig

logger = logging.getLogger(__name__)


class LiteratureSurveyPipeline:
    """Execute literature survey pipeline."""

    def __init__(
        self,
        workspace: str | None = None,
        config: LiteratureSurveyConfig | None = None,
    ):
        self._workspace = workspace
        self._config = config or LiteratureSurveyConfig()
        self._search_tool = PaperSearchTool()

    async def execute(
        self,
        topic: str,
        depth: str = "standard",
        max_papers: int = 30,
        save_to_local: bool = True,
    ) -> tuple[str, str]:
        """
        Execute literature survey pipeline.

        Returns:
            tuple: (markdown_report, topic_slug)
        """
        # 1. 搜索
        search_results = await self._search_papers(topic)
        # 2. 筛选高引
        top_papers = self._filter_top_papers(search_results, max_papers)
        # 3. 保存到本地（可选）
        if save_to_local:
            await self._save_papers(top_papers, topic)
        # 4. 提取方法
        methods = await self._extract_methods(top_papers)
        # 5. 研究空白分析
        gaps = await self._analyze_gaps(topic)
        # 6. 生成报告
        report = self._generate_report(topic, top_papers, methods, gaps)
        # 7. 保存到文件
        topic_slug = self._save_report(report, topic)
        return (report, topic_slug)

    async def _search_papers(self, topic: str) -> list[dict[str, Any]]:
        """Search papers from all configured sources."""
        # 直接调用搜索客户端（避免通过 Tool 返回字符串）
        arxiv_task = search_arxiv(query=topic, max_results=20)
        openalex_task = search_openalex(query=topic, max_results=20)
        semantic_scholar_task = search_semantic_scholar(query=topic, max_results=20)

        arxiv_results, openalex_results, semantic_scholar_results = await asyncio.gather(
            arxiv_task, openalex_task, semantic_scholar_task, return_exceptions=True
        )

        # 使用 PaperSearchTool._aggregate_results 做 DOI 去重和合并
        aggregated = self._search_tool._aggregate_results(
            arxiv_results=[r for r in [arxiv_results] if not isinstance(r, Exception)],
            crossref_results=[],
            openalex_results=[r for r in [openalex_results] if not isinstance(r, Exception)],
            semantic_scholar_results=[r for r in [semantic_scholar_results] if not isinstance(r, Exception)],
        )
        return aggregated

    def _filter_top_papers(self, papers: list[dict], max_papers: int) -> list[dict]:
        """Filter top cited papers."""
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get("citations", 0) or 0,
            reverse=True,
        )
        return sorted_papers[:max_papers]

    async def _save_papers(self, papers: list[dict], topic: str) -> None:
        """Save papers to local database."""
        from researchbot.agent.tools.paper import PaperSaveTool
        save_tool = PaperSaveTool(workspace=self._workspace)
        for paper in papers:
            try:
                await save_tool.execute(paper=paper, topic=topic)
            except Exception as e:
                logger.warning(f"Failed to save paper {paper.get('title', 'unknown')}: {e}")

    async def _extract_methods(self, papers: list[dict]) -> list[dict]:
        """Extract methods from papers."""
        from researchbot.agent.tools.method_extraction import MethodExtractionTool
        method_tool = MethodExtractionTool(workspace=self._workspace)
        results = []
        for paper in papers:
            try:
                paper_id = paper.get("arxiv_id") or paper.get("paper_id")
                abstract = paper.get("abstract", "")
                result = await method_tool.execute(
                    paper_id=paper_id,
                    abstract=abstract,
                )
                if result and "Error" not in result:
                    results.append({"paper": paper, "methods": result})
            except Exception as e:
                logger.warning(f"Failed to extract methods from {paper.get('title', 'unknown')}: {e}")
        return results

    async def _analyze_gaps(self, topic: str) -> list[ResearchGap]:
        """Analyze research gaps."""
        from researchbot.agent.tools.research_gap_discovery import ResearchGapDiscoveryTool
        gap_tool = ResearchGapDiscoveryTool(workspace=self._workspace)
        result = await gap_tool.execute(mode="topic", topic=topic, max_results=10)
        return []  # GapReport 直接生成报告，这里返回空

    def _generate_report(
        self,
        topic: str,
        papers: list[dict],
        methods: list[dict],
        gaps: list[ResearchGap],
    ) -> str:
        """Generate markdown report."""
        lines = [
            f"# 文献调研报告：{topic}\n",
            f"**调研时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**数据来源**: arXiv, Semantic Scholar, OpenAlex\n",
            f"**分析论文数**: {len(papers)} 篇\n",
            "---\n",
        ]

        # 高引论文列表
        lines.append("## 2. 高引论文列表\n")
        for i, p in enumerate(papers[:30], 1):
            lines.append(f"{i}. **{p.get('title', 'Unknown')}**")
            authors = p.get("authors", [])
            if isinstance(authors, list) and authors:
                lines.append(f"   - 作者: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")
            lines.append(f"   - 年份: {p.get('year', 'N/A')} | 引用数: {p.get('citations', 0) or 0}\n")

        return "\n".join(lines)

    def _save_report(self, report: str, topic: str) -> str:
        """Save report to file and return topic slug."""
        slug = topic.lower().replace(" ", "-")[:50]
        output_dir = Path(self._workspace or ".") / self._config.output_dir / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        return slug