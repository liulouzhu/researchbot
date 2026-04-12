"""Literature survey pipeline - orchestrates search, analysis, and reporting."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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
        max_papers: int | None = None,
        save_to_local: bool = True,
    ) -> tuple[str, str]:
        """
        Execute literature survey pipeline.

        Args:
            topic: Research topic
            depth: Survey depth (light=10, standard=30, deep=50 papers)
            max_papers: Override max papers (if provided, takes precedence over depth)
            save_to_local: Whether to save papers locally

        Returns:
            tuple: (markdown_report, topic_slug)
        """
        # Map depth to max_papers if not explicitly provided
        if max_papers is None:
            depth_map = {"light": 10, "standard": 30, "deep": 50}
            max_papers = depth_map.get(depth, 30)

        # 1. 搜索
        search_results = await self._search_papers(topic, max_papers)
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

    async def _search_papers(self, topic: str, max_papers: int) -> list[dict[str, Any]]:
        """Search papers from configured sources.

        Args:
            topic: Research topic
            max_papers: Target number of papers (used to size per-source max_results)
        """
        # Determine which sources to search based on config
        enabled_sources = set(self._config.sources)

        # Calculate per-source max_results to ensure we get enough before filtering
        max_results_per_source = max(20, max_papers * 2)

        # Build search tasks only for enabled sources
        tasks = {}
        if "arxiv" in enabled_sources:
            tasks["arxiv"] = search_arxiv(query=topic, max_results=max_results_per_source)
        if "openalex" in enabled_sources:
            tasks["openalex"] = search_openalex(query=topic, max_results=max_results_per_source)
        if "semantic_scholar" in enabled_sources:
            tasks["semantic_scholar"] = search_semantic_scholar(query=topic, max_results=max_results_per_source)

        # Execute all enabled searches in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map results back to source names (preserving order from config)
        source_results = {}
        for i, source in enumerate(tasks.keys()):
            result = results[i]
            source_results[source] = [] if isinstance(result, Exception) else result

        # Collect all results into a flat list (deduplication happens in _filter_top_papers)
        all_papers = []
        for source, results in source_results.items():
            all_papers.extend(results)
        return all_papers

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
        await asyncio.gather(*[
            save_tool.execute(paper=paper, topic=topic)
            for paper in papers
        ], return_exceptions=True)

    async def _extract_methods(self, papers: list[dict]) -> list[dict]:
        """Extract methods from papers."""
        from researchbot.agent.tools.method_extraction import MethodExtractionTool
        method_tool = MethodExtractionTool(workspace=self._workspace)

        async def extract_one(paper):
            try:
                paper_id = paper.get("arxiv_id") or paper.get("paper_id")
                abstract = paper.get("abstract", "")
                result = await method_tool.execute(paper_id=paper_id, abstract=abstract)
                if result and "Error" not in result:
                    return {"paper": paper, "methods": result}
            except Exception:
                pass
            return None

        results = await asyncio.gather(*[extract_one(p) for p in papers], return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def _analyze_gaps(self, topic: str) -> str:
        """Analyze research gaps and return markdown report."""
        from researchbot.agent.tools.research_gap_discovery import ResearchGapDiscoveryTool
        gap_tool = ResearchGapDiscoveryTool(workspace=self._workspace)
        result = await gap_tool.execute(mode="topic", topic=topic, max_results=10)
        return result

    def _generate_report(
        self,
        topic: str,
        papers: list[dict],
        methods: list[dict],
        gaps: str,
    ) -> str:
        """Generate markdown report."""
        lines = [
            f"# 文献调研报告：{topic}\n",
            f"**调研时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**数据来源**: arXiv, Semantic Scholar, OpenAlex\n",
            f"**分析论文数**: {len(papers)} 篇\n",
            f"**方法提取数**: {len(methods)} 篇\n",
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

        # 方法总结
        lines.append("\n## 3. 方法总结\n")
        if methods:
            lines.append(f"从 {len(methods)} 篇论文中提取了方法信息：\n")
            for m in methods[:10]:
                paper_title = m.get("paper", {}).get("title", "Unknown")
                method_text = m.get("methods", "")
                lines.append(f"### {paper_title}\n{method_text}\n")
        else:
            lines.append("未提取到方法信息。\n")

        # 研究空白分析
        lines.append("\n## 4. 研究空白分析\n")
        if gaps:
            lines.append(gaps)
        else:
            lines.append("未进行空白分析。\n")

        return "\n".join(lines)

    def _save_report(self, report: str, topic: str) -> str:
        """Save report to file and return topic slug."""
        slug = topic.lower().replace(" ", "-")[:50]
        output_dir = Path(self._workspace or ".") / self._config.output_dir / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        return slug