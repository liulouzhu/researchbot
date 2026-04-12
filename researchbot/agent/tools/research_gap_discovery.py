"""Research Gap Discovery tool."""

from __future__ import annotations

import asyncio
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.gap import ResearchGap
from researchbot.agent.tools.gap_analyzer import GapAnalyzer
from researchbot.agent.tools.gap_report import GapReport
from researchbot.config.schema import GapDiscoveryConfig


class ResearchGapDiscoveryTool(Tool):
    """Discover research gaps from papers or topics."""

    name = "research_gap_discovery"
    description = "Analyze papers to find unsolved problems and generate research direction suggestions."

    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["collection", "topic"],
                "description": "Analysis mode: collection (based on your papers) or topic (based on search query)",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paper IDs in your collection (for collection mode)",
            },
            "topic": {
                "type": "string",
                "description": "Research topic for analysis (for topic mode)",
            },
            "gap_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Types of gaps to analyze: methodological, application, evaluation",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of gaps to report",
                "default": 10,
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        config: GapDiscoveryConfig | None = None,
    ):
        self._workspace = workspace
        self._config = config or GapDiscoveryConfig()
        self._analyzer = GapAnalyzer(workspace, self._config)

    async def execute(
        self,
        mode: str,
        paper_ids: list[str] | None = None,
        topic: str | None = None,
        gap_types: list[str] | None = None,
        max_results: int = 10,
        **kwargs: Any,
    ) -> str:
        if mode == "collection":
            if not paper_ids:
                return "Error: paper_ids required for collection mode"
            gaps = await self._analyzer.analyze_papers(paper_ids)
            topic_str = f"收藏论文 ({len(paper_ids)} 篇)"
        elif mode == "topic":
            if not topic:
                return "Error: topic required for topic mode"
            gaps = await self._analyzer.analyze_topic(topic)
            topic_str = topic
        else:
            return f"Error: unknown mode {mode}"

        # 过滤 gap_types
        if gap_types:
            gaps = [g for g in gaps if g.gap_type in gap_types]

        # 限制数量
        gaps = gaps[:max_results]

        report = GapReport(gaps, topic_str, mode)
        return report.to_markdown()