"""Literature survey tool for agent."""

from __future__ import annotations

from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.literature_survey_pipeline import LiteratureSurveyPipeline
from researchbot.config.schema import LiteratureSurveyConfig


class LiteratureSurveyTool(Tool):
    """Conduct a comprehensive literature survey on a research topic."""

    name = "literature_survey"
    description = "Conduct a comprehensive literature survey on a research topic, including paper search, method extraction, gap discovery, and structured report generation."

    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Research topic to survey",
            },
            "depth": {
                "type": "string",
                "enum": ["light", "standard", "deep"],
                "description": "Survey depth",
                "default": "standard",
            },
            "max_papers": {
                "type": "integer",
                "description": "Maximum number of papers to analyze",
                "default": 30,
            },
            "save_to_local": {
                "type": "boolean",
                "description": "Save top papers to local database",
                "default": True,
            },
        },
        "required": ["topic"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        config: LiteratureSurveyConfig | None = None,
    ):
        self._workspace = workspace
        self._config = config or LiteratureSurveyConfig()
        self._pipeline = LiteratureSurveyPipeline(workspace, self._config)

    async def execute(
        self,
        topic: str,
        depth: str = "standard",
        max_papers: int = 30,
        save_to_local: bool = True,
        **kwargs: Any,
    ) -> str:
        report, topic_slug = await self._pipeline.execute(
            topic=topic,
            depth=depth,
            max_papers=max_papers,
            save_to_local=save_to_local,
        )
        return report