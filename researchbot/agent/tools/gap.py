"""Data models for research gap discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Evidence:
    """A piece of evidence supporting a research gap."""

    source: str  # 论文标题
    source_id: str  # 论文 ID
    quote: str  # 原话引用
    context: str  # 上下文


@dataclass
class ResearchGap:
    """A research gap identified from papers."""

    gap_type: Literal["methodological", "application", "evaluation"]
    title: str  # 空白标题
    description: str  # 详细描述
    evidence: list[Evidence] = field(default_factory=list)  # 证据链
    potential_directions: list[str] = field(default_factory=list)  # 潜在方向
    priority: Literal["high", "medium", "low"] = "medium"
    confidence: float = 0.5  # 0-1

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type,
            "title": self.title,
            "description": self.description,
            "evidence": [
                {"source": e.source, "source_id": e.source_id, "quote": e.quote, "context": e.context}
                for e in self.evidence
            ],
            "potential_directions": self.potential_directions,
            "priority": self.priority,
            "confidence": self.confidence,
        }