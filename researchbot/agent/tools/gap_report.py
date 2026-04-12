"""Interactive gap report generator."""

from __future__ import annotations

import json
from researchbot.agent.tools.gap import ResearchGap


class GapReport:
    """Generate interactive reports from research gaps."""

    def __init__(self, gaps: list[ResearchGap], topic: str, mode: str):
        self.gaps = gaps
        self.topic = topic
        self.mode = mode

    def to_markdown(self) -> str:
        """Convert gaps to markdown report."""
        lines = [
            f"# 研究空白发现报告\n",
            f"**主题：** {self.topic}",
            f"**分析模式：** {'基于收藏论文' if self.mode == 'collection' else '基于研究主题'}",
            f"**发现空白数：** {len(self.gaps)}\n",
            "---",
        ]

        for i, gap in enumerate(self.gaps, 1):
            gap_type_display = {
                "methodological": "方法空白",
                "application": "应用空白",
                "evaluation": "评估空白",
            }.get(gap.gap_type, gap.gap_type)

            lines.append(f"\n## {i}. {gap.title}")
            lines.append(f"\n**类型：** {gap_type_display}")
            lines.append(f"**优先级：** {gap.priority.upper()}")
            lines.append(f"**可信度：** {gap.confidence:.0%}\n")
            lines.append(f"### 描述\n{gap.description}\n")

            if gap.evidence:
                lines.append("### 证据链\n")
                lines.append("| 来源 | 原话 | 上下文 |")
                lines.append("|------|------|--------|")
                for e in gap.evidence:
                    lines.append(f"| {e.source} | {e.quote} | {e.context} |")
                lines.append("")

            if gap.potential_directions:
                lines.append("### 潜在研究方向\n")
                for direction in gap.potential_directions:
                    lines.append(f"- {direction}")
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Convert gaps to JSON."""
        return json.dumps(
            {
                "topic": self.topic,
                "mode": self.mode,
                "gaps": [g.to_dict() for g in self.gaps],
            },
            ensure_ascii=False,
            indent=2,
        )

    def query(self, question: str) -> dict:
        """Answer a question about the gaps."""
        question_lower = question.lower()

        if "机会点" in question or "opportun" in question_lower:
            return self._answer_opportunities()
        elif "入门" in question or "推荐论文" in question or "paper" in question_lower:
            return self._answer_papers()
        elif "解决方法" in question or "solution" in question_lower:
            return self._answer_solutions()
        elif "为什么" in question or "why" in question_lower or "priority" in question_lower:
            return self._answer_why_high_priority()
        else:
            return {
                "answer": "我只能回答以下问题：\n"
                         "- 这个方向有哪些具体的机会点？\n"
                         "- 推荐哪些论文作为入门？\n"
                         "- 这个空白对应的解决方法有哪些？\n"
                         "- 为什么说这是一个高优先级的研究空白？",
                "follow_ups": [],
            }

    def _answer_opportunities(self) -> dict:
        """Answer about specific opportunities."""
        all_directions = []
        for gap in self.gaps:
            all_directions.extend(gap.potential_directions)

        if all_directions:
            return {
                "answer": "具体机会点包括：\n" + "\n".join(f"- {d}" for d in all_directions[:5]),
                "follow_ups": ["推荐哪些论文作为入门？", "这个方向目前的SOTA是什么？"],
            }
        return {
            "answer": "当前报告中的潜在方向信息不足，建议使用 innovation_workflow 进行深度分析。",
            "follow_ups": [],
        }

    def _answer_papers(self) -> dict:
        """Recommend introductory papers."""
        papers = []
        for gap in self.gaps:
            for e in gap.evidence:
                papers.append(f"- **{e.source}** (ID: {e.source_id})")

        if papers:
            return {
                "answer": "推荐以下论文作为入门：\n" + "\n".join(papers[:5]),
                "follow_ups": ["这些论文的核心贡献是什么？", "还有其他相关论文吗？"],
            }
        return {"answer": "当前报告证据不足，无法推荐论文。", "follow_ups": []}

    def _answer_solutions(self) -> dict:
        """Answer about existing solutions."""
        return {
            "answer": "研究空白意味着目前没有成熟的解决方法。"
                     "潜在方向包括：\n"
                     "- 全新方法设计\n"
                     "- 现有方法改进\n"
                     "- 跨领域方法迁移",
            "follow_ups": ["有没有初步的探索性工作？", "这个方向的风险在哪里？"],
        }

    def _answer_why_high_priority(self) -> dict:
        """Explain why this is a high priority gap."""
        high_priority_gaps = [g for g in self.gaps if g.priority == "high"]
        if high_priority_gaps:
            reasons = []
            for gap in high_priority_gaps:
                reasons.append(f"- {gap.title}：{gap.description[:50]}...")
            return {
                "answer": "高优先级原因：\n" + "\n".join(reasons),
                "follow_ups": ["如何开展这个方向的研究？", "这个方向的风险有多大？"],
            }
        return {"answer": "当前没有标记为高优先级的空白。", "follow_ups": []}