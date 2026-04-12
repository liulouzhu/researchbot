"""Gap analyzer for identifying research gaps from papers."""

from __future__ import annotations

from researchbot.agent.tools.gap import ResearchGap, Evidence
from researchbot.agent.tools.evidence_chain import EvidenceChain
from researchbot.config.schema import GapDiscoveryConfig


class GapAnalyzer:
    """Analyze papers to identify research gaps."""

    GAP_TYPE_PATTERNS = {
        "methodological": [
            "方法", "method", "approach", "algorithm", "model",
            "无法", "cannot", "cannot handle", "limitation",
            "inefficient", "accuracy", "performance",
        ],
        "application": [
            "应用", "application", "used for", "applied to",
            "未被", "not been", "never", "很少有人",
            "understudied", "underexplored",
        ],
        "evaluation": [
            "评估", "evaluation", "benchmark", "dataset",
            "缺乏", "lack", "metric", "measure",
            "没有标准", "no standard", "insufficient",
        ],
    }

    def __init__(
        self,
        workspace: str | None = None,
        config: GapDiscoveryConfig | None = None,
    ):
        self._workspace = workspace
        self._config = config
        self._evidence_chain = EvidenceChain(workspace)

    async def analyze_papers(self, paper_ids: list[str]) -> list[ResearchGap]:
        """Analyze collected papers to find gaps."""
        evidence = await self._evidence_chain.extract_from_papers(paper_ids)
        gaps = self._cluster_and_classify(evidence)
        return gaps

    async def analyze_topic(self, topic: str) -> list[ResearchGap]:
        """Analyze a research topic to find gaps."""
        evidence = await self._evidence_chain.extract_from_topic(topic)
        gaps = self._cluster_and_classify(evidence)
        return gaps

    def _cluster_and_classify(self, evidence: list[Evidence]) -> list[ResearchGap]:
        """Cluster evidence and classify into gap types."""
        gaps = []
        for e in evidence:
            gap_type = self._classify_evidence(e)
            gaps.append(
                ResearchGap(
                    gap_type=gap_type,
                    title=f"从 {e.source} 识别的 {gap_type} 类型空白",
                    description=e.quote,
                    evidence=[e],
                    priority="medium",
                    confidence=0.6,
                )
            )
        return gaps

    def _classify_evidence(self, evidence: Evidence) -> str:
        """Classify evidence into a gap type."""
        text = f"{evidence.quote} {evidence.context}".lower()
        scores = {}
        for gap_type, patterns in self.GAP_TYPE_PATTERNS.items():
            score = sum(1 for p in patterns if p.lower() in text)
            scores[gap_type] = score
        if max(scores.values()) == 0:
            return "methodological"
        return max(scores, key=scores.get)