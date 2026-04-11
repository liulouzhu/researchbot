"""Innovation Point Workflow: generate candidates, novelty search, and review."""

from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.utils.helpers import utc_now

WORKFLOW_VERSION = "1.2.0"
PROMPT_VERSION = "1.2.0"

# =============================================================================
# Prompt Templates
# =============================================================================

INNOVATION_POINT_PROMPT = """You are a research idea generator. Given a research topic, problem description, or research direction, generate diverse candidate innovation points that cover different angles and innovation levels.

## Research Topic / Problem
{topic}

## Context from Existing Literature (if available)
{context}

## Diversity Requirements (CRITICAL):
1. Candidates must cover DIFFERENT innovation angles. Do NOT generate semantically similar ideas that only differ in wording.
2. Each candidate must state its "key_difference" - the specific aspect that differentiates it from existing work.
3. Cover different innovation levels:
   - Problem definition: new framing, new sub-problem, new task formulation
   - Method improvement: new algorithm, architecture change, optimization
   - System/setting: new dataset, new evaluation protocol, new domain
   - Experimental: new analysis angle, new tooling, new infrastructure
4. Prioritize ideas that are SPECIFIC and VERIFIABLE, not vague conceptual directions.
5. Every candidate should be actionable: a small team should be able to test the core claim in 6-12 months.

## Output Format
Generate 3-8 candidate innovation points. Each must have:
- title: Short descriptive title (max 20 words)
- problem: The specific problem or gap being addressed (1-2 sentences)
- idea: The core idea or approach (2-3 sentences), be specific about what exactly changes
- key_difference: The SPECIFIC aspect that differs from existing work (1-2 sentences, cite existing work if possible)
- expected_value: Potential impact or value (1-2 sentences)
- keywords: 3-5 keywords for search
- innovation_level: "problem" or "method" or "setting" or "experiment"

Please provide the output as a JSON array of innovation point objects.
Return ONLY the JSON array, no additional text."""

NOVELTY_ANALYSIS_PROMPT = """You are a research novelty analyst. Given an innovation point candidate and a list of related papers, determine the novelty by analyzing WHERE the overlap occurs.

## Innovation Point
Title: {title}
Problem: {problem}
Idea: {idea}
Keywords: {keywords}

## Related Papers Found
{papers_json}

## Novelty Analysis Guidelines (IMPORTANT):

### Overlap Dimensions - Determine WHERE the candidate overlaps with existing work:
- PROBLEM: Different problem being solved (good - low overlap)
- METHOD: Different approach/algorithm used (depends - some methods are similar in spirit)
- SETTING: Different dataset/domain/evaluation protocol (good - different context)
- EXPERIMENT: Different analysis angle/tooling/infrastructure (good - complementary)
- ASSUMPTION: Different initial assumptions or constraints (neutral - depends on validity)

### Distinction Between is_duplicate and is_highly_similar:
- is_duplicate = TRUE: The core idea (problem+method+setting) is essentially already done in a related paper
- is_highly_similar = TRUE: Core idea or setting highly overlapping but there is a small modification, different domain, or different scale
- BOTH should be FALSE if the candidate addresses a genuinely different aspect

### Evidence Requirements:
- For "similarities": Cite specific paper IDs and identify WHICH dimension overlaps (problem/method/setting/experiment/assumption)
- For "differences": Cite specific paper IDs and identify WHERE the candidate differs
- "novelty_conclusion" MUST cite evidence from the related papers. E.g.: "[paper_id] showed X, but our candidate does Y, which addresses gap Z"

### novelty_level Guidelines:
- HIGH: Different problem OR different method AND no existing work directly addresses the same gap
- MEDIUM: Some overlap in problem/method but candidate addresses a specific sub-case, combination, or underexplored angle
- LOW: Core idea is similar to multiple existing works, only differs in scale or minor modification

## Output Format
Provide a structured novelty analysis as JSON:
{{
  "overlap_dimensions": ["Which dimensions overlap: problem/method/setting/experiment/assumption"],
  "overlap_with_papers": ["For each overlap, cite paper_id and explain: e.g., [1234.5678] overlaps on METHOD - both use attention but different architecture"],
  "similarities": ["Specific similarities with existing work - cite paper IDs and dimension"],
  "differences": ["Specific differences from existing work - cite paper IDs and dimension"],
  "is_duplicate": true/false (true only if core idea is already done),
  "is_highly_similar": true/false (true if highly overlapping but with small modifications),
  "novelty_level": "high" or "medium" or "low",
  "novelty_conclusion": "2-3 sentence conclusion citing SPECIFIC evidence from related papers",
  "gaps_addressed": ["What gaps in existing work does this address, cite paper_ids if possible"]
}}

Return ONLY the JSON, no additional text."""

REVIEW_PROMPT = """You are a research innovation reviewer. Given an innovation point candidate with its novelty analysis and related papers, provide a structured evidence-based review with STRICT evidence requirements.

## Innovation Point Candidate
Title: {title}
Problem: {problem}
Idea: {idea}
Key Difference: {key_difference}
Expected Value: {expected_value}
Keywords: {keywords}
Innovation Level: {innovation_level}

## Novelty Analysis from Stage 2
Novelty Level: {novelty_level}
Overlap Dimensions: {overlap_dimensions}
Is Duplicate: {is_duplicate}
Is Highly Similar: {is_highly_similar}
Similarities: {similarities}
Differences: {differences}
Novelty Conclusion: {novelty_conclusion}
Gaps Addressed: {gaps_addressed}

## Related Papers (EVIDENCE SOURCE - cite paper_ids in all judgments)
{related_papers}

## Evidence Citation Requirements (MANDATORY):

1. **For EACH score dimension, you MUST cite specific paper_ids** that support the judgment:
   - novelty_score: Which papers show the gap? Which papers demonstrate the key_difference is novel?
   - feasibility_score: Which papers provide building blocks or proof-of-concept?
   - evidence_score: Which papers have theory, experiments, or validation supporting the approach?
   - impact_score: Which papers suggest this direction matters?
   - risk_score: Which papers identify the risks or challenges?

2. **reasoning field MUST contain**:
   - Which specific paper_ids you relied on for each judgment
   - Which overlap_dimensions (problem/method/setting/experiment/assumption) each paper relates to
   - What specifically about each paper supports or undermines the candidate

3. **If related papers lack sufficient detail** (e.g., only titles/abstracts, no experimental results):
   - You MUST acknowledge this limitation explicitly in reasoning
   - You MUST lower evidence_score accordingly (max 4 if only titles/abstracts, max 6 if abstracts only)
   - You MUST NOT fabricate evidence that is not present in the papers

4. **If evidence is insufficient for a claim**:
   - DO NOT claim "strong evidence" or "validated approach"
   - Use conservative language: "limited evidence", "partial support", "suggested but not proven"

## Scoring Rules (STRICT):

### novelty_score (1-10):
- HIGH novelty must be supported by specific papers showing the gap
- If key_difference is not clearly supported by cited papers: cap at 7
- If only assertion without paper evidence: cap at 6

### feasibility_score (1-10):
- Can a small team implement this in 6-12 months?
- If requires new dataset collection: score 1-3
- If requires massive compute or unreleased models: score 1-4
- If requires significant domain expertise outside your team: score 2-5
- If builds on existing tools/frameworks with clear implementation path: score 7-10
- Cite which papers provide the building blocks

### evidence_score (1-10) - MOST CRITICAL:
- 1-3: Only intuition, no supporting evidence from papers
- 4-5: Related works partially validate approach but lack direct evidence
- 6-7: Some theoretical foundation OR preliminary evidence in cited papers
- 8-10: Strong theory AND/OR validated components in multiple cited papers
- If you have only titles/abstracts: score <= 4 (insufficient for strong evidence claims)
- If papers confirm the approach works in adjacent settings: max 6
- NEVER score above 6 unless papers contain actual experiments, benchmarks, or theory

### impact_score (1-10):
- Incremental improvement: 1-4
- Enables new capabilities: 5-7
- Field-changing or foundational: 8-10
- Cite which papers suggest this direction matters

### risk_score (1-10) - higher = MORE RISK:
- Technical risk: Is the method actually implementable?
- Resource risk: What specific resources are needed?
- Competition risk: Are others close to this?
- Time risk: Is the timeline realistic?
- Novelty without feasibility = very high risk (8-10)

## Decision Rules (STRICT - follow EXACTLY):

**"proceed" ONLY when ALL of these are true:**
- evidence_score >= 6 (NOT 5 - must have real evidence)
- feasibility_score >= 5
- risk_score <= 4 (NOT 5 - must be low risk)
- is_duplicate == false
- You MUST cite specific paper_ids in reasoning to justify "proceed"

**"revise" when:**
- Potential exists but evidence_score < 6 OR feasibility_score < 5 OR risk_score > 4
- OR novelty is high but claims need stronger evidence
- You MUST specify exactly what revision is needed

**"drop" when ANY of:**
- feasibility_score <= 3
- evidence_score <= 3
- risk_score >= 7
- is_duplicate == true
- OR reasoning cannot cite ANY supporting papers

## Output Format
Provide a structured review as JSON:
{{
  "novelty_score": 1-10 integer,
  "feasibility_score": 1-10 integer,
  "evidence_score": 1-10 integer (MUST reflect actual evidence quality),
  "impact_score": 1-10 integer,
  "risk_score": 1-10 integer,
  "decision": "proceed" or "revise" or "drop",
  "reasoning": "3-5 sentence explanation that explicitly cites paper_ids and explains which papers/dimensions support each judgment. If evidence is thin, say so explicitly.",
  "main_risks": ["specific risk 1 with paper citation", "specific risk 2 with paper citation", "specific risk 3"],
  "recommended_revision": "If decision is 'revise', exact modifications needed, otherwise empty string",
  "next_step": "Concrete next step: 'find paper showing X', 'validate assumption Y with experiment', 'narrow scope to Z', etc."
}}

CRITICAL: Your reasoning MUST cite specific paper_ids in brackets like [paper_id] for every major claim. If no papers support a claim, say "no supporting paper found" and lower scores accordingly.

Return ONLY the JSON, no additional text."""

EXTERNAL_REVIEWER_CANDIDATES_PROMPT = """You are an external research reviewer. After the executor generated initial innovation candidates, give brief critical commentary on each one.

## Research Topic
{topic}

## Executor-Generated Candidates
{candidates_json}

## Your Task
For each candidate, provide:
1. One-sentence assessment of the idea's potential
2. One specific weakness or risk (be direct, not vague)
3. A suggestion for how to strengthen it

Be concise. No need to re-score — just provide directional feedback.
Return as JSON: {{"reviews": [{{"candidate_title": "...", "assessment": "...", "weakness": "...", "suggestion": "..."}}]}}
Return ONLY the JSON."""

EXTERNAL_REVIEWER_REVIEW_PROMPT = """You are an external research reviewer. Given the executor's review scores for innovation candidates, provide your own independent assessment.

## Research Topic
{topic}

## Executor Review Results
{review_report_summary}

## Innovation Candidates (for reference)
{candidates_json}

## Your Task
For each candidate, give:
1. Whether you agree with the executor's scores (concordant or discordant)
2. Your own brief reasoning (1-2 sentences) — cite specific concerns
3. A final recommendation: PROCEED / REVISE / DROP

Be honest about disagreements. Concordant candidates need less explanation; discordant ones need more.
Return as JSON: {{"external_reviews": [{{"candidate_title": "...", "agreement": "concordant|discordant", "reasoning": "...", "recommendation": "PROCEED|REVISE|DROP"}}]}}
Return ONLY the JSON."""

REVISION_PROMPT = """You are a research idea refiner. Given an innovation point candidate and its review feedback, produce a revised version that addresses the identified weaknesses.

## Original Candidate
Title: {title}
Problem: {problem}
Idea: {idea}
Key Difference: {key_difference}
Expected Value: {expected_value}
Keywords: {keywords}
Innovation Level: {innovation_level}

## Review Feedback
Decision: {decision}
Overall Score: {overall_score}/10
Novelty Score: {novelty_score}/10
Feasibility Score: {feasibility_score}/10
Evidence Score: {evidence_score}/10
Impact Score: {impact_score}/10
Risk Score: {risk_score}/10
Reasoning: {reasoning}
Main Risks: {main_risks}
Recommended Revision: {recommended_revision}
Next Step: {next_step}

## Novelty Analysis
Novelty Level: {novelty_level}
Overlap Dimensions: {overlap_dimensions}
Similarities: {similarities}
Differences: {differences}
Novelty Conclusion: {novelty_conclusion}

## Related Papers (for context)
{related_papers}

## Revision Guidelines (MANDATORY):

Your task is to produce a REVISED candidate that addresses the specific weaknesses identified in the review.

### What you MUST do:
1. Address EACH main risk mentioned in the review feedback
2. Narrow scope if the original was too ambitious (reduce scope to fit 6-12 month timeline)
3. Strengthen the evidence basis if evidence_score was low
4. Make the idea more specific and actionable
5. If feasibility was low, simplify the approach or leverage more existing tools
6. If risk was high, add mitigations or reduce the risk profile
7. Preserve what makes the candidate novel - do not abandon the core innovation
8. **Generate new keywords that reflect the revised idea** - the revision may shift the focus, so keywords must accurately describe the new direction for subsequent novelty search. Extract keywords from the revised title, problem, and idea, not just copied from the original.

### What you MUST NOT do:
- Do not just rephrase the same idea
- Do not make the idea vaguer or more general
- Do not ignore the review feedback
- Do not add unimplementable features

### Key Difference from original:
- State what specifically changed and WHY it is an improvement
- Cite which specific risk or weakness you are addressing

### Output Format
Return a revised candidate as JSON with these extra fields:
{{
  "title": "Revised short title (max 20 words)",
  "problem": "Specific problem addressed (1-2 sentences)",
  "idea": "Core idea (2-3 sentences, more specific than original)",
  "key_difference": "What changed from original and why it is better (1-2 sentences)",
  "expected_value": "Potential impact (1-2 sentences)",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "innovation_level": "problem" or "method" or "setting" or "experiment",
  "parent_candidate": "Original title of parent candidate",
  "revision_round": {round_number},
  "revision_reason": "Which specific weakness(es) from the review this revision addresses",
  "revision_summary": "2-3 sentence summary of what changed and why"
}}

Return ONLY the JSON, no additional text."""


LANDSCAPE_MAP_PROMPT = """You are a research landscape analyst. Given a set of papers related to a research topic, construct a structured literature map that categorizes the research landscape.

## Research Topic
{topic}

## Papers Found ({num_papers} total)
{papers_text}

## Task
Analyze these papers and construct a literature map. Group papers into 3-6 thematic categories that represent distinct research directions or approaches within this topic.

For EACH category, provide:
- name: Short category name (2-5 words)
- description: One-sentence description of this research direction
- representative_papers: List of 2-4 paper titles that best represent this direction (cite titles exactly as given)
- main_problems: 1-2 main research problems this direction addresses
- typical_methods: 1-2 typical methods/approaches used
- covered_gaps: What aspects of the topic this direction covers well
- limitations: 1-2 explicit limitations or weaknesses of this direction
- exploration_opportunities: 1-2 specific opportunities for further research or unexplored sub-problems

Also provide an overall summary:
- total_papers_analyzed: number of papers
- categories_count: number of categories
- overall_trends: 2-3 sentences on major trends observed
- key_gaps: 2-3 specific research gaps that cut across categories
- recommended_exploration: 2-3 specific directions worth exploring that are underrepresented

## Output Format
Return a JSON object with this structure:
{{
  "categories": [
    {{
      "name": "...",
      "description": "...",
      "representative_papers": ["title1", "title2"],
      "main_problems": ["..."],
      "typical_methods": ["..."],
      "covered_gaps": ["..."],
      "limitations": ["..."],
      "exploration_opportunities": ["..."]
    }}
  ],
  "summary": {{
    "total_papers_analyzed": ...,
    "categories_count": ...,
    "overall_trends": "...",
    "key_gaps": ["...", "..."],
    "recommended_exploration": ["...", "..."]
  }}
}}

Return ONLY the JSON object, no additional text."""


# Pre-compiled patterns used across multiple functions
_TITLE_KEY_RE = re.compile(r"[^a-z0-9]")


# =============================================================================
# Utility Functions
# =============================================================================

def _slugify(text: str) -> str:
    """Create a URL-safe slug from text."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:60]


def _format_papers_for_prompt(papers: list[dict[str, Any]]) -> str:
    """Format papers for the prompt."""
    if not papers:
        return "No related papers found."

    parts = []
    for p in papers:
        pid = p.get("paper_id", "?")
        title = p.get("title", "?")
        abstract = p.get("abstract", "")
        if isinstance(abstract, dict):
            abstract = abstract.get("problem", "")
        summary = p.get("summary", {})
        if isinstance(summary, dict):
            problem = summary.get("problem", "")
            method = summary.get("method", "")
            abstract = f"Problem: {problem}\nMethod: {method}"

        parts.append(f"[{pid}] {title}\nAbstract: {abstract[:300]}...")
    return "\n\n".join(parts)


def _parse_json_robust(content: str, expect_array: bool = True) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Robustly parse JSON from LLM output.

    Handles:
    - Fenced code blocks (```json ... ```)
    - JSON embedded in explanatory text
    - Multiple JSON objects (returns first valid one)
    - Leading/trailing whitespace and BOM
    """
    if not content:
        return None

    content = content.lstrip('\ufeff').strip()

    # Strategy 1: Extract from fenced code blocks
    fences = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    for fence in fences:
        try:
            result = json.loads(fence.strip())
            if expect_array and isinstance(result, list):
                return result
            elif not expect_array and isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    # Strategy 2: Find bracket-enclosed arrays or objects
    if expect_array:
        for match in re.finditer(r'\[\s*\{', content):
            start = match.start()
            depth = 0
            for i, ch in enumerate(content[start:], start):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(content[start:i+1])
                            if isinstance(result, list):
                                return result
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
    else:
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content):
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError):
                continue

    # Strategy 3: Fallback to simple bracket matching
    if expect_array:
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start >= 0 and end > start:
                result = json.loads(content[start:end+1])
                if isinstance(result, list):
                    return result
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start >= 0 and end > start:
                result = json.loads(content[start:end+1])
                if isinstance(result, dict):
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _format_json_as_markdown(data: dict[str, Any]) -> str:
    """Format a dict as readable markdown."""
    lines = []
    if "reviews" in data:
        for r in data["reviews"]:
            lines.append(f"### {r.get('candidate_title', 'Unknown')}")
            lines.append(f"- **Assessment**: {r.get('assessment', 'N/A')}")
            lines.append(f"- **Weakness**: {r.get('weakness', 'N/A')}")
            lines.append(f"- **Suggestion**: {r.get('suggestion', 'N/A')}")
            lines.append("")
    elif "external_reviews" in data:
        for r in data["external_reviews"]:
            lines.append(f"### {r.get('candidate_title', 'Unknown')}")
            lines.append(f"- **Agreement**: {r.get('agreement', 'N/A')}")
            lines.append(f"- **Reasoning**: {r.get('reasoning', 'N/A')}")
            lines.append(f"- **Recommendation**: {r.get('recommendation', 'N/A')}")
            lines.append("")
    return "\n".join(lines) if lines else str(data)


def _summarize_review_report(data: dict[str, Any]) -> str:
    """Extract a compact summary of the review report for the external reviewer."""
    results = data.get("results", [])
    lines = [f"Total candidates reviewed: {len(results)}\n"]
    for r in results[:10]:
        lines.append(f"### {r.get('title', r.get('candidate_title', 'Unknown'))}")
        lines.append(f"novelty={r.get('novelty_score', 'N/A')}, feasibility={r.get('feasibility_score', 'N/A')}, "
                     f"evidence={r.get('evidence_score', 'N/A')}, impact={r.get('impact_score', 'N/A')}, "
                     f"risk={r.get('risk_score', 'N/A')}, overall={r.get('overall_score', 'N/A')}")
        lines.append(f"Decision: {r.get('decision', 'N/A')}")
        reasoning = r.get('reasoning', 'N/A')
        if isinstance(reasoning, str):
            reasoning = reasoning[:200]
        lines.append(f"Reasoning: {reasoning}")
        lines.append("")
    return "\n".join(lines)


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a candidate innovation point to standard schema.

    Handles both old format (without key_difference/innovation_level) and new format.
    """
    def _ensure_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        return []

    innovation_level = str(raw.get("innovation_level", "unknown")).lower().strip()
    if innovation_level not in ("problem", "method", "setting", "experiment", "unknown"):
        innovation_level = "unknown"

    return {
        "title": str(raw.get("title", "Untitled"))[:200],
        "problem": str(raw.get("problem", "")),
        "idea": str(raw.get("idea", "")),
        "novelty_claim": str(raw.get("novelty_claim", "")),
        "key_difference": str(raw.get("key_difference", "")),
        "expected_value": str(raw.get("expected_value", "")),
        "keywords": _ensure_list(raw.get("keywords", []))[:10],
        "innovation_level": innovation_level,
        "related_work_hypothesis": str(raw.get("related_work_hypothesis", "")),
    }


def _candidate_signature(candidate: dict[str, Any]) -> str:
    """Build a compact text signature for similarity checks."""
    keywords = candidate.get("keywords", [])
    if isinstance(keywords, list):
        keywords_text = " ".join(str(k) for k in keywords)
    else:
        keywords_text = str(keywords)
    parts = [
        candidate.get("title", ""),
        candidate.get("problem", ""),
        candidate.get("idea", ""),
        candidate.get("key_difference", ""),
        keywords_text,
    ]
    return " ".join(str(p) for p in parts if p).strip().lower()


def _tokenize_signature(signature: str) -> set[str]:
    """Tokenize a candidate signature for coarse Jaccard similarity."""
    text = re.sub(r"\s+", "", signature.lower())
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    tokens_set = {tok for tok in tokens if len(tok.strip()) > 1}

    # Add character bigrams so short Chinese phrases still register as similar.
    for i in range(max(0, len(text) - 1)):
        gram = text[i:i + 2]
        if len(gram.strip()) > 1:
            tokens_set.add(gram)

    return tokens_set


def _candidate_similarity(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> float:
    """Compute a coarse lexical similarity score between two candidates."""
    title_a = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(candidate_a.get("title", "")).lower())
    title_b = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(candidate_b.get("title", "")).lower())
    if title_a and title_b and (title_a in title_b or title_b in title_a):
        return 1.0

    tokens_a = _tokenize_signature(_candidate_signature(candidate_a))
    tokens_b = _tokenize_signature(_candidate_signature(candidate_b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


def _filter_diverse_candidates(
    candidates: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.58,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove near-duplicate candidates while preserving the first occurrence."""
    unique: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        if any(_candidate_similarity(candidate, existing) >= similarity_threshold for existing in unique):
            rejected.append(candidate)
        else:
            unique.append(candidate)

    return unique, rejected


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Safely parse a boolean value from various types.

    Handles: bool, int (0/1), str ("true"/"false", "yes"/"no", "1"/"0", etc.).
    Python's bool() on non-empty strings always returns True, so this is needed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off", ""):
            return False
    return default


def _normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a novelty analysis result to standard schema.

    Handles both old format (without overlap_dimensions) and new format.
    """
    def _ensure_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        if isinstance(value, str):
            return [v.strip() for v in value.split('\n') if v.strip()]
        return []

    novelty_level = str(raw.get("novelty_level", "unknown")).lower().strip()
    if novelty_level not in ("high", "medium", "low", "unknown"):
        novelty_level = "unknown"

    return {
        "overlap_dimensions": _ensure_list(raw.get("overlap_dimensions", [])),
        "overlap_with_papers": _ensure_list(raw.get("overlap_with_papers", [])),
        "similarities": _ensure_list(raw.get("similarities", [])),
        "differences": _ensure_list(raw.get("differences", [])),
        "is_duplicate": _parse_bool(raw.get("is_duplicate"), False),
        "is_highly_similar": _parse_bool(raw.get("is_highly_similar"), False),
        "novelty_level": novelty_level,
        "novelty_conclusion": str(raw.get("novelty_conclusion", "")),
        "gaps_addressed": _ensure_list(raw.get("gaps_addressed", [])),
    }


def _normalize_review(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a review result to standard schema with computed overall_score."""
    def _ensure_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        if isinstance(value, str):
            lines = [l.strip() for l in value.split('\n') if l.strip()]
            # Strip bullet points
            return [re.sub(r'^[\-\*\d\.]+\s*', '', l).strip() for l in lines if l.strip()]
        return []

    def _clamp(value: Any, default: int) -> int:
        try:
            v = int(value)
            return max(1, min(10, v))
        except (TypeError, ValueError):
            return default

    novelty = _clamp(raw.get("novelty_score"), 5)
    feasibility = _clamp(raw.get("feasibility_score"), 5)
    evidence = _clamp(raw.get("evidence_score"), 5)
    impact = _clamp(raw.get("impact_score"), 5)
    risk = _clamp(raw.get("risk_score"), 5)

    # overall_score: weighted average emphasizing feasibility and risk
    # Higher risk = lower overall. Higher novelty/feasibility/evidence/impact = higher overall.
    # Formula: (novelty*0.2 + feasibility*0.25 + evidence*0.15 + impact*0.2 + (10-risk)*0.2) / 1.0
    overall = round(
        novelty * 0.20 +
        feasibility * 0.25 +
        evidence * 0.15 +
        impact * 0.20 +
        (10 - risk) * 0.20
    )

    decision = str(raw.get("decision", "")).lower().strip()
    if decision not in ("proceed", "revise", "drop"):
        # Infer from scores if decision missing
        if risk >= 8 or feasibility <= 2:
            decision = "drop"
        elif novelty <= 3 and feasibility >= 7:
            decision = "revise"
        elif overall >= 6 and risk <= 5:
            decision = "proceed"
        elif overall >= 4:
            decision = "revise"
        else:
            decision = "drop"

    return {
        "novelty_score": novelty,
        "feasibility_score": feasibility,
        "evidence_score": evidence,
        "impact_score": impact,
        "risk_score": risk,
        "overall_score": max(1, min(10, overall)),
        "decision": decision,
        "reasoning": str(raw.get("reasoning", "")),
        "main_risks": _ensure_list(raw.get("main_risks", []))[:5],
        "recommended_revision": str(raw.get("recommended_revision", "")),
        "next_step": str(raw.get("next_step", "")),
    }


# =============================================================================
# Workflow Metadata Helpers
# =============================================================================

def _workflow_info() -> dict[str, Any]:
    """Build the initial workflow metadata structure."""
    return {
        "workflow_version": WORKFLOW_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "topic": None,
        "topic_slug": None,
        "params": {},
        "stages": {
            "stage0_landscape": {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "input_params": {},
                "output_files": {},
                "num_local_papers": 0,
                "num_online_papers": 0,
                "num_categories": 0,
                "error": None,
            },
            "stage1_generate": {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "input_params": {},
                "output_files": {},
                "num_candidates": 0,
                "error": None,
            },
            "stage2_novelty": {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "input_params": {},
                "output_files": {},
                "num_candidates": 0,
                "error": None,
            },
            "stage3_review": {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "input_params": {},
                "output_files": {},
                "num_candidates": 0,
                "error": None,
            },
        },
        "overall_status": "running",
    }


def _save_workflow_metadata(
    metadata: dict[str, Any],
    workspace: Path | None,
    topic: str,
) -> str:
    """Save workflow metadata to workflow.json and return its path."""
    slug = _slugify(topic)
    innovation_dir = (workspace / f"innovation/{slug}") if workspace else Path(f"innovation/{slug}")
    innovation_dir.mkdir(parents=True, exist_ok=True)
    json_path = innovation_dir / "workflow.json"
    metadata["topic"] = topic
    metadata["topic_slug"] = slug
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(json_path)


# =============================================================================
# Iteration Helper Functions
# =============================================================================

def _format_related_papers_for_revision(papers: list[dict[str, Any]], max_papers: int = 5) -> str:
    """Format related papers for the revision prompt."""
    if not papers:
        return "No related papers found."
    lines = []
    for p in papers[:max_papers]:
        pid = p.get("paper_id", "?")
        title = p.get("title", "?")
        year = p.get("year", "") or (p.get("published", "")[:4] if p.get("published") else "")
        authors = p.get("authors", [])
        if isinstance(authors, list) and authors:
            authors_str = ", ".join(authors[:2])
            if len(authors) > 2:
                authors_str += " et al."
        else:
            authors_str = "Unknown"
        abstract = p.get("abstract", "")
        if isinstance(abstract, dict):
            abstract = abstract.get("problem", "") or abstract.get("method", "")
        if abstract:
            abstract = str(abstract)[:300]
        else:
            abstract = "No abstract available."
        lines.append(f"[{pid}] {title} ({year}) - {authors_str}")
        lines.append(f"    Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def _extract_keywords_from_text(text: str, max_keywords: int = 5) -> list[str]:
    """Extract representative keywords from text content.

    Used as fallback when a candidate (especially a revised one) doesn't
    provide explicit keywords.
    """
    # Common stop words to filter out
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "when", "where", "why", "how",
        "for", "with", "without", "from", "into", "through", "during",
        "new", "different", "specific", "approach", "method", "using",
        "based", "method", "problem", "research", "paper", "work",
    }

    # Split into words, filter short words and stop words
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    filtered = [w for w in words if w not in stop_words]

    # Count word frequency
    word_counts = Counter(filtered)

    # Get most common words as keywords
    common = word_counts.most_common(max_keywords * 2)
    keywords = []
    seen_stems: set[str] = set()

    for word, count in common:
        # Avoid near-duplicates (same stem)
        stem = word[:4]
        if stem not in seen_stems and len(keywords) < max_keywords:
            keywords.append(word)
            seen_stems.add(stem)

    return keywords


def _normalize_revised_candidate(
    raw: dict[str, Any],
    parent_title: str,
    round_num: int,
    revision_reason: str,
) -> dict[str, Any]:
    """Normalize a revised candidate, filling in revision tracking fields."""
    base = _normalize_candidate(raw)

    # Fallback: if keywords are empty (LLM didn't generate them), extract from content
    if not base.get("keywords"):
        content_text = " ".join([
            base.get("title", ""),
            base.get("problem", ""),
            base.get("idea", ""),
            base.get("key_difference", ""),
        ])
        base["keywords"] = _extract_keywords_from_text(content_text)

    base["parent_candidate"] = parent_title
    base["revision_round"] = round_num
    base["revision_reason"] = revision_reason
    base["revision_summary"] = str(raw.get("revision_summary", ""))
    return base


def _classify_candidates(
    review_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify review results into proceed/revise/drop buckets."""
    proceed = []
    revise = []
    drop = []
    for r in review_results:
        decision = r.get("review", {}).get("decision", "drop").lower()
        if decision == "proceed":
            proceed.append(r)
        elif decision == "revise":
            revise.append(r)
        else:
            drop.append(r)
    return proceed, revise, drop


# =============================================================================
# Innovation Workflow Tool
# =============================================================================

class InnovationWorkflowTool(Tool):
    """Generate innovation point candidates, perform novelty search, and review.

    Three-stage workflow with optional multi-round iteration:
    1. Generate candidate innovation points from a research topic
    2. For each candidate, search for related work and assess novelty
    3. Review and score each candidate, produce final recommendations
    4. (If enable_iteration=True) Revise "revise" candidates and iterate until convergence

    Supports partial re-runs: if output files already exist (and overwrite=False),
    previously completed stages are skipped and their results are reused.

    Outputs:
    - innovation/<topic>/workflow.json (always, tracks run metadata and stage status)
    - innovation/<topic>/candidates.json / .md (initial generation)
    - innovation/<topic>/novelty_report.json / .md
    - innovation/<topic>/review_report.json / .md
    - innovation/<topic>/iterations/round_N/ (if enable_iteration=True)
    - innovation/<topic>/iteration_report.json / .md (if enable_iteration=True)
    """

    name = "innovation_workflow"
    description = (
        "Generate innovation point candidates from a research topic, perform novelty search, "
        "and review/score candidates. Takes a topic/problem, generates candidates, searches "
        "for related work, assesses novelty, reviews/scores each candidate, and optionally "
        "iterates on revise candidates across multiple rounds. Use overwrite=True when you want "
        "a fresh batch instead of reusing cached results. "
        "Outputs to innovation/<topic>/ directory."
    )

    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Research topic, problem description, or research direction",
            },
            "num_candidates": {
                "type": "integer",
                "description": "Number of candidates to generate (3-8, default: 5)",
                "minimum": 3,
                "maximum": 8,
                "default": 5,
            },
            "search_local": {
                "type": "boolean",
                "description": "Whether to search local literature first (default: true)",
                "default": True,
            },
            "search_online": {
                "type": "boolean",
                "description": "Whether to search online (arXiv) for related work (default: true)",
                "default": True,
            },
            "max_related": {
                "type": "integer",
                "description": "Maximum related papers to find per candidate (default: 5)",
                "default": 5,
            },
            "enable_review": {
                "type": "boolean",
                "description": "Whether to enable review/scoring stage (default: true)",
                "default": True,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top candidates to recommend (default: 3)",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite existing results (default: false)",
                "default": False,
            },
            "enable_iteration": {
                "type": "boolean",
                "description": "Enable multi-round iteration: revise candidates and re-review (default: false)",
                "default": False,
            },
            "max_rounds": {
                "type": "integer",
                "description": "Maximum iteration rounds when enable_iteration=True (default: 2)",
                "minimum": 1,
                "maximum": 5,
                "default": 2,
            },
            "min_proceed": {
                "type": "integer",
                "description": "Stop iterating when this many 'proceed' candidates are found (default: 1)",
                "minimum": 1,
                "default": 1,
            },
            "revise_top_k": {
                "type": "integer",
                "description": "Maximum revise candidates to revise per round (default: 3)",
                "minimum": 1,
                "default": 3,
            },
            "stop_if_no_change": {
                "type": "boolean",
                "description": "Stop if no new proceed candidates in a round (default: true)",
                "default": True,
            },
            "enable_landscape": {
                "type": "boolean",
                "description": "Enable landscape survey stage before candidate generation: local scan + multi-source search + literature map (default: true, auto-enabled when no local papers)",
                "default": True,
            },
            "landscape_max_online": {
                "type": "integer",
                "description": "Max papers per online source (arXiv/Crossref/OpenAlex) during landscape survey (default: 8)",
                "minimum": 1,
                "maximum": 30,
                "default": 8,
            },
            "reviewer_model": {
                "type": "string",
                "nullable": True,
                "description": "External reviewer model for dual-model collaboration (e.g. 'gpt-4o', 'claude-sonnet-4-7'). When null, runs in single-model mode (default: null)",
            },
            "executor_model": {
                "type": "string",
                "nullable": True,
                "description": "Override the executor model for this workflow run (e.g. 'gpt-4o', 'claude-sonnet-4-7'). Uses config default when null (default: null)",
            },
        },
        "required": ["topic"],
    }

    # Class-level locks per topic slug to prevent concurrent same-topic execution
    _locks: dict[str, asyncio.Lock] = {}

    def __init__(
        self,
        workspace: str | None = None,
        provider: Any = None,
        proxy: str | None = None,
        semantic_config: Any = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._provider = provider
        self._proxy = proxy
        self._semantic_config = semantic_config
        self._search_index: Any = None

    def _resolve_path(self, path: str) -> Path:
        if not self._workspace:
            return Path(path)
        return self._workspace / path

    def _load_local_papers(self) -> list[dict[str, Any]]:
        """Load all papers from local literature storage."""
        if not self._workspace:
            return []
        papers_dir = self._resolve_path("literature/papers")
        if not papers_dir.exists():
            return []
        papers = []
        for fp in papers_dir.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    papers.append(json.load(f))
            except Exception:
                continue
        return papers

    def _score_paper_relevance(
        self,
        paper: dict[str, Any],
        query: str,
    ) -> float:
        """Score a paper's relevance to a query using keyword matching."""
        score = 0.0
        query_lower = query.lower()
        query_terms = query_lower.split()

        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        summary = paper.get("summary", {})
        if isinstance(summary, dict):
            keywords_list = " ".join(summary.get("keywords", [])).lower()
            topic_tags = " ".join(summary.get("topic_tags", [])).lower()
            one_sentence = summary.get("one_sentence", "").lower()
        else:
            keywords_list = ""
            topic_tags = ""
            one_sentence = ""

        for term in query_terms:
            if term in title:
                score += 5.0
            if term in keywords_list:
                score += 3.0
            if term in topic_tags:
                score += 3.0
            if term in one_sentence:
                score += 2.0
            if term in abstract:
                score += 1.0

        return score

    def _find_relevant_papers(
        self,
        query: str,
        max_papers: int = 5,
    ) -> list[dict[str, Any]]:
        """Find most relevant papers for a query using keyword fallback only.

        For async contexts, use _find_relevant_papers_async() instead which
        properly handles the semantic search index.
        """
        all_papers = self._load_local_papers()
        if not all_papers:
            return []

        # Fallback: keyword-based relevance scoring
        scored = []
        for paper in all_papers:
            score = self._score_paper_relevance(paper, query)
            if score > 0:
                scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored[:max_papers] if s > 0]

    async def _find_relevant_papers_async(
        self,
        query: str,
        max_papers: int = 5,
    ) -> list[dict[str, Any]]:
        """Find most relevant papers for a query using semantic index + keyword scoring.

        Uses hybrid search (semantic + keyword) as the primary path and merges
        results to maximize coverage. Falls back to keyword-only scoring if
        the semantic index is unavailable.
        """
        all_papers = self._load_local_papers()
        if not all_papers:
            return []

        paper_by_id: dict[str, dict[str, Any]] = {}
        for p in all_papers:
            pid = p.get("paper_id")
            if pid:
                paper_by_id[pid] = p

        # Semantic search score for each paper_id (0 if not returned)
        semantic_scores: dict[str, float] = {}
        search_index = self._get_search_index()
        if search_index is not None:
            try:
                await search_index.initialize()
                results = await search_index.search(query, top_k=max_papers * 2, rerank=False)
                search_index.close()
                for r in results:
                    pid = r.get("paper_id")
                    if pid:
                        # Use combined rank as score (lower rank = higher score)
                        # RRF score is in r.get("score", 0), fall back to position
                        semantic_scores[pid] = r.get("score", 0) or (1.0 / (r.get("rank", max_papers) + 1))
            except Exception:
                semantic_scores = {}

        # Keyword-based relevance scoring for all papers
        keyword_scores: dict[str, float] = {}
        for pid, paper in paper_by_id.items():
            score = self._score_paper_relevance(paper, query)
            if score > 0:
                keyword_scores[pid] = score

        # Merge: combine semantic and keyword scores, deduplicate
        # Score = normalized semantic score * keyword score (both must agree)
        # Or semantic-only if keyword score is 0 (semantic found something keyword missed)
        combined: list[tuple[float, dict[str, Any]]] = []
        seen_pids: set[str] = set()

        # Papers returned by semantic search (sorted by semantic score)
        for pid, sem_score in sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True):
            kw_score = keyword_scores.get(pid, 0.0)
            paper = paper_by_id.get(pid)
            if not paper or pid in seen_pids:
                continue
            seen_pids.add(pid)

            if kw_score > 0:
                # Both signals agree → boost
                combined_score = sem_score * kw_score
            else:
                # Semantic only → use semantic score alone
                combined_score = sem_score * 0.5  # dampen to not over-rank semantic-only
            combined.append((combined_score, paper))

        # Papers found only by keyword search (not in semantic results)
        for pid, kw_score in keyword_scores.items():
            if pid in seen_pids:
                continue
            paper = paper_by_id.get(pid)
            if paper:
                seen_pids.add(pid)
                combined.append((kw_score * 0.5, paper))  # keyword-only gets lower priority

        # Sort by combined score descending
        combined.sort(key=lambda x: x[0], reverse=True)
        return [p for _score, p in combined[:max_papers]]

    def _get_search_index(self):
        """Get or create the search index (lazy initialization)."""
        if self._workspace is None or self._semantic_config is None:
            return None
        if self._search_index is None:
            from researchbot.search_index import SearchIndex
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._search_index = SearchIndex(db_path, self._semantic_config)
        return self._search_index

    async def _search_arxiv(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            from researchbot.agent.tools.arxiv_client import search_arxiv, DEFAULT_TIMEOUT
            entries = await search_arxiv(
                query=query,
                start=0,
                max_results=max_results,
                sort_by="relevance",
                sort_order="descending",
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
            papers = []
            for entry in entries:
                papers.append({
                    "paper_id": entry.paper_id,
                    "title": entry.title,
                    "authors": entry.authors,
                    "abstract": entry.summary,
                    "published": entry.published,
                    "categories": entry.categories,
                })
            return papers
        except Exception:
            return []

    async def _search_crossref(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        """Search Crossref for papers."""
        try:
            from researchbot.agent.tools.crossref_client import search_crossref, DEFAULT_TIMEOUT
            works = await search_crossref(
                query=query,
                max_results=max_results,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
            papers = []
            for w in works:
                papers.append({
                    "paper_id": w.doi or "",
                    "title": w.title,
                    "authors": w.authors,
                    "abstract": w.abstract,
                    "year": w.year,
                    "published": w.year,
                    "source": "crossref",
                    "external_ids": {"doi": w.doi} if w.doi else {},
                    "journal": w.journal,
                    "cited_by_count": w.cited_by_count,
                    "url": w.url,
                })
            return papers
        except Exception:
            return []

    async def _search_openalex(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        """Search OpenAlex for papers."""
        try:
            from researchbot.agent.tools.openalex_client import search_openalex, DEFAULT_TIMEOUT
            works = await search_openalex(
                query=query,
                max_results=max_results,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
            papers = []
            for w in works:
                papers.append({
                    "paper_id": w.id or w.doi or "",
                    "title": w.title,
                    "authors": w.authors,
                    "abstract": w.abstract,
                    "year": w.year,
                    "published": w.year,
                    "source": "openalex",
                    "external_ids": {"doi": w.doi, "openalex": w.id} if w.doi else {"openalex": w.id},
                    "journal": w.journal,
                    "cited_by_count": w.cited_by_count,
                    "url": w.url,
                })
            return papers
        except Exception:
            return []

    async def _search_multi_source(
        self,
        topic: str,
        max_per_source: int = 8,
    ) -> list[dict[str, Any]]:
        """Search multiple academic sources and merge results, deduplicating by title similarity."""
        all_papers: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        def _add_papers(papers: list[dict[str, Any]]):
            for p in papers:
                key = _TITLE_KEY_RE.sub("", p.get("title", "").lower())[:80]
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append(p)

        # Run all three searches concurrently
        results = await asyncio.gather(
            self._search_arxiv(topic, max_per_source),
            self._search_crossref(topic, max_per_source),
            self._search_openalex(topic, max_per_source),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, list):
                _add_papers(r)

        return all_papers

    async def _build_literature_map(
        self,
        topic: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Categorize papers into a literature map using LLM."""
        if not self._provider or not papers:
            return None

        # Format papers for the prompt
        paper_lines = []
        for i, p in enumerate(papers[:40], 1):
            title = p.get("title", "?")
            year = p.get("year", "") or p.get("published", "")[:4]
            source = p.get("source", "")
            authors = p.get("authors", [])
            authors_str = ", ".join(authors[:2]) if authors else "Unknown"
            if len(authors) > 2:
                authors_str += " et al."
            abstract = p.get("abstract", "")
            if isinstance(abstract, dict):
                abstract = str(abstract)[:200]
            else:
                abstract = str(abstract)[:200]
            paper_lines.append(f"[{i}] ({source}, {year}) {title} - {authors_str}\n    Abstract: {abstract}")

        papers_text = "\n".join(paper_lines)

        prompt = LANDSCAPE_MAP_PROMPT.format(
            topic=topic,
            num_papers=len(papers),
            papers_text=papers_text,
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                response = await self._provider.chat_with_retry(messages)
                content = response.content or ""
                result = _parse_json_robust(content, expect_array=False)
                if result is not None and isinstance(result, dict):
                    if "categories" in result and "summary" in result:
                        return result

                if attempt == 0 and content:
                    clarification = (
                        "\n\nNote: The above output was not valid JSON. "
                        "Please provide ONLY the JSON object with no other text."
                    )
                    messages = [{"role": "user", "content": prompt + clarification}]
                    continue
            except Exception:
                if attempt == 0:
                    continue
                pass

        return None

    def _save_landscape_report(
        self,
        topic: str,
        local_papers: list[dict[str, Any]],
        online_papers: list[dict[str, Any]],
        all_papers: list[dict[str, Any]],
        literature_map: dict[str, Any] | None,
        search_local: bool = True,
        max_online: int = 8,
    ) -> tuple[str, str]:
        """Save landscape report to JSON and markdown files."""
        slug = _slugify(topic)
        innovation_dir = self._resolve_path(f"innovation/{slug}")
        innovation_dir.mkdir(parents=True, exist_ok=True)

        now = utc_now()

        # Build JSON report
        report_data: dict[str, Any] = {
            "topic": topic,
            "topic_slug": slug,
            "generated_at": now,
            "params": {
                "search_local": search_local,
                "max_online": max_online,
            },
            "local_papers_count": len(local_papers),
            "online_papers_count": len(online_papers),
            "total_papers_count": len(all_papers),
            "local_papers": [
                {
                    "paper_id": p.get("paper_id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "source": p.get("source", "local"),
                }
                for p in local_papers
            ],
            "online_papers": [
                {
                    "paper_id": p.get("paper_id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "source": p.get("source", ""),
                    "cited_by_count": p.get("cited_by_count", 0),
                }
                for p in online_papers
            ],
        }

        if literature_map:
            report_data["literature_map"] = literature_map

        json_path = innovation_dir / "landscape_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # Build markdown report
        md_lines = [
            f"# Landscape Survey: {topic}\n",
            f"\n**Generated**: {now}\n",
            f"\n**Papers Found**: {len(all_papers)} total "
            f"({len(local_papers)} local, {len(online_papers)} online)\n",
        ]

        if local_papers:
            md_lines.append("\n## Local Papers\n")
            for p in local_papers:
                md_lines.append(
                    f"- {p.get('title', '?')} ({p.get('year', '?')})\n"
                )

        if online_papers:
            md_lines.append("\n## Online Papers\n")
            by_source: dict[str, list[dict]] = {}
            for p in online_papers:
                src = p.get("source", "unknown")
                by_source.setdefault(src, []).append(p)
            for src, papers in by_source.items():
                md_lines.append(f"\n### {src.title()}\n")
                for p in papers:
                    title = p.get("title", "?")
                    year = p.get("year", "?")
                    cited = p.get("cited_by_count", 0)
                    cite_str = f" (cited: {cited})" if cited else ""
                    md_lines.append(f"- {title} ({year}){cite_str}\n")

        if literature_map:
            categories = literature_map.get("categories", [])
            summary = literature_map.get("summary", {})

            md_lines.append(f"\n## Literature Map ({len(categories)} categories)\n")

            if summary.get("overall_trends"):
                md_lines.append(f"\n**Overall Trends**: {summary['overall_trends']}\n")

            for i, cat in enumerate(categories, 1):
                md_lines.append(f"\n### {i}. {cat.get('name', 'Unnamed')}\n")
                md_lines.append(f"{cat.get('description', '')}\n")

                reps = cat.get("representative_papers", [])
                if reps:
                    md_lines.append("\n**Representative Papers**:\n")
                    for r in reps:
                        md_lines.append(f"- {r}\n")

                problems = cat.get("main_problems", [])
                if problems:
                    md_lines.append(f"\n**Main Problems**: {'; '.join(problems)}\n")

                methods = cat.get("typical_methods", [])
                if methods:
                    md_lines.append(f"\n**Typical Methods**: {'; '.join(methods)}\n")

                gaps = cat.get("covered_gaps", [])
                if gaps:
                    md_lines.append(f"\n**Covered Gaps**: {'; '.join(gaps)}\n")

                limits = cat.get("limitations", [])
                if limits:
                    md_lines.append(f"\n**Limitations**: {'; '.join(limits)}\n")

                opps = cat.get("exploration_opportunities", [])
                if opps:
                    md_lines.append(f"\n**Exploration Opportunities**: {'; '.join(opps)}\n")

            if summary.get("key_gaps"):
                md_lines.append("\n## Key Gaps Across Categories\n")
                for g in summary["key_gaps"]:
                    md_lines.append(f"- {g}\n")

            if summary.get("recommended_exploration"):
                md_lines.append("\n## Recommended Exploration\n")
                for r in summary["recommended_exploration"]:
                    md_lines.append(f"- {r}\n")

        md_path = innovation_dir / "landscape_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)

        return str(json_path), str(md_path)

    def _format_landscape_context(
        self,
        literature_map: dict[str, Any] | None,
        all_papers: list[dict[str, Any]],
    ) -> str:
        """Format landscape survey results as context for candidate generation."""
        if not literature_map:
            # Fallback: just list paper titles and abstracts
            parts = []
            for p in all_papers[:10]:
                title = p.get("title", "?")
                abstract = str(p.get("abstract", ""))[:150]
                parts.append(f"- {title}: {abstract}")
            return "\n".join(parts) if parts else ""

        parts = []
        categories = literature_map.get("categories", [])
        summary = literature_map.get("summary", {})

        if summary.get("overall_trends"):
            parts.append(f"Overall trends: {summary['overall_trends']}")

        for cat in categories:
            name = cat.get("name", "Unnamed")
            desc = cat.get("description", "")
            problems = "; ".join(cat.get("main_problems", []))
            limits = "; ".join(cat.get("limitations", []))
            opps = "; ".join(cat.get("exploration_opportunities", []))
            line = f"- [{name}] {desc}"
            if problems:
                line += f" Problems: {problems}"
            if limits:
                line += f" Limitations: {limits}"
            if opps:
                line += f" Opportunities: {opps}"
            parts.append(line)

        if summary.get("key_gaps"):
            parts.append(f"Key gaps: {'; '.join(summary['key_gaps'])}")

        return "\n".join(parts)

    async def _run_landscape_survey(
        self,
        topic: str,
        output_dir: Path,
        metadata: dict[str, Any],
        search_local: bool = True,
        max_online: int = 8,
        overwrite: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run landscape survey: local scan + multi-source search + literature map.

        Returns (all_papers, context_string) for use in candidate generation.
        """
        landscape_json_path = str(output_dir / "landscape_report.json")
        landscape_md_path = str(output_dir / "landscape_report.md")
        landscape_exists = (output_dir / "landscape_report.json").exists()

        if landscape_exists and not overwrite:
            with open(output_dir / "landscape_report.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate that cached params match current params
            cached_params = data.get("params", {})
            current_params = {"search_local": search_local, "max_online": max_online}
            if cached_params != current_params:
                # Params changed, discard stale cache
                landscape_exists = False
                landscape_json_path = None
            else:
                local_papers = data.get("local_papers", [])
                online_papers = data.get("online_papers", [])
                all_papers = local_papers + online_papers
                literature_map = data.get("literature_map")

                now = utc_now()
                metadata["stages"]["stage0_landscape"].update({
                    "status": "skipped",
                    "started_at": now,
                    "completed_at": now,
                    "output_files": {
                        "landscape_json": landscape_json_path,
                        "landscape_md": landscape_md_path,
                    },
                    "num_local_papers": len(local_papers),
                    "num_online_papers": len(online_papers),
                    "num_categories": len(literature_map.get("categories", [])) if literature_map else 0,
                    "note": "Reused existing file",
                })

                context = self._format_landscape_context(literature_map, all_papers)
                return all_papers, context

        # Run the landscape survey
        metadata["stages"]["stage0_landscape"]["status"] = "running"
        metadata["stages"]["stage0_landscape"]["started_at"] = utc_now()
        metadata["stages"]["stage0_landscape"]["input_params"] = {
            "search_local": search_local,
            "max_online": max_online,
        }
        _save_workflow_metadata(metadata, self._workspace, topic)

        # Step 1: Local scan
        local_papers: list[dict[str, Any]] = []
        if search_local:
            local_papers = await self._find_relevant_papers_async(topic, max_papers=15)

        # Step 2: Multi-source online search
        online_papers = await self._search_multi_source(topic, max_per_source=max_online)

        # Deduplicate: combine and remove duplicates by title
        seen: set[str] = set()
        all_papers: list[dict[str, Any]] = []

        for p in local_papers:
            key = _TITLE_KEY_RE.sub("", p.get("title", "").lower())[:80]
            if key and key not in seen:
                seen.add(key)
                all_papers.append(p)

        for p in online_papers:
            key = _TITLE_KEY_RE.sub("", p.get("title", "").lower())[:80]
            if key and key not in seen:
                seen.add(key)
                all_papers.append(p)

        # Step 3: Build literature map
        literature_map = await self._build_literature_map(topic, all_papers)

        # Save report
        json_path, md_path = self._save_landscape_report(
            topic, local_papers, online_papers, all_papers, literature_map,
            search_local=search_local, max_online=max_online,
        )

        metadata["stages"]["stage0_landscape"]["status"] = "completed"
        metadata["stages"]["stage0_landscape"]["completed_at"] = utc_now()
        metadata["stages"]["stage0_landscape"]["output_files"] = {
            "landscape_json": json_path,
            "landscape_md": md_path,
        }
        metadata["stages"]["stage0_landscape"]["num_local_papers"] = len(local_papers)
        metadata["stages"]["stage0_landscape"]["num_online_papers"] = len(online_papers)
        metadata["stages"]["stage0_landscape"]["num_categories"] = (
            len(literature_map.get("categories", [])) if literature_map else 0
        )
        _save_workflow_metadata(metadata, self._workspace, topic)

        context = self._format_landscape_context(literature_map, all_papers)
        return all_papers, context

    async def _generate_candidates(
        self,
        topic: str,
        context: str,
        num_candidates: int,
    ) -> list[dict[str, Any]]:
        """Generate innovation point candidates using LLM with robust JSON parsing."""
        if not self._provider:
            return []

        prompt = INNOVATION_POINT_PROMPT.format(
            topic=topic,
            context=context or "No existing literature context available.",
        )
        prompt = prompt.replace(
            "Generate 3-8 candidate innovation points.",
            f"Generate exactly {num_candidates} candidate innovation points.",
        )

        # Random angle emphasis to ensure different candidates on each run
        angle_prompts = [
            "Focus on problem definition innovations: new task framings, sub-problem identifications, or evaluation metric designs.",
            "Focus on method innovations: new algorithms, architecture changes, or optimization techniques.",
            "Focus on experimental innovations: new analysis frameworks, tooling, or infrastructure approaches.",
            "Focus on setting innovations: new datasets, domain adaptations, or evaluation protocols.",
            "Prioritize highly specific and actionable ideas with concrete mechanisms rather than broad conceptual directions.",
            "Prioritize novel combinations of existing techniques that haven't been explored in this research area.",
            "Explore underexplored angles: consider practical deployment challenges, domain-specific constraints, or cross-domain applications.",
        ]
        random_angle = random.choice(angle_prompts)

        run_nonce = uuid.uuid4().hex[:8]

        diversity_plan = (
            f"Diversity plan: {random_angle} "
            "Make each candidate clearly different in angle, not just wording. "
            "Use concrete mechanisms and specific sub-problems instead of broad umbrella statements."
        )
        if num_candidates >= 4:
            diversity_plan += (
                " Try to cover problem definition, method improvement, system/setting, and "
                "experimental/tooling angles at least once."
            )
        elif num_candidates == 3:
            diversity_plan += " Prefer three different innovation levels."
        prompt = (
            f"{prompt}\n\n{diversity_plan}\n"
            f"Variation token: {run_nonce}. Treat this run as an independent ideation batch and avoid reusing prior phrasings."
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                response = await self._provider.chat_with_retry(messages, temperature=0.85)
                content = response.content or ""

                result = _parse_json_robust(content, expect_array=True)
                if result is not None and isinstance(result, list) and len(result) > 0:
                    normalized = [_normalize_candidate(c) for c in result]
                    diverse, rejected = _filter_diverse_candidates(normalized)
                    if len(diverse) >= num_candidates:
                        return diverse[:num_candidates]

                    if attempt == 0 and rejected:
                        rejected_json = json.dumps(rejected[:num_candidates], ensure_ascii=False, indent=2)
                        kept_json = json.dumps(diverse[:num_candidates], ensure_ascii=False, indent=2)
                        regeneration_hint = (
                            "\n\nImportant: Several candidates above were too similar. "
                            f"Regenerate the full JSON array with exactly {num_candidates} candidates. "
                            "Keep the good diversity of the remaining ideas, but replace the rejected ones with "
                            "clearly different angles and different innovation levels when possible. "
                            "Do not reuse, paraphrase, or slightly rename the rejected candidates.\n\n"
                            f"Rejected candidates:\n{rejected_json}\n\n"
                            f"Acceptable candidates to stay different from:\n{kept_json}\n"
                        )
                        messages = [{"role": "user", "content": prompt + regeneration_hint}]
                        continue

                    return diverse if diverse else normalized[:num_candidates]

                if attempt == 0 and content:
                    clarification = (
                        "\n\nNote: The above output was not valid JSON. "
                        "Please regenerate with ONLY a JSON array of innovation points, "
                        f"exactly {num_candidates} objects, with no other text. "
                        "Make each candidate clearly different in innovation angle."
                    )
                    messages = [{"role": "user", "content": prompt + clarification}]
                    continue

            except Exception:
                if attempt == 0:
                    continue
                pass

        return []

    async def _analyze_novelty(
        self,
        candidate: dict[str, Any],
        related_papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze novelty of a candidate against related papers using LLM."""
        if not self._provider:
            return _normalize_analysis({
                "novelty_level": "unknown",
                "novelty_conclusion": "LLM provider not configured",
            })

        papers_json = json.dumps(related_papers[:5], ensure_ascii=False, indent=2)

        prompt = NOVELTY_ANALYSIS_PROMPT.format(
            title=candidate.get("title", ""),
            problem=candidate.get("problem", ""),
            idea=candidate.get("idea", ""),
            novelty_claim=candidate.get("novelty_claim", ""),
            keywords=", ".join(candidate.get("keywords", [])),
            papers_json=papers_json,
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                response = await self._provider.chat_with_retry(messages)
                content = response.content or ""

                result = _parse_json_robust(content, expect_array=False)
                if result is not None and isinstance(result, dict):
                    normalized = _normalize_analysis(result)
                    return normalized

                if attempt == 0 and content:
                    clarification = (
                        "\n\nNote: The above output was not valid JSON. "
                        "Please provide ONLY the JSON object with no other text."
                    )
                    messages = [{"role": "user", "content": prompt + clarification}]
                    continue

            except Exception:
                if attempt == 0:
                    continue
                pass

        return _normalize_analysis({
            "novelty_level": "unknown",
            "novelty_conclusion": "Failed to parse novelty analysis from LLM",
        })

    def _format_related_papers_for_review(
        self,
        related_papers: list[dict[str, Any]],
        max_papers: int = 5,
        abstract_chars: int = 300,
    ) -> str:
        """Format related papers for the review prompt with full evidence.

        Includes paper_id, title, year, and abstract (truncated).
        Total prompt budget per paper is kept reasonable.
        """
        if not related_papers:
            return "No related papers found."

        lines = []
        for p in related_papers[:max_papers]:
            pid = p.get("paper_id", "?")
            title = p.get("title", "?")
            year = p.get("year", "") or (p.get("published", "")[:4] if p.get("published") else "")
            authors = p.get("authors", [])
            if isinstance(authors, list) and authors:
                authors_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    authors_str += " et al."
            else:
                authors_str = "Unknown"

            abstract = p.get("abstract", "")
            if isinstance(abstract, dict):
                abstract = abstract.get("problem", "") or abstract.get("method", "")
            if abstract:
                abstract = str(abstract)[:abstract_chars]
            else:
                abstract = "No abstract available."

            lines.append(f"[{pid}] {title} ({year}) - {authors_str}")
            lines.append(f"    Abstract: {abstract}")
            lines.append("")

        return "\n".join(lines)

    async def _review_candidate(
        self,
        candidate: dict[str, Any],
        analysis: dict[str, Any],
        related_papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Review and score a candidate using LLM with full evidence."""
        if not self._provider:
            return _normalize_review({
                "reasoning": "LLM provider not configured",
                "decision": "drop",
            })

        # Format related papers with full evidence (abstract, year, authors)
        related_papers_str = self._format_related_papers_for_review(related_papers, max_papers=5, abstract_chars=300)

        prompt = REVIEW_PROMPT.format(
            title=candidate.get("title", ""),
            problem=candidate.get("problem", ""),
            idea=candidate.get("idea", ""),
            key_difference=candidate.get("key_difference", candidate.get("novelty_claim", "")),
            expected_value=candidate.get("expected_value", ""),
            keywords=", ".join(candidate.get("keywords", [])),
            innovation_level=candidate.get("innovation_level", "unknown"),
            novelty_level=analysis.get("novelty_level", "unknown"),
            overlap_dimensions=", ".join(analysis.get("overlap_dimensions", [])) or "Not specified",
            is_duplicate=str(analysis.get("is_duplicate", False)),
            is_highly_similar=str(analysis.get("is_highly_similar", False)),
            similarities=", ".join(analysis.get("similarities", [])) or "None identified",
            differences=", ".join(analysis.get("differences", [])) or "None identified",
            novelty_conclusion=analysis.get("novelty_conclusion", ""),
            gaps_addressed=", ".join(analysis.get("gaps_addressed", [])) or "None identified",
            related_papers=related_papers_str,
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                response = await self._provider.chat_with_retry(messages)
                content = response.content or ""

                result = _parse_json_robust(content, expect_array=False)
                if result is not None and isinstance(result, dict):
                    normalized = _normalize_review(result)
                    return normalized

                if attempt == 0 and content:
                    clarification = (
                        "\n\nNote: The above output was not valid JSON. "
                        "Please provide ONLY the JSON object with no other text."
                    )
                    messages = [{"role": "user", "content": prompt + clarification}]
                    continue

            except Exception:
                if attempt == 0:
                    continue
                pass

        return _normalize_review({
            "reasoning": "Failed to parse review from LLM",
            "decision": "drop",
        })

    def _save_candidates(
        self,
        topic: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Save candidates to JSON and markdown files."""
        slug = _slugify(topic)
        innovation_dir = self._resolve_path(f"innovation/{slug}")
        innovation_dir.mkdir(parents=True, exist_ok=True)

        candidates_data = {
            "topic": topic,
            "topic_slug": slug,
            "generated_at": utc_now(),
            "num_candidates": len(candidates),
            "candidates": candidates,
        }
        json_path = innovation_dir / "candidates.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, ensure_ascii=False, indent=2)

        md_path = innovation_dir / "candidates.md"
        lines = [
            f"# Innovation Point Candidates: {topic}\n",
            f"\n**Generated**: {candidates_data['generated_at']}\n",
            f"**Number of Candidates**: {len(candidates)}\n",
            "\n## Candidates\n",
        ]
        for i, c in enumerate(candidates, 1):
            lines.append(f"\n### {i}. {c.get('title', 'Untitled')}\n")
            lines.append(f"**Problem**: {c.get('problem', 'N/A')}\n")
            lines.append(f"**Idea**: {c.get('idea', 'N/A')}\n")
            lines.append(f"**Novelty Claim**: {c.get('novelty_claim', 'N/A')}\n")
            lines.append(f"**Expected Value**: {c.get('expected_value', 'N/A')}\n")
            lines.append(f"**Keywords**: {', '.join(c.get('keywords', []))}\n")
            if c.get("related_work_hypothesis"):
                lines.append(f"**Related Work Hypothesis**: {c['related_work_hypothesis']}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return str(json_path), str(md_path)

    def _save_novelty_report(
        self,
        topic: str,
        novelty_results: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Save novelty report to JSON and markdown files."""
        slug = _slugify(topic)
        innovation_dir = self._resolve_path(f"innovation/{slug}")
        innovation_dir.mkdir(parents=True, exist_ok=True)

        report_data = {
            "topic": topic,
            "topic_slug": slug,
            "generated_at": utc_now(),
            "results": novelty_results,
        }
        json_path = innovation_dir / "novelty_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        md_path = innovation_dir / "novelty_report.md"
        lines = [
            f"# Novelty Report: {topic}\n",
            f"\n**Generated**: {report_data['generated_at']}\n",
            "\n## Novelty Analysis Results\n",
        ]

        for i, result in enumerate(novelty_results, 1):
            candidate = result.get("candidate", {})
            analysis = result.get("analysis", {})
            related = result.get("related_papers", [])

            novelty_level = analysis.get("novelty_level", "unknown")
            level_indicator = {
                "high": "[HIGH]",
                "medium": "[MEDIUM]",
                "low": "[LOW]",
                "unknown": "[UNKNOWN]",
            }.get(novelty_level, "[UNKNOWN]")

            lines.append(f"\n### {i}. {candidate.get('title', 'Untitled')} {level_indicator}\n")
            lines.append(f"**Novelty Level**: {novelty_level.upper()}\n")
            lines.append(f"**Is Duplicate**: {analysis.get('is_duplicate', False)}\n")
            lines.append(f"**Is Highly Similar**: {analysis.get('is_highly_similar', False)}\n")
            lines.append(f"**Conclusion**: {analysis.get('novelty_conclusion', 'N/A')}\n")

            if related:
                lines.append("\n**Related Papers Found**:\n")
                for p in related[:5]:
                    pid = p.get("paper_id", "?")
                    title = p.get("title", "?")
                    year = p.get("year", "")
                    source = p.get("source", "arxiv")
                    lines.append(f"- [{pid}] {title} ({year}) [{source}]\n")

            if analysis.get("similarities"):
                lines.append("\n**Similarities**:\n")
                for s in analysis["similarities"]:
                    lines.append(f"- {s}\n")

            if analysis.get("differences"):
                lines.append("\n**Differences**:\n")
                for d in analysis["differences"]:
                    lines.append(f"- {d}\n")

            if analysis.get("gaps_addressed"):
                lines.append("\n**Gaps Addressed**:\n")
                for g in analysis["gaps_addressed"]:
                    lines.append(f"- {g}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return str(json_path), str(md_path)

    def _save_review_report(
        self,
        topic: str,
        review_results: list[dict[str, Any]],
        top_candidates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Save review report to JSON and markdown files."""
        slug = _slugify(topic)
        innovation_dir = self._resolve_path(f"innovation/{slug}")
        innovation_dir.mkdir(parents=True, exist_ok=True)

        # Build summary
        proceed = [r for r in review_results if r.get("review", {}).get("decision") == "proceed"]
        revise = [r for r in review_results if r.get("review", {}).get("decision") == "revise"]
        drop = [r for r in review_results if r.get("review", {}).get("decision") == "drop"]

        report_data = {
            "topic": topic,
            "topic_slug": slug,
            "generated_at": utc_now(),
            "review_summary": {
                "total": len(review_results),
                "proceed": len(proceed),
                "revise": len(revise),
                "drop": len(drop),
            },
            "results": review_results,
            "top_candidates": top_candidates,
            "recommendation": self._build_recommendation_text(top_candidates, review_results),
        }

        json_path = innovation_dir / "review_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        md_path = innovation_dir / "review_report.md"
        lines = [
            f"# Review Report: {topic}\n",
            f"\n**Generated**: {report_data['generated_at']}\n",
            f"\n**Total Candidates Reviewed**: {len(review_results)}\n",
            f"- Proceed: {len(proceed)}\n",
            f"- Revise: {len(revise)}\n",
            f"- Drop: {len(drop)}\n",
            "\n## Top Recommended Candidates\n",
        ]

        for i, tc in enumerate(top_candidates, 1):
            candidate = tc.get("candidate", {})
            review = tc.get("review", {})
            lines.append(
                f"\n### {i}. {candidate.get('title', 'Untitled')} "
                f"(Overall: {review.get('overall_score', '?')}/10)\n"
            )
            lines.append(f"**Decision**: {review.get('decision', '?').upper()}\n")
            lines.append(f"**Reasoning**: {review.get('reasoning', 'N/A')}\n")
            lines.append(
                f"**Scores**: Novelty={review.get('novelty_score', '?')}, "
                f"Feasibility={review.get('feasibility_score', '?')}, "
                f"Evidence={review.get('evidence_score', '?')}, "
                f"Impact={review.get('impact_score', '?')}, "
                f"Risk={review.get('risk_score', '?')}\n"
            )
            if review.get("next_step"):
                lines.append(f"**Next Step**: {review['next_step']}\n")

        lines.append("\n## All Candidates Summary\n")
        lines.append("| # | Title | Decision | Overall | Novelty | Feasibility | Evidence | Impact | Risk |\n")
        lines.append("|---|-------|----------|---------|---------|------------|----------|--------|------|\n")
        for i, rr in enumerate(review_results, 1):
            c = rr.get("candidate", {})
            r = rr.get("review", {})
            lines.append(
                f"| {i} | {c.get('title', 'Untitled')[:40]} | "
                f"{r.get('decision', '?')} | {r.get('overall_score', '?')} | "
                f"{r.get('novelty_score', '?')} | {r.get('feasibility_score', '?')} | "
                f"{r.get('evidence_score', '?')} | {r.get('impact_score', '?')} | "
                f"{r.get('risk_score', '?')} |\n"
            )

        lines.append("\n## Detailed Reviews\n")
        for i, rr in enumerate(review_results, 1):
            c = rr.get("candidate", {})
            a = rr.get("analysis", {})
            r = rr.get("review", {})

            lines.append(f"\n### {i}. {c.get('title', 'Untitled')}\n")
            lines.append(f"**Problem**: {c.get('problem', 'N/A')[:200]}\n")
            lines.append(f"**Idea**: {c.get('idea', 'N/A')[:200]}\n")

            novelty_level = a.get("novelty_level", "unknown")
            lines.append(
                f"\n**Novelty Assessment**: {novelty_level.upper()} "
                f"(from novelty search)\n"
            )
            if a.get("novelty_conclusion"):
                lines.append(f"**Novelty Conclusion**: {a['novelty_conclusion']}\n")

            lines.append(f"\n**Review Decision**: {r.get('decision', '?').upper()}\n")
            lines.append(f"**Overall Score**: {r.get('overall_score', '?')}/10\n")
            lines.append(
                f"**Scores**: Novelty={r.get('novelty_score', '?')}, "
                f"Feasibility={r.get('feasibility_score', '?')}, "
                f"Evidence={r.get('evidence_score', '?')}, "
                f"Impact={r.get('impact_score', '?')}, "
                f"Risk={r.get('risk_score', '?')}\n"
            )
            if r.get("reasoning"):
                lines.append(f"**Reasoning**: {r['reasoning']}\n")
            if r.get("main_risks"):
                lines.append("**Main Risks**:\n")
                for risk in r["main_risks"]:
                    lines.append(f"- {risk}\n")
            if r.get("recommended_revision"):
                lines.append(f"**Recommended Revision**: {r['recommended_revision']}\n")
            if r.get("next_step"):
                lines.append(f"**Next Step**: {r['next_step']}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return str(json_path), str(md_path)

    async def _call_reviewer(
        self,
        prompt_template: str,
        format_kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call the external reviewer model via the existing provider.

        Returns parsed JSON dict on success, None on failure.
        Silently returns None if reviewer_model is not configured or provider is missing.
        """
        if not self._provider:
            return None

        model = getattr(self, "_reviewer_model", None)
        if not model:
            return None

        prompt = prompt_template.format(**format_kwargs)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._provider.chat_with_retry(messages, model=model)
            content = response.content or ""
            result = _parse_json_robust(content, expect_array=False)
            return result
        except Exception:
            return None

    def _build_recommendation_text(
        self,
        top_candidates: list[dict[str, Any]],
        all_results: list[dict[str, Any]],
    ) -> str:
        """Build human-readable recommendation text."""
        if not top_candidates:
            return "No candidates meet the threshold for recommendation."

        lines = [
            f"Based on the review of {len(all_results)} candidates, "
            f"{len(top_candidates)} candidate(s) are recommended for proceeding:",
            "",
        ]
        for i, tc in enumerate(top_candidates, 1):
            c = tc.get("candidate", {})
            r = tc.get("review", {})
            lines.append(
                f"{i}. **{c.get('title', 'Untitled')}** "
                f"(overall score: {r.get('overall_score', '?')}/10, "
                f"decision: {r.get('decision', '?')})"
            )
            if r.get("reasoning"):
                lines.append(f"   Reason: {r['reasoning'][:150]}")
            if r.get("next_step"):
                lines.append(f"   Next step: {r['next_step']}")
            lines.append("")

        # Add why others were not chosen
        drop_count = len([r for r in all_results if r.get("review", {}).get("decision") == "drop"])
        revise_count = len([r for r in all_results if r.get("review", {}).get("decision") == "revise"])
        if drop_count or revise_count:
            lines.append(
                f"Others were not selected: {drop_count} dropped, "
                f"{revise_count} need revision before proceeding."
            )

        return "\n".join(lines)

    # =============================================================================
    # Iteration Methods
    # =============================================================================

    async def _generate_revised_candidate(
        self,
        candidate: dict[str, Any],
        analysis: dict[str, Any],
        review: dict[str, Any],
        related_papers: list[dict[str, Any]],
        round_num: int,
    ) -> dict[str, Any] | None:
        """Generate a revised candidate from a 'revise' result."""
        if not self._provider:
            return None

        related_papers_str = _format_related_papers_for_revision(related_papers, max_papers=5)
        parent_title = candidate.get("title", "Unknown")

        prompt = REVISION_PROMPT.format(
            title=candidate.get("title", ""),
            problem=candidate.get("problem", ""),
            idea=candidate.get("idea", ""),
            key_difference=candidate.get("key_difference", ""),
            expected_value=candidate.get("expected_value", ""),
            keywords=", ".join(candidate.get("keywords", [])),
            innovation_level=candidate.get("innovation_level", "unknown"),
            decision=review.get("decision", "revise"),
            overall_score=review.get("overall_score", 5),
            novelty_score=review.get("novelty_score", 5),
            feasibility_score=review.get("feasibility_score", 5),
            evidence_score=review.get("evidence_score", 5),
            impact_score=review.get("impact_score", 5),
            risk_score=review.get("risk_score", 5),
            reasoning=review.get("reasoning", ""),
            main_risks=", ".join(review.get("main_risks", [])) or "None specified",
            recommended_revision=review.get("recommended_revision", ""),
            next_step=review.get("next_step", ""),
            novelty_level=analysis.get("novelty_level", "unknown"),
            overlap_dimensions=", ".join(analysis.get("overlap_dimensions", [])) or "Not specified",
            similarities=", ".join(analysis.get("similarities", [])) or "None identified",
            differences=", ".join(analysis.get("differences", [])) or "None identified",
            novelty_conclusion=analysis.get("novelty_conclusion", ""),
            related_papers=related_papers_str,
            round_number=round_num,
        )

        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            try:
                response = await self._provider.chat_with_retry(messages)
                content = response.content or ""
                result = _parse_json_robust(content, expect_array=False)
                if result is not None and isinstance(result, dict):
                    revision_reason = review.get("recommended_revision", "") or review.get("next_step", "addressed review feedback")
                    return _normalize_revised_candidate(result, parent_title, round_num, revision_reason)
                if attempt == 0 and content:
                    clarification = (
                        "\n\nNote: The above output was not valid JSON. "
                        "Please provide ONLY the JSON object for the revised candidate, no other text."
                    )
                    messages = [{"role": "user", "content": prompt + clarification}]
            except Exception:
                if attempt == 0:
                    continue
        return None

    async def _run_iteration_round(
        self,
        topic: str,
        candidates: list[dict[str, Any]],
        search_local: bool,
        search_online: bool,
        max_related: int,
        round_num: int,
        enable_review: bool = True,
    ) -> list[dict[str, Any]]:
        """Run novelty search and review for a batch of candidates in one iteration round."""

        async def process_single_candidate(
            candidate: dict[str, Any],
        ) -> dict[str, Any]:
            """Process novelty search and review for a single candidate."""
            keywords = candidate.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]

            related_papers: list[dict[str, Any]] = []

            if search_local and keywords:
                query = " ".join(keywords)
                local_results = await self._find_relevant_papers_async(query, max_papers=max_related)
                related_papers.extend(local_results)

            if search_online and keywords:
                query = " OR ".join(keywords[:3])
                try:
                    arxiv_results = await self._search_arxiv(query, max_related)
                    existing_ids = {p.get("paper_id") for p in related_papers}
                    for p in arxiv_results:
                        if p.get("paper_id") not in existing_ids:
                            related_papers.append(p)
                            existing_ids.add(p.get("paper_id"))
                except Exception:
                    pass

            # Store full paper info
            full_related_papers: list[dict[str, Any]] = []
            for p in related_papers[:max_related]:
                paper_record: dict[str, Any] = {
                    "paper_id": p.get("paper_id", ""),
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "published": p.get("published", ""),
                    "year": p.get("year", "") or (p.get("published", "")[:4] if p.get("published") else ""),
                    "categories": p.get("categories", []) if isinstance(p.get("categories"), list) else [],
                    "authors": p.get("authors", [])[:3] if isinstance(p.get("authors"), list) else [],
                    "source": p.get("source", "arxiv"),
                }
                full_related_papers.append(paper_record)

            # Novelty analysis
            analysis = await self._analyze_novelty(candidate, full_related_papers)

            # Review (skip LLM call when disabled)
            if enable_review:
                review = await self._review_candidate(candidate, analysis, full_related_papers)
            else:
                review = _normalize_review({
                    "reasoning": "Review skipped (enable_review=False)",
                    "decision": "revise",
                })

            return {
                "candidate": candidate,
                "analysis": analysis,
                "review": review,
                "related_papers": full_related_papers,
            }

        results = await asyncio.gather(
            *[process_single_candidate(c) for c in candidates]
        )
        return list(results)

    def _save_iteration_reports(
        self,
        topic: str,
        rounds_data: list[dict[str, Any]],
        final_candidates: list[dict[str, Any]],
        stopping_reason: str,
        iteration_params: dict[str, Any],
    ) -> tuple[str, str]:
        """Save the final iteration report (JSON and MD)."""
        slug = _slugify(topic)
        innovation_dir = self._resolve_path(f"innovation/{slug}")
        innovation_dir.mkdir(parents=True, exist_ok=True)

        # Save per-round reports
        for rd in rounds_data:
            round_num = rd.get("round", 0)
            round_dir = innovation_dir / "iterations" / f"round_{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)

            # candidates.json
            candidates_data = {
                "topic": topic,
                "round": round_num,
                "generated_at": rd.get("completed_at", ""),
                "candidates": [r.get("candidate", {}) for r in rd.get("results", [])],
            }
            with open(round_dir / "candidates.json", "w", encoding="utf-8") as f:
                json.dump(candidates_data, f, ensure_ascii=False, indent=2)

            # novelty_report.json
            novelty_data = {
                "topic": topic,
                "round": round_num,
                "generated_at": rd.get("completed_at", ""),
                "results": rd.get("results", []),
            }
            with open(round_dir / "novelty_report.json", "w", encoding="utf-8") as f:
                json.dump(novelty_data, f, ensure_ascii=False, indent=2)

            # review_report.json
            review_data = {
                "topic": topic,
                "round": round_num,
                "generated_at": rd.get("completed_at", ""),
                "results": rd.get("results", []),
                "summary": {
                    "total": len(rd.get("results", [])),
                    "proceed": rd.get("counts", {}).get("proceed", 0),
                    "revise": rd.get("counts", {}).get("revise", 0),
                    "drop": rd.get("counts", {}).get("drop", 0),
                },
            }
            with open(round_dir / "review_report.json", "w", encoding="utf-8") as f:
                json.dump(review_data, f, ensure_ascii=False, indent=2)

        # Build final report
        report_data = {
            "topic": topic,
            "topic_slug": slug,
            "generated_at": utc_now(),
            "workflow_version": WORKFLOW_VERSION,
            "prompt_version": PROMPT_VERSION,
            "params": iteration_params,
            "max_rounds": iteration_params.get("max_rounds", 2),
            "rounds_completed": len(rounds_data),
            "stopping_reason": stopping_reason,
            "rounds": [
                {
                    "round": rd.get("round", i),
                    "status": rd.get("status", "completed"),
                    "num_candidates": len(rd.get("results", [])),
                    "counts": rd.get("counts", {}),
                    "proceed_titles": [
                        r.get("candidate", {}).get("title", "?")
                        for r in rd.get("results", [])
                        if r.get("review", {}).get("decision", "").lower() == "proceed"
                    ],
                    "revise_titles": [
                        r.get("candidate", {}).get("title", "?")
                        for r in rd.get("results", [])
                        if r.get("review", {}).get("decision", "").lower() == "revise"
                    ],
                    "drop_titles": [
                        r.get("candidate", {}).get("title", "?")
                        for r in rd.get("results", [])
                        if r.get("review", {}).get("decision", "").lower() == "drop"
                    ],
                    "revised_from": rd.get("revised_from", []),
                }
                for i, rd in enumerate(rounds_data)
            ],
            "final_candidates": final_candidates,
        }

        json_path = innovation_dir / "iteration_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # Markdown report
        md_path = innovation_dir / "iteration_report.md"
        lines = [
            f"# Iteration Report: {topic}\n",
            f"\n**Generated**: {report_data['generated_at']}\n",
            f"**Workflow version**: {WORKFLOW_VERSION}\n",
            f"**Parameters**: max_rounds={iteration_params.get('max_rounds')}, "
            f"min_proceed={iteration_params.get('min_proceed')}, "
            f"revise_top_k={iteration_params.get('revise_top_k')}\n",
            f"\n**Rounds completed**: {len(rounds_data)}\n",
            f"**Stopping reason**: {stopping_reason}\n",
            "\n## Round-by-Round Summary\n",
        ]

        for i, rd in enumerate(rounds_data):
            counts = rd.get("counts", {})
            lines.append(f"\n### Round {rd.get('round', i)}\n")
            lines.append(f"- Status: {rd.get('status', 'completed').upper()}\n")
            lines.append(f"- Total candidates: {len(rd.get('results', []))}\n")
            lines.append(f"- Proceed: {counts.get('proceed', 0)}, ")
            lines.append(f"Revise: {counts.get('revise', 0)}, ")
            lines.append(f"Drop: {counts.get('drop', 0)}\n")
            if rd.get("revised_from"):
                lines.append(f"- Revised from: {', '.join(rd['revised_from'])}\n")

        lines.append("\n## Final Recommended Candidates\n")
        for i, fc in enumerate(final_candidates, 1):
            c = fc.get("candidate", {})
            r = fc.get("review", {})
            parent = c.get("parent_candidate", "")
            round_info = f" (revised, round {c.get('revision_round', '?')})" if parent else ""
            lines.append(f"\n### {i}. {c.get('title', 'Untitled')}{round_info}\n")
            lines.append(f"**Source**: Round {c.get('revision_round', 0)}, parent: {parent or 'original'}\n")
            lines.append(f"**Decision**: {r.get('decision', '?').upper()}\n")
            lines.append(f"**Overall Score**: {r.get('overall_score', '?')}/10\n")
            lines.append(
                f"**Scores**: Novelty={r.get('novelty_score', '?')}, "
                f"Feasibility={r.get('feasibility_score', '?')}, "
                f"Evidence={r.get('evidence_score', '?')}, "
                f"Impact={r.get('impact_score', '?')}, "
                f"Risk={r.get('risk_score', '?')}\n"
            )
            if r.get("reasoning"):
                lines.append(f"**Reasoning**: {r['reasoning']}\n")

        lines.append("\n## All Candidates Across Rounds\n")
        lines.append("| Round | Title | Decision | Overall | Novelty | Feas. | Evid. | Impact | Risk |\n")
        lines.append("|-------|-------|----------|---------|---------|-------|-------|--------|------|\n")
        for rd in rounds_data:
            rnd = rd.get("round", "?")
            for r in rd.get("results", []):
                c = r.get("candidate", {})
                rev = r.get("review", {})
                title = (c.get("title", "?")[:30] + "...") if len(c.get("title", "")) > 30 else c.get("title", "?")
                lines.append(
                    f"| {rnd} | {title} | {rev.get('decision', '?')} | "
                    f"{rev.get('overall_score', '?')} | {rev.get('novelty_score', '?')} | "
                    f"{rev.get('feasibility_score', '?')} | {rev.get('evidence_score', '?')} | "
                    f"{rev.get('impact_score', '?')} | {rev.get('risk_score', '?')} |\n"
                )

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return str(json_path), str(md_path)

    # =============================================================================
    # Iteration Entry Point
    # =============================================================================

    async def _run_iteration(
        self,
        topic: str,
        output_dir: Path,
        metadata: dict[str, Any],
        num_candidates: int,
        search_local: bool,
        search_online: bool,
        max_related: int,
        enable_review: bool,
        top_k: int,
        overwrite: bool,
        max_rounds: int,
        min_proceed: int,
        revise_top_k: int,
        stop_if_no_change: bool,
        landscape_context: str = "",
    ) -> str:
        """Run multi-round iterative workflow."""
        innovation_dir = output_dir

        # If workflow already completed and overwrite=False, return cached results directly
        # This makes repeated identical calls idempotent
        if not overwrite:
            workflow_file = innovation_dir / "workflow.json"
            if workflow_file.exists():
                try:
                    cached = json.loads(workflow_file.read_text(encoding="utf-8"))
                    if cached.get("overall_status") == "completed":
                        iteration_data = cached.get("iteration", {})
                        rounds_completed = iteration_data.get("rounds_completed", 0)
                        stopping_reason = iteration_data.get("stopping_reason", "cached")
                        # Load existing reports
                        iteration_json = innovation_dir / "iteration_report.json"
                        iteration_md = innovation_dir / "iteration_report.md"
                        # Load rounds data from per-round review files
                        rounds_data: list[dict[str, Any]] = []
                        all_proceed_candidates: list[dict[str, Any]] = []
                        if (innovation_dir / "iterations").exists():
                            for round_dir in sorted((innovation_dir / "iterations").iterdir()):
                                if round_dir.is_dir() and round_dir.name.startswith("round_"):
                                    review_file = round_dir / "review_report.json"
                                    if review_file.exists():
                                        rd = json.loads(review_file.read_text(encoding="utf-8"))
                                        results = rd.get("results", [])
                                        rounds_data.append({
                                            "round": rd.get("round", 0),
                                            "status": "completed",
                                            "completed_at": rd.get("generated_at", ""),
                                            "results": results,
                                            "counts": rd.get("summary", {}),
                                        })
                                        # Collect all proceed candidates
                                        for r in results:
                                            if r.get("review", {}).get("decision", "").lower() == "proceed":
                                                all_proceed_candidates.append(r)
                        # Reconstruct top candidates
                        lineage_map: dict[str, dict[str, Any]] = {}
                        for cand in all_proceed_candidates:
                            c = cand.get("candidate", {})
                            lineage_key = c.get("parent_candidate") or c.get("title", "")
                            if not lineage_key:
                                continue
                            score = cand.get("review", {}).get("overall_score", 0)
                            existing = lineage_map.get(lineage_key)
                            if existing is None:
                                lineage_map[lineage_key] = cand
                            else:
                                existing_score = existing.get("review", {}).get("overall_score", 0)
                                existing_round = existing.get("candidate", {}).get("revision_round", 0)
                                cand_round = c.get("revision_round", 0)
                                if (score > existing_score) or (
                                    score == existing_score and cand_round > existing_round
                                ):
                                    lineage_map[lineage_key] = cand
                        final_sorted = sorted(
                            lineage_map.values(),
                            key=lambda x: (
                                x.get("review", {}).get("overall_score", 0),
                                x.get("candidate", {}).get("revision_round", 0),
                            ),
                            reverse=True,
                        )[:top_k]
                        return self._build_iteration_output(
                            topic, rounds_data, final_sorted,
                            stopping_reason, innovation_dir,
                            str(iteration_json), str(iteration_md),
                        )
                except Exception:
                    pass  # Fall through to re-run if cache read fails

        now = utc_now()

        # Update workflow metadata with iteration info
        metadata["iteration"] = {
            "enabled": True,
            "max_rounds": max_rounds,
            "min_proceed": min_proceed,
            "revise_top_k": revise_top_k,
            "stop_if_no_change": stop_if_no_change,
            "rounds_completed": 0,
            "current_round": 0,
            "stopping_reason": None,
        }

        rounds_data: list[dict[str, Any]] = []
        all_proceed_candidates: list[dict[str, Any]] = []

        # Helper to build final candidates: deduplicate by lineage, then sort by score
        def _sorted_top_candidates() -> list[dict[str, Any]]:
            # Deduplicate by lineage: parent_candidate if exists, else own title as key
            lineage_map: dict[str, dict[str, Any]] = {}
            for cand in all_proceed_candidates:
                c = cand.get("candidate", {})
                lineage_key = c.get("parent_candidate") or c.get("title", "")
                if not lineage_key:
                    continue
                score = cand.get("review", {}).get("overall_score", 0)
                existing = lineage_map.get(lineage_key)
                if existing is None:
                    lineage_map[lineage_key] = cand
                else:
                    existing_score = existing.get("review", {}).get("overall_score", 0)
                    existing_round = existing.get("candidate", {}).get("revision_round", 0)
                    cand_round = c.get("revision_round", 0)
                    # Keep higher score; tie-break by later round
                    if (score > existing_score) or (
                        score == existing_score and cand_round > existing_round
                    ):
                        lineage_map[lineage_key] = cand

            # Sort: primary by overall_score desc, secondary by revision_round desc (later revisions win ties)
            sorted_all = sorted(
                lineage_map.values(),
                key=lambda x: (
                    x.get("review", {}).get("overall_score", 0),
                    x.get("candidate", {}).get("revision_round", 0),
                ),
                reverse=True,
            )
            return sorted_all[:top_k]

        # Helper to update stage metadata after round 0
        def _finalize_iteration_stages(
            reason: str,
            stage3_skipped: bool = False,
        ) -> None:
            """Update all stage statuses in metadata before saving."""
            # Stage 1: always completed (candidates generated)
            metadata["stages"]["stage1_generate"].update({
                "status": "completed",
                "completed_at": rounds_data[0]["completed_at"],
                "output_files": {
                    "candidates_json": str(innovation_dir / "candidates.json"),
                    "candidates_md": str(innovation_dir / "candidates.md"),
                },
                "num_candidates": len(initial_candidates),
            })
            # Stage 2: always completed (novelty done for round 0)
            metadata["stages"]["stage2_novelty"].update({
                "status": "completed",
                "completed_at": rounds_data[0]["completed_at"],
                "output_files": {
                    "novelty_json": str(innovation_dir / "novelty_report.json"),
                    "novelty_md": str(innovation_dir / "novelty_report.md"),
                },
                "num_candidates": len(initial_candidates),
            })
            # Stage 3: completed or skipped based on enable_review
            if stage3_skipped or not enable_review:
                metadata["stages"]["stage3_review"].update({
                    "status": "skipped",
                    "completed_at": rounds_data[0]["completed_at"],
                    "note": "enable_review=False" if not enable_review else "Skipped in iteration",
                })
            else:
                metadata["stages"]["stage3_review"].update({
                    "status": "completed",
                    "completed_at": rounds_data[0]["completed_at"],
                    "output_files": {
                        "review_json": str(innovation_dir / "review_report.json"),
                        "review_md": str(innovation_dir / "review_report.md"),
                    },
                    "num_candidates": len(initial_candidates),
                })
            metadata["iteration"]["stopping_reason"] = reason
            metadata["overall_status"] = "completed"

        # =====================================================================
        # Round 0: Generate initial candidates + novelty + review
        # =====================================================================
        context = ""
        if landscape_context:
            context = f"## Landscape Survey Results\n{landscape_context}\n"
        if search_local:
            relevant_papers = await self._find_relevant_papers_async(topic, max_papers=5)
            if relevant_papers:
                context_parts = []
                if context:
                    context_parts.append(context)
                    context_parts.append("\n## Key Local Papers\n")
                for p in relevant_papers:
                    title = p.get("title", "?")
                    summary = p.get("summary", {})
                    if isinstance(summary, dict):
                        problem = summary.get("problem", "")
                        method = summary.get("method", "")
                        if problem or method:
                            context_parts.append(
                                f"- {title}: Problem: {problem[:150]} Method: {method[:150]}"
                            )
                        else:
                            abstract = p.get("abstract", "")[:200]
                            context_parts.append(f"- {title}: {abstract}")
                    else:
                        abstract = p.get("abstract", "")[:200]
                        context_parts.append(f"- {title}: {abstract}")
                context = "\n".join(context_parts)

        # Stage 1: generate candidates
        metadata["stages"]["stage1_generate"]["status"] = "running"
        metadata["stages"]["stage1_generate"]["started_at"] = now
        metadata["stages"]["stage1_generate"]["input_params"] = {
            "num_candidates": num_candidates,
            "search_local": search_local,
        }
        _save_workflow_metadata(metadata, self._workspace, topic)

        initial_candidates = await self._generate_candidates(topic, context, num_candidates)
        if not initial_candidates:
            metadata["stages"]["stage1_generate"]["status"] = "failed"
            metadata["stages"]["stage1_generate"]["error"] = "Failed to generate initial candidates"
            metadata["overall_status"] = "failed"
            _save_workflow_metadata(metadata, self._workspace, topic)
            return f"Error: Failed to generate candidates for topic: {topic}"

        metadata["stages"]["stage1_generate"]["status"] = "completed"
        metadata["stages"]["stage1_generate"]["completed_at"] = utc_now()
        metadata["stages"]["stage1_generate"]["output_files"] = {
            "candidates_json": str(innovation_dir / "candidates.json"),
            "candidates_md": str(innovation_dir / "candidates.md"),
        }
        metadata["stages"]["stage1_generate"]["num_candidates"] = len(initial_candidates)
        _save_workflow_metadata(metadata, self._workspace, topic)

        # Save round 0 candidates to top-level files (for compatibility)
        self._save_candidates(topic, initial_candidates)

        # Stage 2: novelty search
        metadata["stages"]["stage2_novelty"]["status"] = "running"
        metadata["stages"]["stage2_novelty"]["started_at"] = utc_now()
        metadata["stages"]["stage2_novelty"]["input_params"] = {
            "search_local": search_local,
            "search_online": search_online,
            "max_related": max_related,
        }
        _save_workflow_metadata(metadata, self._workspace, topic)

        # Run novelty + review on round 0
        round0_results = await self._run_iteration_round(
            topic, initial_candidates, search_local, search_online, max_related, round_num=0,
            enable_review=enable_review,
        )

        metadata["stages"]["stage2_novelty"]["status"] = "completed"
        metadata["stages"]["stage2_novelty"]["completed_at"] = utc_now()
        metadata["stages"]["stage2_novelty"]["output_files"] = {
            "novelty_json": str(innovation_dir / "novelty_report.json"),
            "novelty_md": str(innovation_dir / "novelty_report.md"),
        }
        metadata["stages"]["stage2_novelty"]["num_candidates"] = len(initial_candidates)
        _save_workflow_metadata(metadata, self._workspace, topic)

        # Stage 3: review (if enabled)
        if enable_review:
            metadata["stages"]["stage3_review"]["status"] = "running"
            metadata["stages"]["stage3_review"]["started_at"] = utc_now()
            metadata["stages"]["stage3_review"]["input_params"] = {"top_k": top_k}
            _save_workflow_metadata(metadata, self._workspace, topic)
            # _run_iteration_round already includes review via _review_candidate
            metadata["stages"]["stage3_review"]["status"] = "completed"
            metadata["stages"]["stage3_review"]["completed_at"] = utc_now()
            metadata["stages"]["stage3_review"]["output_files"] = {
                "review_json": str(innovation_dir / "review_report.json"),
                "review_md": str(innovation_dir / "review_report.md"),
            }
            metadata["stages"]["stage3_review"]["num_candidates"] = len(initial_candidates)
            _save_workflow_metadata(metadata, self._workspace, topic)
        else:
            metadata["stages"]["stage3_review"]["status"] = "skipped"
            metadata["stages"]["stage3_review"]["completed_at"] = utc_now()
            metadata["stages"]["stage3_review"]["note"] = "enable_review=False"
            _save_workflow_metadata(metadata, self._workspace, topic)

        proceed0, revise0, drop0 = _classify_candidates(round0_results)
        all_proceed_candidates.extend(proceed0)

        now_ts = utc_now()
        rounds_data.append({
            "round": 0,
            "status": "completed",
            "started_at": now,
            "completed_at": now_ts,
            "results": round0_results,
            "counts": {"proceed": len(proceed0), "revise": len(revise0), "drop": len(drop0)},
            "revised_from": [],
        })

        metadata["iteration"]["rounds_completed"] = 1
        metadata["iteration"]["current_round"] = 0
        _save_workflow_metadata(metadata, self._workspace, topic)

        # Check stop condition: enough proceed already?
        if len(proceed0) >= min_proceed:
            stopping_reason = f"Found {len(proceed0)} proceed candidates (min_proceed={min_proceed})"
            _finalize_iteration_stages(stopping_reason)
            _save_workflow_metadata(metadata, self._workspace, topic)
            iteration_report_path, iteration_md_path = self._save_iteration_reports(
                topic, rounds_data, _sorted_top_candidates(),
                stopping_reason, metadata["params"],
            )
            return self._build_iteration_output(
                topic, rounds_data, _sorted_top_candidates(),
                stopping_reason, innovation_dir, iteration_report_path, iteration_md_path,
            )

        # =====================================================================
        # Rounds 1 to max_rounds: revise + re-review
        # =====================================================================
        for round_num in range(1, max_rounds + 1):
            now = utc_now()
            metadata["iteration"]["current_round"] = round_num
            _save_workflow_metadata(metadata, self._workspace, topic)

            # Get revise candidates from previous round (up to revise_top_k)
            prev_round = rounds_data[-1]
            prev_revise = [
                r for r in prev_round["results"]
                if r.get("review", {}).get("decision", "").lower() == "revise"
            ]
            to_revise = prev_revise[:revise_top_k]

            if not to_revise:
                stopping_reason = f"Round {round_num}: no revise candidates to iterate on"
                _finalize_iteration_stages(stopping_reason)
                _save_workflow_metadata(metadata, self._workspace, topic)
                break

            # Generate revised candidates
            revised_candidates: list[dict[str, Any]] = []
            revised_from_titles: list[str] = []
            for r in to_revise:
                candidate = r.get("candidate", {})
                analysis = r.get("analysis", {})
                review = r.get("review", {})
                related = r.get("related_papers", [])

                revised = await self._generate_revised_candidate(
                    candidate, analysis, review, related, round_num,
                )
                if revised:
                    revised_candidates.append(revised)
                    revised_from_titles.append(candidate.get("title", "?"))

            if not revised_candidates:
                stopping_reason = f"Round {round_num}: failed to generate revised candidates"
                _finalize_iteration_stages(stopping_reason)
                _save_workflow_metadata(metadata, self._workspace, topic)
                break

            # Run novelty + review on revised candidates
            round_results = await self._run_iteration_round(
                topic, revised_candidates, search_local, search_online, max_related, round_num,
                enable_review=enable_review,
            )

            proceed_n, revise_n, drop_n = _classify_candidates(round_results)
            all_proceed_candidates.extend(proceed_n)

            now_ts = utc_now()
            rounds_data.append({
                "round": round_num,
                "status": "completed",
                "started_at": now,
                "completed_at": now_ts,
                "results": round_results,
                "counts": {"proceed": len(proceed_n), "revise": len(revise_n), "drop": len(drop_n)},
                "revised_from": revised_from_titles,
            })

            metadata["iteration"]["rounds_completed"] = len(rounds_data)
            _save_workflow_metadata(metadata, self._workspace, topic)

            # Check stop conditions (cumulative proceed count)
            if len(all_proceed_candidates) >= min_proceed:
                stopping_reason = f"Round {round_num}: cumulative {len(all_proceed_candidates)} proceed candidates (min_proceed={min_proceed})"
                _finalize_iteration_stages(stopping_reason)
                _save_workflow_metadata(metadata, self._workspace, topic)
                break

            if stop_if_no_change and len(proceed_n) == 0:
                stopping_reason = f"Round {round_num}: no new proceed candidates (stop_if_no_change=True)"
                _finalize_iteration_stages(stopping_reason)
                _save_workflow_metadata(metadata, self._workspace, topic)
                break

            if round_num == max_rounds:
                stopping_reason = f"Reached max_rounds={max_rounds}"
                _finalize_iteration_stages(stopping_reason)
                _save_workflow_metadata(metadata, self._workspace, topic)
                break

        else:
            stopping_reason = f"Completed all {max_rounds} rounds"
            _finalize_iteration_stages(stopping_reason)
            _save_workflow_metadata(metadata, self._workspace, topic)

        # Save final iteration reports using sorted top candidates
        final_sorted = _sorted_top_candidates()
        iteration_report_path, iteration_md_path = self._save_iteration_reports(
            topic, rounds_data, final_sorted,
            metadata["iteration"].get("stopping_reason", stopping_reason),
            metadata["params"],
        )

        workflow_path = _save_workflow_metadata(metadata, self._workspace, topic)

        return self._build_iteration_output(
            topic, rounds_data, final_sorted,
            metadata["iteration"].get("stopping_reason", stopping_reason),
            innovation_dir, iteration_report_path, iteration_md_path,
        )

    def _build_iteration_output(
        self,
        topic: str,
        rounds_data: list[dict[str, Any]],
        final_candidates: list[dict[str, Any]],
        stopping_reason: str,
        innovation_dir: Path,
        iteration_report_path: str,
        iteration_md_path: str,
    ) -> str:
        """Build the text output for an iteration run."""
        output_parts = [
            f"# Innovation Workflow (Iteration Mode): {topic}\n",
            f"\n**Workflow version**: {WORKFLOW_VERSION}\n",
            f"**Prompt version**: {PROMPT_VERSION}\n",
            f"\n**Rounds completed**: {len(rounds_data)}\n",
            f"**Stopping reason**: {stopping_reason}\n",
            f"\n**Output Files**:\n",
            f"- Iteration Report JSON: {iteration_report_path}\n",
            f"- Iteration Report MD: {iteration_md_path}\n",
            f"- Workflow metadata: {str(innovation_dir / 'workflow.json')}\n",
        ]

        output_parts.append("\n## Round Summary\n")
        for rd in rounds_data:
            counts = rd.get("counts", {})
            line = (
                f"- Round {rd.get('round', '?')}: "
                f"Proceed={counts.get('proceed', 0)}, "
                f"Revise={counts.get('revise', 0)}, "
                f"Drop={counts.get('drop', 0)}"
            )
            if rd.get("revised_from"):
                line += f" (revised from: {', '.join(rd['revised_from'][:3])})"
            output_parts.append(line + "\n")

        output_parts.append("\n## Final Recommended Candidates\n")
        for i, fc in enumerate(final_candidates, 1):
            c = fc.get("candidate", {})
            r = fc.get("review", {})
            parent = c.get("parent_candidate", "")
            if parent:
                round_info = f" (revised from '{parent}', round {c.get('revision_round', '?')})"
            else:
                round_info = f" (round {c.get('revision_round', 0)})"
            output_parts.append(
                f"{i}. **{c.get('title', 'Untitled')}**{round_info} "
                f"[{r.get('decision', '?').upper()}] "
                f"(Overall: {r.get('overall_score', '?')}/10)\n"
            )

        return "".join(output_parts)

    async def execute(
        self,
        search_online: bool = True,
        max_related: int = 18,
        enable_review: bool = True,
        top_k: int = 3,
        overwrite: bool = False,
        enable_iteration: bool = False,
        max_rounds: int = 2,
        min_proceed: int = 1,
        revise_top_k: int = 3,
        stop_if_no_change: bool = True,
        enable_landscape: bool = True,
        landscape_max_online: int = 8,
        reviewer_model: str | None = None,
        executor_model: str | None = None,
        **kwargs: Any,
    ) -> str:
        topic = kwargs.get("topic")
        num_candidates = kwargs.get("num_candidates", 5)
        search_local = kwargs.get("search_local", True)
        if not topic:
            return "Error: topic is required"

        if not self._workspace:
            return "Error: workspace not configured"

        slug = _slugify(topic)

        # Prevent concurrent same-topic execution with per-slug lock
        if slug not in InnovationWorkflowTool._locks:
            InnovationWorkflowTool._locks[slug] = asyncio.Lock()
        async with InnovationWorkflowTool._locks[slug]:
            return await self._execute_impl(
                topic, slug, num_candidates, search_local,
                search_online, max_related, enable_review, top_k,
                overwrite, enable_iteration, max_rounds, min_proceed,
                revise_top_k, stop_if_no_change, enable_landscape,
                landscape_max_online, reviewer_model, executor_model,
            )

    async def _execute_impl(
        self,
        topic: str,
        slug: str,
        num_candidates: int,
        search_local: bool,
        search_online: bool,
        max_related: int,
        enable_review: bool,
        top_k: int,
        overwrite: bool,
        enable_iteration: bool,
        max_rounds: int,
        min_proceed: int,
        revise_top_k: int,
        stop_if_no_change: bool,
        enable_landscape: bool,
        landscape_max_online: int,
        reviewer_model: str | None,
        executor_model: str | None,
    ) -> str:
        output_dir = self._resolve_path(f"innovation/{slug}")

        # Store reviewer_model for use by _call_reviewer
        self._reviewer_model = reviewer_model

        # Build workflow metadata with input params
        metadata = _workflow_info()
        iteration_params = {
            "enable_iteration": enable_iteration,
            "max_rounds": max_rounds,
            "min_proceed": min_proceed,
            "revise_top_k": revise_top_k,
            "stop_if_no_change": stop_if_no_change,
        }
        metadata["params"] = {
            "num_candidates": num_candidates,
            "search_local": search_local,
            "search_online": search_online,
            "max_related": max_related,
            "enable_review": enable_review,
            "top_k": top_k,
            "overwrite": overwrite,
            "enable_landscape": enable_landscape,
            "landscape_max_online": landscape_max_online,
        }
        metadata["params"].update(iteration_params)

        # =====================================================================
        # Stage 0: Landscape survey (optional, before everything else)
        # =====================================================================
        landscape_context = ""

        # Auto-enable landscape if no local papers found (ensure context coverage)
        if not enable_landscape and search_local:
            local_papers = self._load_local_papers()
            # Filter papers relevant to topic using simple keyword matching
            if local_papers:
                topic_lower = topic.lower()
                relevant_count = sum(
                    1 for p in local_papers
                    if topic_lower in p.get("title", "").lower()
                    or topic_lower in p.get("abstract", "").lower()
                )
                if relevant_count == 0:
                    enable_landscape = True

        if enable_landscape:
            _, landscape_context = await self._run_landscape_survey(
                topic, output_dir, metadata,
                search_local=search_local,
                max_online=landscape_max_online,
                overwrite=overwrite,
            )

        # =====================================================================
        # Iteration mode: multi-round iterative workflow
        # =====================================================================
        if enable_iteration:
            if not enable_review:
                return (
                    "Error: enable_iteration=True requires enable_review=True. "
                    "The iteration loop depends on review scores to decide which candidates "
                    "to revise. Please set enable_review=True when using enable_iteration."
                )
            return await self._run_iteration(
                topic, output_dir, metadata, num_candidates,
                search_local, search_online, max_related,
                enable_review, top_k, overwrite,
                max_rounds, min_proceed, revise_top_k, stop_if_no_change,
                landscape_context=landscape_context,
            )

        # =====================================================================
        # Standard 3-stage mode (existing logic, unchanged)
        # =====================================================================

        # =====================================================================
        # Determine which stages to run vs reuse
        # =====================================================================
        candidates_exists = (output_dir / "candidates.json").exists()
        novelty_exists = (output_dir / "novelty_report.json").exists()
        review_exists = (output_dir / "review_report.json").exists()

        skip_stage1 = candidates_exists and not overwrite
        skip_stage2 = novelty_exists and not overwrite
        skip_stage3 = review_exists and not overwrite

        # When enable_review=False but novelty already exists, still skip stage3
        if not enable_review and novelty_exists and not overwrite:
            skip_stage3 = True

        # Short-circuit: all results exist and no new work needed
        if skip_stage1 and skip_stage2 and skip_stage3:
            now = utc_now()
            candidates_json_path = str(output_dir / "candidates.json")
            candidates_md_path = str(output_dir / "candidates.md")
            novelty_json_path = str(output_dir / "novelty_report.json")
            novelty_md_path = str(output_dir / "novelty_report.md")

            # Load candidate count
            with open(output_dir / "candidates.json", "r", encoding="utf-8") as f:
                candidates_count = len(json.load(f).get("candidates", []))

            # Mark stage1 as skipped
            metadata["stages"]["stage1_generate"].update({
                "status": "skipped",
                "started_at": now,
                "completed_at": now,
                "input_params": {"num_candidates": num_candidates, "search_local": search_local},
                "output_files": {"candidates_json": candidates_json_path, "candidates_md": candidates_md_path},
                "num_candidates": candidates_count,
                "note": "Reused existing file",
            })

            # Mark stage2 as skipped
            metadata["stages"]["stage2_novelty"].update({
                "status": "skipped",
                "started_at": now,
                "completed_at": now,
                "input_params": {"search_local": search_local, "search_online": search_online, "max_related": max_related},
                "output_files": {"novelty_json": novelty_json_path, "novelty_md": novelty_md_path},
                "num_candidates": candidates_count,
                "note": "Reused existing file",
            })

            # Mark stage3 as skipped (review was already done)
            metadata["stages"]["stage3_review"].update({
                "status": "skipped",
                "started_at": now,
                "completed_at": now,
                "input_params": {"top_k": top_k},
                "output_files": {"review_json": str(output_dir / "review_report.json"), "review_md": str(output_dir / "review_report.md")},
                "num_candidates": candidates_count,
                "note": "Reused existing file",
            })
            metadata["overall_status"] = "completed"
            workflow_path = _save_workflow_metadata(metadata, self._workspace, topic)
            return (
                f"# Innovation Workflow: {topic}\n"
                f"\n**All stages already complete (reused existing results).**\n"
                f"**Output directory**: {output_dir}\n"
                f"**Workflow metadata**: {workflow_path}\n"
                f"\nUse overwrite=True to regenerate all stages."
            )

        # =====================================================================
        # Stage 1: Generate candidates (skip if already exists and not overwrite)
        # =====================================================================
        candidates_json_path = str(output_dir / "candidates.json")
        candidates_md_path = str(output_dir / "candidates.md")
        novelty_json_path = str(output_dir / "novelty_report.json")
        novelty_md_path = str(output_dir / "novelty_report.md")
        review_json_path = ""
        review_md_path = ""

        metadata["stages"]["stage1_generate"]["status"] = "running"
        metadata["stages"]["stage1_generate"]["started_at"] = utc_now()
        metadata["stages"]["stage1_generate"]["input_params"] = {
            "num_candidates": num_candidates,
            "search_local": search_local,
        }
        _save_workflow_metadata(metadata, self._workspace, topic)

        if skip_stage1:
            # Load existing candidates
            with open(output_dir / "candidates.json", "r", encoding="utf-8") as f:
                candidates_data = json.load(f)
            candidates = candidates_data.get("candidates", [])
            metadata["stages"]["stage1_generate"]["status"] = "skipped"
            metadata["stages"]["stage1_generate"]["completed_at"] = utc_now()
            metadata["stages"]["stage1_generate"]["output_files"] = {
                "candidates_json": candidates_json_path,
                "candidates_md": candidates_md_path,
            }
            metadata["stages"]["stage1_generate"]["num_candidates"] = len(candidates)
            metadata["stages"]["stage1_generate"]["note"] = "Reused existing file"

            # Dual-model: external reviewer evaluates existing candidates
            if reviewer_model:
                try:
                    candidates_review_content = await self._call_reviewer(
                        EXTERNAL_REVIEWER_CANDIDATES_PROMPT,
                        {
                            "topic": topic,
                            "candidates_json": json.dumps(candidates[:10], ensure_ascii=False, indent=2),
                        },
                    )
                    if candidates_review_content:
                        ext_review_path = output_dir / "candidates_review.md"
                        content = "# External Reviewer: Candidate Assessment\n\n## Research Topic\n{topic}\n\n## Candidate Reviews\n{content}".format(
                            topic=topic,
                            content=_format_json_as_markdown(candidates_review_content),
                        )
                        with open(ext_review_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except Exception:
                    pass
        else:
            context = ""
            if landscape_context:
                context = f"## Landscape Survey Results\n{landscape_context}\n"
            if search_local:
                relevant_papers = await self._find_relevant_papers_async(topic, max_papers=5)
                if relevant_papers:
                    context_parts = []
                    if context:
                        context_parts.append(context)
                        context_parts.append("\n## Key Local Papers\n")
                    for p in relevant_papers:
                        title = p.get("title", "?")
                        summary = p.get("summary", {})
                        if isinstance(summary, dict):
                            problem = summary.get("problem", "")
                            method = summary.get("method", "")
                            if problem or method:
                                context_parts.append(
                                    f"- {title}: Problem: {problem[:150]} Method: {method[:150]}"
                                )
                            else:
                                abstract = p.get("abstract", "")[:200]
                                context_parts.append(f"- {title}: {abstract}")
                        else:
                            abstract = p.get("abstract", "")[:200]
                            context_parts.append(f"- {title}: {abstract}")
                    context = "\n".join(context_parts)

            candidates = await self._generate_candidates(topic, context, num_candidates)

            if not candidates:
                metadata["stages"]["stage1_generate"]["status"] = "failed"
                metadata["stages"]["stage1_generate"]["error"] = "Failed to generate candidates"
                metadata["overall_status"] = "failed"
                _save_workflow_metadata(metadata, self._workspace, topic)
                return f"Error: Failed to generate candidates for topic: {topic}"

            candidates_json_path, candidates_md_path = self._save_candidates(topic, candidates)
            metadata["stages"]["stage1_generate"]["status"] = "completed"
            metadata["stages"]["stage1_generate"]["completed_at"] = utc_now()
            metadata["stages"]["stage1_generate"]["output_files"] = {
                "candidates_json": candidates_json_path,
                "candidates_md": candidates_md_path,
            }
            metadata["stages"]["stage1_generate"]["num_candidates"] = len(candidates)
            _save_workflow_metadata(metadata, self._workspace, topic)

        # =====================================================================
        # Dual-model: external reviewer evaluates candidates after Stage 1
        # =====================================================================
        candidates_review_content = None
        if reviewer_model and not skip_stage1:
            try:
                candidates_review_content = await self._call_reviewer(
                    EXTERNAL_REVIEWER_CANDIDATES_PROMPT,
                    {
                        "topic": topic,
                        "candidates_json": json.dumps(candidates[:10], ensure_ascii=False, indent=2),
                    },
                )
                if candidates_review_content:
                    ext_review_path = output_dir / "candidates_review.md"
                    content = "# External Reviewer: Candidate Assessment\n\n## Research Topic\n{topic}\n\n## Candidate Reviews\n{content}".format(
                        topic=topic,
                        content=_format_json_as_markdown(candidates_review_content),
                    )
                    with open(ext_review_path, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception:
                pass  # Silently skip if reviewer fails

        # =====================================================================
        # Stage 2: Novelty search (skip if already exists and not overwrite)
        # =====================================================================
        metadata["stages"]["stage2_novelty"]["status"] = "running"
        metadata["stages"]["stage2_novelty"]["started_at"] = utc_now()
        metadata["stages"]["stage2_novelty"]["input_params"] = {
            "search_local": search_local,
            "search_online": search_online,
            "max_related": max_related,
        }
        _save_workflow_metadata(metadata, self._workspace, topic)

        if skip_stage2:
            # Load existing novelty results
            with open(output_dir / "novelty_report.json", "r", encoding="utf-8") as f:
                novelty_data = json.load(f)
            novelty_results = novelty_data.get("results", [])
            metadata["stages"]["stage2_novelty"]["status"] = "skipped"
            metadata["stages"]["stage2_novelty"]["completed_at"] = utc_now()
            metadata["stages"]["stage2_novelty"]["output_files"] = {
                "novelty_json": novelty_json_path,
                "novelty_md": novelty_md_path,
            }
            metadata["stages"]["stage2_novelty"]["num_candidates"] = len(novelty_results)
            metadata["stages"]["stage2_novelty"]["note"] = "Reused existing file"
            _save_workflow_metadata(metadata, self._workspace, topic)
        else:
            # Concurrent novelty search for all candidates
            async def process_candidate_novelty(
                candidate: dict[str, Any],
            ) -> dict[str, Any]:
                """Process novelty search for a single candidate."""
                keywords = candidate.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",")]

                related_papers: list[dict[str, Any]] = []

                if search_local and keywords:
                    query = " ".join(keywords)
                    local_results = await self._find_relevant_papers_async(query, max_papers=max_related)
                    related_papers.extend(local_results)

                if search_online and keywords:
                    query = " OR ".join(keywords[:3])
                    try:
                        arxiv_results = await self._search_arxiv(query, max_related)
                        existing_ids = {p.get("paper_id") for p in related_papers}
                        for p in arxiv_results:
                            if p.get("paper_id") not in existing_ids:
                                related_papers.append(p)
                                existing_ids.add(p.get("paper_id"))
                    except Exception:
                        pass

                analysis = await self._analyze_novelty(candidate, related_papers)

                # Store full paper info for Stage 3 review
                full_related_papers: list[dict[str, Any]] = []
                for p in related_papers[:max_related]:
                    paper_record: dict[str, Any] = {
                        "paper_id": p.get("paper_id", ""),
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "published": p.get("published", ""),
                        "year": p.get("year", "") or (p.get("published", "")[:4] if p.get("published") else ""),
                        "categories": p.get("categories", []) if isinstance(p.get("categories"), list) else [],
                        "authors": p.get("authors", [])[:3] if isinstance(p.get("authors"), list) else [],
                        "source": p.get("source", "arxiv"),
                    }
                    full_related_papers.append(paper_record)

                return {
                    "candidate": candidate,
                    "analysis": analysis,
                    "related_papers": full_related_papers,
                }

            novelty_results = await asyncio.gather(
                *[process_candidate_novelty(c) for c in candidates]
            )

            novelty_json_path, novelty_md_path = self._save_novelty_report(topic, novelty_results)
            metadata["stages"]["stage2_novelty"]["status"] = "completed"
            metadata["stages"]["stage2_novelty"]["completed_at"] = utc_now()
            metadata["stages"]["stage2_novelty"]["output_files"] = {
                "novelty_json": novelty_json_path,
                "novelty_md": novelty_md_path,
            }
            metadata["stages"]["stage2_novelty"]["num_candidates"] = len(novelty_results)
            _save_workflow_metadata(metadata, self._workspace, topic)

        # =====================================================================
        # Stage 3: Review and score (skip if already exists and not overwrite)
        # =====================================================================
        top_candidates = []

        if enable_review:
            metadata["stages"]["stage3_review"]["status"] = "running"
            metadata["stages"]["stage3_review"]["started_at"] = utc_now()
            metadata["stages"]["stage3_review"]["input_params"] = {
                "top_k": top_k,
            }
            _save_workflow_metadata(metadata, self._workspace, topic)

            if skip_stage3:
                # Load existing review results
                with open(output_dir / "review_report.json", "r", encoding="utf-8") as f:
                    review_data = json.load(f)
                review_results = review_data.get("results", [])
                top_candidates = review_data.get("top_candidates", [])
                metadata["stages"]["stage3_review"]["status"] = "skipped"
                metadata["stages"]["stage3_review"]["completed_at"] = utc_now()
                metadata["stages"]["stage3_review"]["output_files"] = {
                    "review_json": str(output_dir / "review_report.json"),
                    "review_md": str(output_dir / "review_report.md"),
                }
                metadata["stages"]["stage3_review"]["num_candidates"] = len(review_results)
                metadata["stages"]["stage3_review"]["note"] = "Reused existing file"
                _save_workflow_metadata(metadata, self._workspace, topic)

                # Dual-model: external reviewer independently assesses at Stage 3 (reused)
                if self._reviewer_model:
                    try:
                        with open(output_dir / "review_report.json", "r", encoding="utf-8") as f:
                            review_data_for_ext = json.load(f)
                        ext_review = await self._call_reviewer(
                            EXTERNAL_REVIEWER_REVIEW_PROMPT,
                            {
                                "topic": topic,
                                "review_report_summary": _summarize_review_report(review_data_for_ext),
                                "candidates_json": json.dumps([r.get("candidate", {}) for r in review_data_for_ext.get("results", [])][:10], ensure_ascii=False, indent=2),
                            },
                        )
                        if ext_review:
                            ext_review_path = output_dir / "review_report_external.md"
                            with open(ext_review_path, "w", encoding="utf-8") as f:
                                f.write("# External Reviewer: Independent Assessment\n\n## Research Topic\n{topic}\n\n## External Reviews\n{content}".format(
                                    topic=topic,
                                    content=_format_json_as_markdown(ext_review),
                                ))
                    except Exception:
                        pass
            else:
                # Concurrent review for all candidates
                async def process_candidate_review(
                    result: dict[str, Any],
                ) -> dict[str, Any]:
                    """Review a single candidate."""
                    candidate = result.get("candidate", {})
                    analysis = result.get("analysis", {})
                    related = result.get("related_papers", [])

                    review = await self._review_candidate(
                        candidate, analysis, related
                    )

                    return {
                        "candidate": candidate,
                        "analysis": analysis,
                        "review": review,
                    }

                review_results = await asyncio.gather(
                    *[process_candidate_review(r) for r in novelty_results]
                )

                # Sort by overall_score descending and pick top_k
                sorted_results = sorted(
                    review_results,
                    key=lambda x: (x.get("review", {}).get("overall_score", 0)),
                    reverse=True,
                )
                top_candidates = sorted_results[:top_k]

                review_json_path, review_md_path = self._save_review_report(
                    topic, review_results, top_candidates
                )
                metadata["stages"]["stage3_review"]["status"] = "completed"
                metadata["stages"]["stage3_review"]["completed_at"] = utc_now()
                metadata["stages"]["stage3_review"]["output_files"] = {
                    "review_json": review_json_path,
                    "review_md": review_md_path,
                }
                metadata["stages"]["stage3_review"]["num_candidates"] = len(review_results)
                _save_workflow_metadata(metadata, self._workspace, topic)

                # Dual-model: external reviewer independently assesses at Stage 3
                if self._reviewer_model:
                    try:
                        review_data_for_ext = {
                            "results": [
                                {
                                    "title": r.get("candidate", {}).get("title", ""),
                                    "novelty_score": r.get("review", {}).get("novelty_score", 0),
                                    "feasibility_score": r.get("review", {}).get("feasibility_score", 0),
                                    "evidence_score": r.get("review", {}).get("evidence_score", 0),
                                    "impact_score": r.get("review", {}).get("impact_score", 0),
                                    "risk_score": r.get("review", {}).get("risk_score", 0),
                                    "overall_score": r.get("review", {}).get("overall_score", 0),
                                    "decision": r.get("review", {}).get("decision", ""),
                                    "reasoning": r.get("review", {}).get("reasoning", ""),
                                }
                                for r in review_results
                            ]
                        }
                        ext_review = await self._call_reviewer(
                            EXTERNAL_REVIEWER_REVIEW_PROMPT,
                            {
                                "topic": topic,
                                "review_report_summary": _summarize_review_report(review_data_for_ext),
                                "candidates_json": json.dumps([r.get("candidate", {}) for r in review_results][:10], ensure_ascii=False, indent=2),
                            },
                        )
                        if ext_review:
                            ext_review_path = output_dir / "review_report_external.md"
                            with open(ext_review_path, "w", encoding="utf-8") as f:
                                f.write("# External Reviewer: Independent Assessment\n\n## Research Topic\n{topic}\n\n## External Reviews\n{content}".format(
                                    topic=topic,
                                    content=_format_json_as_markdown(ext_review),
                                ))
                    except Exception:
                        pass
        else:
            metadata["stages"]["stage3_review"]["status"] = "skipped"
            metadata["stages"]["stage3_review"]["completed_at"] = utc_now()
            metadata["stages"]["stage3_review"]["note"] = "enable_review=False"

        # =====================================================================
        # Finalize workflow metadata
        # =====================================================================
        metadata["overall_status"] = "completed"
        workflow_path = _save_workflow_metadata(metadata, self._workspace, topic)

        # =====================================================================
        # Build summary output
        # =====================================================================
        stage_summaries = []
        for stage_name, stage_info in metadata["stages"].items():
            status = stage_info["status"]
            note = stage_info.get("note", "")
            stage_summaries.append(f"  - {stage_name}: {status.upper()}" + (f" ({note})" if note else ""))

        output_parts = [
            f"# Innovation Workflow Complete: {topic}\n",
            f"\n**Run Info**:\n",
            f"- Workflow version: {WORKFLOW_VERSION}\n",
            f"- Prompt version: {PROMPT_VERSION}\n",
            f"- Workflow metadata: {workflow_path}\n",
            f"\n**Stages**:\n",
            *stage_summaries,
            f"\n**Output Files**:\n",
        ]

        if enable_landscape:
            landscape_json = str(output_dir / "landscape_report.json")
            landscape_md = str(output_dir / "landscape_report.md")
            output_parts.append(f"- Landscape Report JSON: {landscape_json}\n")
            output_parts.append(f"- Landscape Report MD: {landscape_md}\n")

        output_parts.extend([
            f"- Candidates JSON: {candidates_json_path}\n",
            f"- Candidates MD: {candidates_md_path}\n",
            f"- Novelty Report JSON: {novelty_json_path}\n",
            f"- Novelty Report MD: {novelty_md_path}\n",
        ])

        if enable_review:
            output_parts.append(f"- Review Report JSON: {review_json_path}\n")
            output_parts.append(f"- Review Report MD: {review_md_path}\n")

            output_parts.append("\n## Novelty Summary\n")
            for i, result in enumerate(novelty_results, 1):
                candidate = result.get("candidate", {})
                analysis = result.get("analysis", {})
                novelty_level = analysis.get("novelty_level", "unknown")
                level_str = {
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]",
                    "low": "[LOW]",
                    "unknown": "[UNKNOWN]",
                }.get(novelty_level, "[UNKNOWN]")
                output_parts.append(
                    f"{i}. **{candidate.get('title', 'Untitled')}** {level_str}"
                )

            output_parts.append("\n## Review Summary\n")
            for i, tc in enumerate(top_candidates, 1):
                c = tc.get("candidate", {})
                r = tc.get("review", {})
                output_parts.append(
                    f"{i}. **{c.get('title', 'Untitled')}** "
                    f"[{r.get('decision', '?').upper()}] "
                    f"(Overall: {r.get('overall_score', '?')}/10)"
                )
        else:
            output_parts.append("\n## Novelty Summary\n")
            for i, result in enumerate(novelty_results, 1):
                candidate = result.get("candidate", {})
                analysis = result.get("analysis", {})
                novelty_level = analysis.get("novelty_level", "unknown")
                level_str = {
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]",
                    "low": "[LOW]",
                    "unknown": "[UNKNOWN]",
                }.get(novelty_level, "[UNKNOWN]")
                output_parts.append(
                    f"{i}. **{candidate.get('title', 'Untitled')}** {level_str}"
                )

        return "\n".join(output_parts)
