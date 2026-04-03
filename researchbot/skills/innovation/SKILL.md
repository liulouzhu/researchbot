---
name: innovation-workflow
description: Generate innovation point candidates, perform novelty search, and review/score candidates for research topics.
metadata: {"researchbot":{"emoji":"💡"}}
---

# Innovation Workflow

Generate and evaluate innovation point candidates for research topics. This workflow helps identify novel research directions through three stages:

1. **Generate candidates** from a research topic/problem
2. **Search related work** in local literature and arXiv, assess novelty
3. **Review and score** each candidate, produce final recommendations

## When to use (trigger phrases)

Use this skill when the user asks:
- "generate innovation points", "创新点"
- "find research gaps", "研究空白"
- "novelty search", "查新"
- "brainstorm research ideas", "研究方向"
- "suggest research topics", "研究课题"
- "what are some novel research directions", "新颖的研究方向"
- "identify research opportunities", "研究机会"
- "review research ideas", "审阅研究想法"
- "score innovation points", "评分创新点"

## Available Tools

### innovation_workflow

Generate innovation point candidates, perform novelty search, and review/score.

```python
# Basic usage - generates candidates, searches novelty, and reviews all
innovation_workflow(topic="large language model security")

# Skip review stage (faster, only stages 1-2)
innovation_workflow(topic="graph neural networks", enable_review=False)

# Specify number of candidates and top-k recommendations
innovation_workflow(topic="retrieval augmented generation", num_candidates=6, top_k=3)

# Overwrite existing results
innovation_workflow(topic="my research topic", overwrite=True)
```

**Parameters:**
- `topic` (required): Research topic, problem description, or research direction
- `num_candidates`: Number of candidates to generate (3-8, default: 5)
- `search_local`: Search local literature first (default: True)
- `search_online`: Search arXiv online (default: True)
- `max_related`: Max related papers per candidate (default: 5)
- `enable_review`: Enable review/scoring stage (default: True)
- `top_k`: Number of top candidates to recommend (default: 3)
- `overwrite`: Overwrite existing results (default: False)
- `enable_iteration`: Enable multi-round iteration with candidate revision (default: False). **Requires `enable_review=True`.**
- `max_rounds`: Maximum iteration rounds when enable_iteration=True (1-5, default: 2)
- `min_proceed`: Stop iterating when this many "proceed" candidates found (default: 1)
- `revise_top_k`: Max revise candidates to revise per round (default: 3)
- `stop_if_no_change`: Stop if no new proceed candidates in a round (default: True)

> **Note:** `enable_iteration=True` requires `enable_review=True`. The iteration loop depends on review scores to decide which candidates to revise and when to stop.

## Workflow Stages

### Stage 1: Candidate Generation

Generates 3-8 innovation point candidates based on:
- The provided research topic/problem
- Context from existing local literature (if available)

Each candidate includes:
- `title`: Short descriptive title
- `problem`: Specific problem or gap being addressed
- `idea`: Core idea or approach
- `novelty_claim`: What makes this novel
- `expected_value`: Potential impact
- `keywords`: Keywords for related work search
- `related_work_hypothesis`: Hypothesized relationship to existing work

### Stage 2: Novelty Search

For each candidate:
1. Search local literature by keywords (relevance-ranked)
2. Search arXiv online by keywords
3. Analyze novelty against related papers

Novelty analysis includes:
- `similarities`: Common aspects with existing work
- `differences`: Distinguishing features
- `is_duplicate`: Whether it's essentially duplicate
- `is_highly_similar`: Whether it heavily overlaps
- `novelty_level`: "high", "medium", or "low"
- `novelty_conclusion`: Brief assessment
- `gaps_addressed`: Research gaps this addresses

### Stage 3: Review and Scoring (if enable_review=True)

For each candidate, structured review with:
- `novelty_score`: How novel (1-10)
- `feasibility_score`: How feasible given resources (1-10)
- `evidence_score`: How well-supported by evidence (1-10)
- `impact_score`: Potential impact if successful (1-10)
- `risk_score`: Risk level, higher = more risky (1-10)
- `overall_score`: Weighted average (1-10)
- `decision`: "proceed", "revise", or "drop"
- `reasoning`: 2-3 sentence explanation
- `main_risks`: List of key risks
- `recommended_revision`: Specific modifications if "revise"
- `next_step`: Concrete next step

**Overall Score Formula:**
```
overall = novelty*0.20 + feasibility*0.25 + evidence*0.15 + impact*0.20 + (10-risk)*0.20
```

**Decision Rules:**
- `proceed`: Overall promising, reasonable feasibility, acceptable risk
- `revise`: Has potential but needs modification
- `drop`: Fundamental issues that cannot be easily resolved

### Stage 4: Iteration (if enable_iteration=True)

Multi-round iteration that refines "revise" candidates:
1. Round 0: Initial candidates + novelty + review
2. Rounds 1..N: For each "revise" candidate (up to revise_top_k per round):
   - Generate revised candidate using review feedback
   - Re-run novelty search and review on revised version
   - Check stop conditions

**Iteration stops when any of:**
- Reached max_rounds
- Found min_proceed "proceed" candidates
- No new "proceed" candidates in a round (if stop_if_no_change=True)
- No more "revise" candidates to iterate on

**Revised candidates** carry tracking fields:
- `parent_candidate`: Title of original candidate
- `revision_round`: Which round this revision was generated
- `revision_reason`: Which review weakness this addresses
- `revision_summary`: Summary of changes

## Output Files

Results are saved to `innovation/<topic_slug>/`:

| File | Description |
|------|-------------|
| `candidates.json` | Structured JSON of all candidates |
| `candidates.md` | Human-readable markdown of candidates |
| `novelty_report.json` | Structured JSON of novelty analysis |
| `novelty_report.md` | Human-readable novelty report |
| `review_report.json` | Structured JSON of review/scores (if enable_review=True) |
| `review_report.md` | Human-readable review report (if enable_review=True) |
| `iterations/round_N/` | Per-round outputs (if enable_iteration=True) |
| `iteration_report.json` | Structured JSON of all rounds summary (if enable_iteration=True) |
| `iteration_report.md` | Human-readable iteration summary (if enable_iteration=True) |

## Review Report Structure

`review_report.json` contains:
- `topic`, `topic_slug`, `generated_at`
- `review_summary`: counts of proceed/revise/drop
- `results`: per-candidate results with candidate, analysis, review
- `top_candidates`: top-K candidates sorted by overall_score
- `recommendation`: human-readable recommendation text

`review_report.md` contains:
- Overall summary table (all candidates with scores)
- Top candidates with detailed reasoning
- Per-candidate detailed review sections

## Usage Tips

1. **Be specific with topics**: "large language model security" works better than "AI"
2. **Check local literature first**: If you have relevant papers saved, `search_online=False` is faster
3. **Use overwrite carefully**: It regenerates all stages
4. **Review the review scores**: Look at feasibility and risk, not just novelty
5. **Consider top_k**: Default is 3, adjust based on your needs

## Example

```python
# Generate innovation points with full review
result = innovation_workflow(
    topic="efficient inference for large language models",
    num_candidates=6,
    enable_review=True,
    top_k=3,
)
```

## Limitations

- Candidate generation depends on LLM quality and context
- Novelty search is keyword-based, may miss semantic matches
- Review scores are subjective LLM assessments, not ground truth
- No experimental validation of ideas
- Review quality depends on the related papers found
