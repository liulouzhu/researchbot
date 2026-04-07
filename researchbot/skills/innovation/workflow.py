"""Innovation workflow entry point for exec-based invocations.

This module provides a simple function interface for generating innovation point
candidates. It wraps InnovationWorkflowTool, exposing only what exec calls need.

Usage (for exec/integration):
    from skills.innovation.workflow import innovation_workflow
    result = innovation_workflow(
        topic="your research topic",
        num_candidates=5,
        overwrite=True,  # CRITICAL: set True to regenerate, False to reuse cached
    )

For direct tool usage in the Agent, use the 'innovation_workflow' tool directly,
NOT this module.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any


def _resolve_workspace() -> Path:
    """Resolve the workspace directory for innovation output."""
    # Try environment variable first
    ws = os.environ.get("NANOBOT_WORKSPACE") or os.environ.get("WORKSPACE")
    if ws:
        return Path(ws)
    # Default workspace
    return Path.home() / ".nanobot" / "workspace"


def _load_provider():
    """Load the LLM provider for the workflow."""
    # Import here to avoid circular deps
    from researchbot.providers.openai_like import OpenAILikeProvider
    from researchbot.config.loader import load_runtime_config

    config = load_runtime_config()
    provider_config = config.providers.get("default") or config.providers.get("openai")
    if provider_config:
        return OpenAILikeProvider(provider_config)
    # Fallback: try to get from environment
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL", "gpt-4o")
    if api_key:
        return OpenAILikeProvider({
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        })
    return None


def innovation_workflow(
    topic: str,
    num_candidates: int = 5,
    search_local: bool = True,
    search_online: bool = True,
    max_related: int = 5,
    enable_review: bool = True,
    top_k: int = 3,
    overwrite: bool = True,  # CHANGED: default True for exec contexts - always regenerate by default
    enable_iteration: bool = False,
    max_rounds: int = 2,
    min_proceed: int = 1,
    revise_top_k: int = 3,
    stop_if_no_change: bool = True,
    enable_landscape: bool = True,
    landscape_max_online: int = 8,
) -> str:
    """Generate innovation point candidates.

    This is a synchronous wrapper around InnovationWorkflowTool for exec/integration
    contexts. For direct Agent tool calls, use InnovationWorkflowTool.execute().

    NOTE: overwrite defaults to True because exec callers (e.g. Agent using python -c "...")
    tend to call this without explicit overwrite=True, which would otherwise silently
    return cached results from the previous run. Set overwrite=False explicitly if you
    want to reuse cached results.

    Args:
        topic: Research topic/problem (required)
        num_candidates: Number of candidates to generate (3-8, default: 5)
        search_local: Search local literature (default: True)
        search_online: Search arXiv online (default: True)
        max_related: Max related papers per candidate (default: 5)
        enable_review: Enable review/scoring (default: True)
        top_k: Number of top candidates to recommend (default: 3)
        overwrite: Overwrite existing cached results (default: True).
            IMPORTANT: Set to True when user wants "another batch" or "再换几个"
            to ensure new candidates are generated instead of reusing cached ones.
        enable_iteration: Enable multi-round iteration (default: False)
        max_rounds: Maximum iteration rounds (default: 2)
        min_proceed: Stop when this many "proceed" candidates found (default: 1)
        revise_top_k: Max revise candidates to revise per round (default: 3)
        stop_if_no_change: Stop if no new proceed candidates in a round (default: True)
        enable_landscape: Enable landscape survey before candidate generation (default: True)
        landscape_max_online: Max papers per online source in landscape survey (default: 8)

    Returns:
        Formatted string output with innovation candidates and review results.
    """
    # Set up paths
    workspace = _resolve_workspace()
    innovation_dir = workspace / "innovation"
    slug = _slugify(topic)
    output_dir = innovation_dir / slug

    # Check if we should use cached results
    candidates_file = output_dir / "candidates.json"
    if candidates_file.exists() and not overwrite:
        # Load and format cached results
        return _format_cached_results(topic, output_dir)

    # Run the async workflow
    provider = _load_provider()
    if provider is None:
        return f"Error: Could not load LLM provider. Set OPENAI_API_KEY or configure providers."

    # Import the tool
    from researchbot.agent.tools.innovation import InnovationWorkflowTool

    # Get semantic search config
    from researchbot.config.loader import load_runtime_config
    config = load_runtime_config()
    literature_config = getattr(config, "literature", None)
    semantic_config = getattr(literature_config, "semantic_search", None) if literature_config else None

    # Get proxy
    proxy = config.tools.web.proxy if hasattr(config, "tools") else None

    tool = InnovationWorkflowTool(
        provider=provider,
        workspace=str(workspace),
        semantic_config=semantic_config,
        proxy=proxy,
    )

    # Run async
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(
        tool.execute(
            topic=topic,
            num_candidates=num_candidates,
            search_local=search_local,
            search_online=search_online,
            max_related=max_related,
            enable_review=enable_review,
            top_k=top_k,
            overwrite=overwrite,
            enable_iteration=enable_iteration,
            max_rounds=max_rounds,
            min_proceed=min_proceed,
            revise_top_k=revise_top_k,
            stop_if_no_change=stop_if_no_change,
            enable_landscape=enable_landscape,
            landscape_max_online=landscape_max_online,
        )
    )
    return result


def _slugify(text: str) -> str:
    """Create a URL-safe slug from text."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:60]


def _format_cached_results(topic: str, output_dir: Path) -> str:
    """Format cached results from JSON files for display."""
    import json

    candidates_file = output_dir / "candidates.json"
    review_file = output_dir / "review_report.json"
    novelty_file = output_dir / "novelty_report.json"

    if not candidates_file.exists():
        return f"Error: No cached results found for topic: {topic}"

    with open(candidates_file, "r", encoding="utf-8") as f:
        candidates_data = json.load(f)
    candidates = candidates_data.get("candidates", [])

    novelty_map = {}
    if novelty_file.exists():
        with open(novelty_file, "r", encoding="utf-8") as f:
            novelty_data = json.load(f)
        for r in novelty_data.get("results", []):
            title = r.get("candidate", {}).get("title", "")
            novelty_map[title] = r.get("analysis", {})

    review_map = {}
    top_candidates = []
    if review_file.exists():
        with open(review_file, "r", encoding="utf-8") as f:
            review_data = json.load(f)
        for r in review_data.get("results", []):
            title = r.get("candidate", {}).get("title", "")
            review_map[title] = r.get("review", {})
        top_candidates = review_data.get("top_candidates", [])

    # Format output
    lines = [
        f"# Innovation Workflow (Cached): {topic}\n",
        f"\n**Note**: Using cached results. Use overwrite=True to regenerate.\n",
    ]

    if top_candidates:
        lines.append("\n## Top Recommended Candidates\n")
        for i, tc in enumerate(top_candidates, 1):
            c = tc.get("candidate", {})
            r = tc.get("review", {})
            lines.append(
                f"{i}. **{c.get('title', 'Untitled')}** "
                f"[{r.get('decision', '?').upper()}] "
                f"(Overall: {r.get('overall_score', '?')}/10)\n"
            )

    if review_map:
        proceed = sum(1 for r in review_map.values() if r.get("decision") == "proceed")
        revise = sum(1 for r in review_map.values() if r.get("decision") == "revise")
        drop = sum(1 for r in review_map.values() if r.get("decision") == "drop")
        lines.append(f"\n**评审总结**: 推荐执行: {proceed}个 | 需要修改: {revise}个 | 不建议: {drop}个\n")

    lines.append(f"\n所有候选结果已保存在: {output_dir}\n")
    return "".join(lines)
