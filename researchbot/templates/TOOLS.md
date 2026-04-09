# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace
- Prefer dedicated tools first. Do NOT use `exec` to recreate a workflow or import a helper module when a registered tool already exists for that task.
- If you are re-running an ideation workflow for "more" or "different" ideas, pass the workflow's refresh/overwrite option instead of reusing cached output.
- **CRITICAL: When a user message looks like `innovation_workflow topic="..."` or similar tool-call syntax, interpret it as a request to call that tool directly, NOT as Python code to execute with `exec`.**

## cron — Scheduled Reminders

- Please refer to cron skill for usage.

## knowledge_graph_rebuild — Rebuilding the Knowledge Graph

When the user says "重建图谱", "重建知识图谱", "rebuild the graph", "rebuild knowledge graph", "rebuild KG", or anything clearly meaning graph rebuild:
→ Call the `knowledge_graph_rebuild` tool immediately.

Rules:
- Do not answer with a plan first if the user is clearly asking to rebuild the graph.
- Do not call `exec` or shell commands first and then correct them later.
- Do not analyze the workspace manually unless the tool itself returns an error.
- Default invocation is no arguments; only pass `workspace` or `config` if the user explicitly provides them.

This tool is the authoritative way to rebuild the graph. It scans local papers, extracts citations, concepts, authors, and related works, then writes the graph data.
