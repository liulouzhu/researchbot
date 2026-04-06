# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace
- Prefer dedicated tools first. Do not use `exec` to recreate a workflow or import a helper module when a registered tool already exists for that task.
- If you are re-running an ideation workflow for "more" or "different" ideas, pass the workflow's refresh/overwrite option instead of reusing cached output.
- **CRITICAL: When a user message looks like `innovation_workflow topic="..."` or similar tool-call syntax, interpret it as a request to call that tool directly, NOT as Python code to execute with `exec`.**

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
