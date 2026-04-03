"""Citation export tool: export paper citations in BibTeX, RIS, CSL-JSON, APA, MLA, GB/T 7714."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.citations import (
    SUPPORTED_FORMATS,
    CitationEntry,
    papers_to_entries,
    render_citation,
    render_citations,
)


class PaperCiteTool(Tool):
    """Export paper citations in various academic formats."""

    name = "paper_cite"
    description = (
        "Export paper citations in standard academic formats: BibTeX, RIS, CSL-JSON, APA, MLA, GB/T 7714. "
        "Reads from local literature storage. Supports single paper, multiple papers, or all saved papers."
    )

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Export citation for a single paper by its ID (e.g. '2401.12345').",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Export citations for multiple papers by their IDs.",
            },
            "paper": {
                "type": "object",
                "description": "A paper dict to export directly (without loading from local storage).",
            },
            "format": {
                "type": "string",
                "enum": sorted(SUPPORTED_FORMATS),
                "default": "bibtex",
                "description": "Citation output format.",
            },
            "output": {
                "type": "string",
                "enum": ["text", "file"],
                "default": "text",
                "description": "'text' returns the citation directly, 'file' saves to a file.",
            },
            "path": {
                "type": "string",
                "description": "File path for 'file' output mode. Defaults to literature/citations/<format>.<ext>.",
            },
        },
        "required": [],
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = Path(workspace) if workspace else None

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _load_all_papers(self) -> list[dict[str, Any]]:
        if not self._workspace:
            return []
        papers_dir = self._resolve_path("literature/papers")
        papers = []
        for fp in papers_dir.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    papers.append(json.load(f))
            except Exception:
                continue
        return papers

    def _default_ext(self, fmt: str) -> str:
        """Return default file extension for a format."""
        return {
            "bibtex": "bib",
            "ris": "ris",
            "csl-json": "json",
            "apa": "txt",
            "mla": "txt",
            "gbt7714": "txt",
        }.get(fmt, "txt")

    async def execute(
        self,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
        paper: dict[str, Any] | None = None,
        format: str = "bibtex",
        output: str = "text",
        path: str | None = None,
        **kwargs: Any,
    ) -> str:
        fmt = format.lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            return f"Error: Unsupported format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}"

        papers: list[dict[str, Any]] = []

        if paper is not None:
            papers.append(paper)

        if paper_id is not None:
            loaded = self._load_local_paper(paper_id)
            if loaded is None:
                return f"Error: Paper '{paper_id}' not found in local storage."
            papers.append(loaded)

        if paper_ids is not None:
            for pid in paper_ids:
                loaded = self._load_local_paper(pid)
                if loaded is None:
                    return f"Error: Paper '{pid}' not found in local storage."
                papers.append(loaded)

        if not papers:
            papers = self._load_all_papers()
            if not papers:
                return "Error: No papers found in local storage. Save papers first with paper_save."

        entries = papers_to_entries(papers)

        if len(entries) == 1:
            result = render_citation(entries[0], fmt)
        else:
            result = render_citations(entries, fmt)

        if output == "file":
            if path:
                out_path = self._resolve_path(path)
            else:
                ext = self._default_ext(fmt)
                out_path = self._resolve_path(f"literature/citations/export.{ext}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result)
            return f"Citation exported to: {out_path}\nFormat: {fmt}\nEntries: {len(entries)}"

        header = f"--- {fmt.upper()} citation ({len(entries)} {'entry' if len(entries) == 1 else 'entries'}) ---\n\n"
        return header + result
