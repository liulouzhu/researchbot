"""Paper tools using arXiv API: search, get, and save."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.arxiv_client import (
    PaperEntry,
    get_paper_by_id,
    search_arxiv,
    DEFAULT_TIMEOUT,
)
from researchbot.config.schema import MethodExtractionConfig, SemanticSearchConfig
from researchbot.search_index import SearchIndex
from researchbot.utils.helpers import compute_short_id, extract_json_array

logger = logging.getLogger(__name__)


def _format_paper(entry: PaperEntry) -> str:
    """Format a single paper entry into a readable string."""
    authors_str = ", ".join(entry.authors[:5])
    if len(entry.authors) > 5:
        authors_str += f" et al. ({len(entry.authors)} total)"

    lines = [
        f"Title: {entry.title}",
        f"Authors: {authors_str}",
        f"Published: {entry.published[:10]}",
        f"Updated: {entry.updated[:10]}",
        f"Primary Category: {entry.primary_category}",
        f"Categories: {', '.join(entry.categories)}",
        f"arXiv ID: {entry.paper_id}",
    ]

    if entry.doi:
        lines.append(f"DOI: {entry.doi}")
    if entry.journal_ref:
        lines.append(f"Journal: {entry.journal_ref}")

    summary = entry.summary[:500]
    if len(entry.summary) > 500:
        summary += "..."
    lines.append(f"Summary: {summary}")

    lines.append(f"Abstract URL: {entry.abs_url}")
    lines.append(f"PDF URL: {entry.pdf_url}")

    return "\n".join(lines)


def _format_results(entries: list[PaperEntry], query: str) -> str:
    """Format a list of paper entries into output string."""
    if not entries:
        return f"No papers found for query: {query}"

    lines = [f"Search results for: {query}\n"]
    lines.append(f"Found {len(entries)} paper(s):\n")

    for i, entry in enumerate(entries, 1):
        lines.append(f"[{i}] {_format_paper(entry)}")
        if i < len(entries):
            lines.append("")

    lines.append("\nSources: arXiv")
    return "\n".join(lines)


class PaperSearchTool(Tool):
    """Search arXiv for academic papers by topic."""

    name = "paper_search"
    description = "Search arXiv for academic papers by topic. Returns title, authors, abstract, and links."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for arXiv (e.g. 'ti:transformer AND au:hinton' or 'abstract:reinforcement learning')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20)",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "start": {
                "type": "integer",
                "description": "Start index for pagination",
                "minimum": 0,
                "default": 0,
            },
            "sort_by": {
                "type": "string",
                "description": "Sort criterion",
                "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                "default": "relevance",
            },
            "sort_order": {
                "type": "string",
                "description": "Sort order",
                "enum": ["ascending", "descending"],
                "default": "descending",
            },
        },
        "required": ["query"],
    }

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        **kwargs: Any,
    ) -> str:
        try:
            entries = await search_arxiv(
                query=query,
                start=start,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
                timeout=DEFAULT_TIMEOUT,
                proxy=self.proxy,
            )
            return _format_results(entries, query)
        except Exception as e:
            return f"Error: {e}"


def paper_to_dict(entry: PaperEntry) -> dict[str, Any]:
    """Convert PaperEntry to full standardized dict for paper_get output."""
    year = ""
    if entry.published:
        year = entry.published[:4]

    return {
        "paper_id": entry.paper_id,
        "source": "arxiv",
        "external_ids": {
            "arxiv": entry.paper_id,
            "doi": entry.doi or "",
        },
        "title": entry.title,
        "authors": entry.authors,
        "year": year,
        "venue": entry.journal_ref or "",
        "publication_type": "preprint",
        "url": entry.abs_url,
        "pdf_url": entry.pdf_url,
        "abstract": entry.summary,
        "keywords": [],
        "topic_tags": entry.categories,
        "summary": {
            "one_sentence": "",
            "problem": "",
            "method": "",
            "findings": "",
            "limitations": "",
        },
        "provenance": {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "fetch_tool": "paper_get",
            "raw_source_url": entry.abs_url,
        },
        "status": {
            "reviewed": False,
            "pdf_downloaded": False,
            "summary_generated": False,
        },
    }


def _format_paper_detail(entry: PaperEntry) -> str:
    """Format a detailed paper entry for paper_get output."""
    data = paper_to_dict(entry)
    lines = [
        f"Paper ID: {data['paper_id']}",
        f"Source: {data['source']}",
        f"Title: {data['title']}",
        f"Authors: {', '.join(data['authors'])}",
        f"Year: {data['year']}",
        f"Venue: {data['venue'] or 'N/A'}",
        f"Publication Type: {data['publication_type']}",
        f"URL: {data['url']}",
        f"PDF URL: {data['pdf_url']}",
        f"Abstract: {data['abstract']}",
        f"Keywords: {', '.join(data['keywords']) if data['keywords'] else 'None'}",
        f"Topic Tags: {', '.join(data['topic_tags'])}",
        f"Status: reviewed={data['status']['reviewed']}, "
        f"pdf_downloaded={data['status']['pdf_downloaded']}, "
        f"summary_generated={data['status']['summary_generated']}",
        f"Fetched: {data['provenance']['fetched_at']}",
        f"Fetch Tool: {data['provenance']['fetch_tool']}",
    ]
    return "\n".join(lines)


class PaperGetTool(Tool):
    """Get detailed information about a single paper from arXiv by ID or URL."""

    name = "paper_get"
    description = "Get detailed information about a single arXiv paper by its ID or URL."

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID (e.g. '2401.12345' or '2401.12345v2')",
            },
            "url": {
                "type": "string",
                "description": "arXiv URL (e.g. 'https://arxiv.org/abs/2401.12345')",
            },
        },
        "required": [],
    }

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def execute(
        self,
        paper_id: str | None = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not paper_id and not url:
            return "Error: Either paper_id or url must be provided"

        try:
            entry = await get_paper_by_id(
                paper_id=paper_id,
                url=url,
                timeout=DEFAULT_TIMEOUT,
                proxy=self.proxy,
            )
            return _format_paper_detail(entry)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"


QUICK_EXTRACT_METHODS_PROMPT = """从论文标题和摘要中提取可复用的技术方法和模块。

返回 JSON 数组，每个方法包含：
- method_name: 方法名称
- task_type: 任务类型 (classification/detection/generation/embedding/...)
- description: 一句话描述

标题: {title}
摘要: {abstract}

只返回 JSON 数组，不要其他文字。"""


class PaperSaveTool(Tool):
    """Save a paper to local literature knowledge base."""

    name = "paper_save"
    description = "Save a paper's metadata and summary to local literature storage."

    parameters = {
        "type": "object",
        "properties": {
            "paper": {
                "type": "object",
                "description": "Paper object (standardized dict with paper_id, title, authors, etc.)",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic/theme for this paper",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for this paper",
            },
        },
        "required": ["paper"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
        config: Any = None,
        provider: Any = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None
        self._config = config
        self._provider = provider

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._workspace is None or self._semantic_config is None:
            return None
        if self._search_index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._search_index = SearchIndex(db_path, self._semantic_config)
        return self._search_index

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_index(self, index_path: Path) -> dict[str, Any]:
        """Load papers index, return empty structure if not exists."""
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"papers": []}

    def _save_index(self, index_path: Path, index: dict[str, Any]) -> None:
        """Save papers index."""
        self._ensure_dir(index_path)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    async def _quick_extract_methods(
        self,
        paper_id: str,
        title: str,
        abstract: str,
    ) -> list[dict[str, Any]]:
        """Extract methods from paper title and abstract using LLM.

        Returns a list of method records with keys: id, paper_id, method_name,
        task_type, description, module_interface, dependencies, extracted_at.
        """
        if not self._provider:
            return []

        prompt = QUICK_EXTRACT_METHODS_PROMPT.format(
            title=title or "Unknown",
            abstract=abstract[:2000] if abstract else "",
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._provider.chat_with_retry(messages)
            content = response.content or "[]"
            methods = extract_json_array(content)
            if not methods:
                return []
        except Exception:
            return []

        now = datetime.now(timezone.utc).isoformat()
        validated = []
        for m in methods:
            if isinstance(m, dict) and m.get("method_name"):
                method_id = compute_short_id(f"{paper_id}:{m['method_name']}")
                validated.append({
                    "id": method_id,
                    "paper_id": paper_id,
                    "method_name": m.get("method_name", ""),
                    "task_type": m.get("task_type", "other"),
                    "description": m.get("description", ""),
                    "module_interface": m.get("module_interface", ""),
                    "dependencies": m.get("dependencies", []) if isinstance(m.get("dependencies"), list) else [],
                    "extracted_at": now,
                })
        return validated

    async def execute(
        self,
        paper: dict[str, Any],
        topic: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._workspace:
            return "Error: workspace not configured"

        paper_id = paper.get("paper_id")
        if not paper_id:
            return "Error: paper_id is required in paper object"

        literature_dir = self._resolve_path("literature")
        papers_dir = literature_dir / "papers"
        indexes_dir = literature_dir / "indexes"

        self._ensure_dir(papers_dir / f"{paper_id}.json")

        paper_record = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year", ""),
            "topic": topic or "",
            "tags": tags or [],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "url": paper.get("url", ""),
            "pdf_url": paper.get("pdf_url", ""),
            "abstract": paper.get("abstract", ""),
        }

        json_path = papers_dir / f"{paper_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(paper_record, f, ensure_ascii=False, indent=2)

        md_path = papers_dir / f"{paper_id}.md"
        md_content = self._generate_md(paper_record)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        readme_path = literature_dir / "README.md"
        self._update_readme(readme_path, paper_record)

        index_path = indexes_dir / "papers.json"
        index = self._load_index(index_path)
        papers_list = index.get("papers", [])
        existing_idx = next((i for i, p in enumerate(papers_list) if p.get("paper_id") == paper_id), -1)
        if existing_idx >= 0:
            papers_list[existing_idx] = paper_record
        else:
            papers_list.append(paper_record)
        index["papers"] = papers_list
        self._save_index(index_path, index)

        # Update search index
        index_status = "skipped (not configured)"
        search_index = self._get_search_index()
        if search_index is not None:
            try:
                await search_index.initialize()
                sync_result = await search_index.upsert_paper(paper_record)
                index_status = f"ok (content_changed={sync_result['content_changed']})"
                if sync_result["graph_sync_status"] == "failed":
                    index_status = f"graph sync failed: {sync_result['graph_sync_error']}"
                    logger.warning(f"Graph sync failed for {paper_id}: {sync_result['graph_sync_error']}")
            except Exception as e:
                index_status = f"error: {e}"
                logger.warning(f"Search index update failed for {paper_id}: {e}")

        # Auto-extract methods if configured
        if search_index is not None:
            config = getattr(self, '_config', None)
            method_extraction_config: MethodExtractionConfig | None = None
            if config is not None:
                method_extraction_config = getattr(config.literature, 'method_extraction', None)
            if method_extraction_config and method_extraction_config.auto_extract:
                    try:
                        title = paper_record.get("title", "")
                        abstract = paper_record.get("abstract", "")
                        methods = await self._quick_extract_methods(paper_id, title, abstract)
                        for method in methods:
                            await search_index.upsert_method_with_vector(method)
                        if methods:
                            logger.info(f"Extracted {len(methods)} methods from {paper_id}")
                    except Exception as e:
                        logger.warning(f"Failed to extract methods for {paper_id}: {e}")

        # Close search index after all operations
        if search_index is not None:
            search_index.close()

        return (
            f"Saved paper: {paper_id}\n"
            f"  JSON: {json_path}\n"
            f"  MD: {md_path}\n"
            f"  Index updated: {index_path}\n"
            f"  Index sync: {index_status}"
        )

    def _generate_md(self, paper: dict[str, Any]) -> str:
        """Generate markdown file for paper."""
        pdf_path = paper.get("pdf_path", "")
        extracted_path = paper.get("extracted_text_path", "")
        summary = paper.get("summary", {})
        summary_source = paper.get("summary_source", "")

        source_label = ""
        if summary_source:
            source_label = {
                "abstract": " (Abstract only)",
                "full_text": " (Full text)",
                "abstract+full_text": " (Abstract + Full text)",
            }.get(summary_source, f" ({summary_source})")

        lines = [
            f"# {paper['title']}",
            "",
            "## Metadata",
            "",
            f"- **Paper ID**: {paper['paper_id']}",
            f"- **Authors**: {', '.join(paper['authors'])}",
            f"- **Year**: {paper['year']}",
            f"- **Topic**: {paper.get('topic') or 'N/A'}",
            f"- **Tags**: {', '.join(paper['tags']) if paper.get('tags') else 'None'}",
            f"- **URL**: {paper.get('url', 'N/A')}",
            f"- **PDF**: {paper.get('pdf_url', 'N/A')}",
        ]

        if pdf_path:
            lines.append(f"- **Local PDF**: {pdf_path}")
        if extracted_path:
            lines.append(f"- **Extracted Text**: {extracted_path}")

        lines.extend([
            "",
            "## Abstract",
            "",
            paper.get('abstract', 'N/A'),
            "",
            "## Structured Summary",
            source_label,
            "",
            "**One Sentence**: " + summary.get("one_sentence", ""),
            "",
            "**Problem**: " + summary.get("problem", ""),
            "",
            "**Method**: " + summary.get("method", ""),
            "",
            "**Findings**: " + summary.get("findings", ""),
            "",
            "**Limitations**: " + summary.get("limitations", ""),
            "",
            "**Keywords**: " + ", ".join(summary.get("keywords", [])),
            "",
            "**Topic Tags**: " + ", ".join(summary.get("topic_tags", [])),
            "",
            "## Notes",
            "",
            "*Add your notes here*",
            "",
            f"---\n*Saved at: {paper.get('saved_at', datetime.now(timezone.utc).isoformat())}*",
        ])
        return "\n".join(lines)

    def _update_readme(self, readme_path: Path, paper: dict[str, Any]) -> None:
        """Update or create literature README."""
        self._ensure_dir(readme_path)
        lines = []
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        header = "# Literature\n\n"
        if not lines or lines[0] != header:
            lines = [header] + lines

        for i, line in enumerate(lines):
            if line.startswith("## Papers"):
                break
        else:
            lines.append("\n## Papers\n")
            i = len(lines)

        entry = f"- [{paper['paper_id']}] {paper['title']} ({paper.get('year', 'N/A')})\n"
        if entry not in lines:
            lines.insert(i + 1, entry)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


SUMMARIZE_PROMPT = """You are a research paper analyst. Given the metadata of an academic paper, generate a structured summary.

## Paper Information
Title: {title}
Authors: {authors}
Year: {year}
Categories: {categories}
Abstract: {abstract}

## Output Format
Please provide a structured summary in JSON format:
{{
  "one_sentence": "A single sentence summarizing the main contribution (max 30 words)",
  "problem": "What problem does this paper address? (2-3 sentences)",
  "method": "What is the proposed method/approach? (3-4 sentences)",
  "findings": "What are the main findings/results? (3-4 sentences)",
  "limitations": "What are the limitations or potential concerns? (2-3 sentences)",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "topic_tags": ["tag1", "tag2", "tag3"]
}}

## Guidelines
- Extract 3-5 representative keywords that describe the paper's focus
- Use 2-4 topic tags that categorize the research area
- Be concise but informative
- If information is not available in the abstract, indicate that rather than fabricate
- Return ONLY the JSON, no additional text"""

SUMMARIZE_FULLTEXT_PROMPT = """You are a research paper analyst. Given the full text of an academic paper, generate a structured summary.

## Paper Information
Title: {title}
Authors: {authors}
Year: {year}
Categories: {categories}

## Full Text (truncated)
{full_text}

## Output Format
Please provide a structured summary in JSON format:
{{
  "one_sentence": "A single sentence summarizing the main contribution (max 30 words)",
  "problem": "What problem does this paper address? (2-3 sentences)",
  "method": "What is the proposed method/approach? (3-4 sentences)",
  "findings": "What are the main findings/results? (3-4 sentences)",
  "limitations": "What are the limitations or potential concerns? (2-3 sentences)",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "topic_tags": ["tag1", "tag2", "tag3"]
}}

## Guidelines
- Focus on the full text content for accuracy
- Extract 3-5 representative keywords
- Use 2-4 topic tags for categorization
- Be thorough given the full content
- Return ONLY the JSON, no additional text"""


class PaperSummarizeTool(Tool):
    """Generate structured summary for a paper using LLM."""

    name = "paper_summarize"
    description = "Generate a structured summary for an academic paper including one-sentence summary, problem, method, findings, limitations, and keywords."

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID (e.g. '2401.12345' or '2401.12345v2')",
            },
            "url": {
                "type": "string",
                "description": "arXiv URL (e.g. 'https://arxiv.org/abs/2401.12345')",
            },
            "paper": {
                "type": "object",
                "description": "Paper object (standardized dict with title, abstract, authors, etc.)",
            },
            "save": {
                "type": "boolean",
                "description": "Whether to save the summary back to local literature storage",
                "default": True,
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite existing summary in local storage",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        provider: Any = None,
        workspace: str | None = None,
        proxy: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._provider = provider
        self._workspace = Path(workspace) if workspace else None
        self._proxy = proxy
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._workspace is None or self._semantic_config is None:
            return None
        if self._search_index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._search_index = SearchIndex(db_path, self._semantic_config)
        return self._search_index

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Try to load paper from local storage."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_summary(self, paper_id: str, summary: dict[str, Any], paper: dict[str, Any]) -> str:
        """Save summary back to local literature storage."""
        if not self._workspace:
            return ""

        literature_dir = self._resolve_path("literature")
        json_path = literature_dir / "papers" / f"{paper_id}.json"
        md_path = literature_dir / "papers" / f"{paper_id}.md"
        index_path = literature_dir / "indexes" / "papers.json"

        self._ensure_dir(json_path)

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                paper_record = json.load(f)
        else:
            paper_record = dict(paper)

        paper_record["summary"] = summary
        paper_record["summary_generated_at"] = datetime.now(timezone.utc).isoformat()
        paper_record["summary_source"] = summary.get("text_source", "abstract")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(paper_record, f, ensure_ascii=False, indent=2)

        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            md_content = self._update_md_summary(md_path, summary, paper_record)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            papers_list = index.get("papers", [])
            for i, p in enumerate(papers_list):
                if p.get("paper_id") == paper_id:
                    papers_list[i] = paper_record
                    break
            index["papers"] = papers_list
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        return str(json_path)

    def _update_md_summary(self, md_path: Path, summary: dict[str, Any], paper: dict[str, Any]) -> str:
        """Update markdown file with structured summary."""
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        summary_marker = "## Structured Summary"
        found_summary = False
        new_lines = []
        for i, line in enumerate(lines):
            if line.startswith(summary_marker):
                found_summary = True
                new_lines.append(line)
                source = summary.get("text_source", "abstract")
                source_label = {
                    "abstract": "(Based on abstract only)",
                    "full_text": "(Based on full text - extracted PDF)",
                    "abstract+full_text": "(Based on abstract + full text)",
                    "existing": "(Preserved existing summary)",
                }.get(source, f"({source})")
                new_lines.append(f"\n**Source**: {source_label}\n\n")
                new_lines.append("**One Sentence**: ")
                new_lines.append(summary.get("one_sentence", "") + "\n\n")
                new_lines.append("**Problem**: ")
                new_lines.append(summary.get("problem", "") + "\n\n")
                new_lines.append("**Method**: ")
                new_lines.append(summary.get("method", "") + "\n\n")
                new_lines.append("**Findings**: ")
                new_lines.append(summary.get("findings", "") + "\n\n")
                new_lines.append("**Limitations**: ")
                new_lines.append(summary.get("limitations", "") + "\n\n")
                new_lines.append("**Keywords**: " + ", ".join(summary.get("keywords", [])) + "\n")
                new_lines.append("**Topic Tags**: " + ", ".join(summary.get("topic_tags", [])) + "\n")

                j = i + 1
                while j < len(lines) and not lines[j].startswith("## Notes"):
                    j += 1
                if j < len(lines):
                    new_lines.extend(lines[j:])
                break
            new_lines.append(line)

        if not found_summary:
            new_lines.extend(lines)

        return "".join(new_lines)

    def _build_summary_result(
        self,
        summary: dict[str, Any],
        paper_id: str,
        saved: str = "",
        text_source: str = "abstract",
        index_status: str = "",
    ) -> str:
        """Format the summary result as a readable string."""
        source_label = {
            "abstract": "Abstract only",
            "full_text": "Full text (extracted PDF)",
            "abstract+full_text": "Abstract + Full text",
            "existing": "Existing summary (preserved)",
        }.get(text_source, text_source)

        lines = [
            f"Summary for: {paper_id}",
            f"(Source: {source_label})",
            "",
            "## One-Sentence Summary",
            summary.get("one_sentence", ""),
            "",
            "## Problem",
            summary.get("problem", ""),
            "",
            "## Method",
            summary.get("method", ""),
            "",
            "## Findings",
            summary.get("findings", ""),
            "",
            "## Limitations",
            summary.get("limitations", ""),
            "",
            "## Keywords",
            ", ".join(summary.get("keywords", [])),
            "",
            "## Topic Tags",
            ", ".join(summary.get("topic_tags", [])),
        ]
        if saved:
            lines.append("")
            lines.append(f"Saved to: {saved}")
        if index_status:
            lines.append(f"Index sync: {index_status}")
        return "\n".join(lines)

    async def _generate_summary(
        self,
        paper_data: dict[str, Any],
        text_source: str = "abstract",
    ) -> dict[str, Any]:
        """Call LLM to generate structured summary.

        Args:
            paper_data: Paper metadata dict
            text_source: Either "abstract", "full_text", or "abstract+full_text"
        """
        if not self._provider:
            return {
                "one_sentence": "[LLM provider not configured]",
                "problem": "",
                "method": "",
                "findings": "",
                "limitations": "",
                "keywords": [],
                "topic_tags": [],
                "text_source": text_source,
            }

        title = paper_data.get("title", "Unknown")
        authors = ", ".join(paper_data.get("authors", []))
        year = paper_data.get("year", "Unknown")
        categories = ", ".join(paper_data.get("topic_tags", paper_data.get("categories", [])))
        # abstract should be a string; summary might be a dict, so handle that case
        _abstract = paper_data.get("abstract") or ""
        if isinstance(_abstract, str):
            abstract = _abstract
        else:
            abstract = ""

        if text_source == "full_text":
            full_text = paper_data.get("extracted_text", "")
            prompt = SUMMARIZE_FULLTEXT_PROMPT.format(
                title=title,
                authors=authors,
                year=year,
                categories=categories,
                full_text=full_text[:8000] if full_text else "No full text available",
            )
        elif text_source == "abstract+full_text":
            full_text = paper_data.get("extracted_text", "")
            prompt = SUMMARIZE_FULLTEXT_PROMPT.format(
                title=title,
                authors=authors,
                year=year,
                categories=categories,
                full_text=f"Abstract:\n{abstract[:2000]}\n\nFull Text (first 6000 chars):\n{full_text[:6000]}",
            )
        else:
            prompt = SUMMARIZE_PROMPT.format(
                title=title,
                authors=authors,
                year=year,
                categories=categories,
                abstract=abstract[:2000] if abstract else "No abstract available",
            )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._provider.chat_with_retry(messages)
            content = response.content or "{}"

            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                return {
                    "one_sentence": result.get("one_sentence", ""),
                    "problem": result.get("problem", ""),
                    "method": result.get("method", ""),
                    "findings": result.get("findings", ""),
                    "limitations": result.get("limitations", ""),
                    "keywords": result.get("keywords", []),
                    "topic_tags": result.get("topic_tags", []),
                    "text_source": text_source,
                }
        except Exception as e:
            return {
                "one_sentence": f"[Error generating summary: {e}]",
                "problem": "",
                "method": "",
                "findings": "",
                "limitations": "",
                "keywords": [],
                "topic_tags": [],
                "text_source": text_source,
            }

        return {
            "one_sentence": "[Failed to parse LLM response]",
            "problem": "",
            "method": "",
            "findings": "",
            "limitations": "",
            "keywords": [],
            "topic_tags": [],
            "text_source": text_source,
        }

    async def execute(
        self,
        paper_id: str | None = None,
        url: str | None = None,
        paper: dict[str, Any] | None = None,
        save: bool = True,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        if not paper_id and not url and not paper:
            return "Error: Either paper_id, url, or paper must be provided"

        resolved_paper = None
        resolved_paper_id = None

        if paper_id:
            resolved_paper_id = paper_id
            local = self._load_local_paper(paper_id)
            if local:
                resolved_paper = local
            else:
                try:
                    entry = await get_paper_by_id(paper_id=paper_id, proxy=self._proxy)
                    resolved_paper = paper_to_dict(entry)
                except Exception as e:
                    return f"Error fetching paper {paper_id}: {e}"
        elif url:
            try:
                entry = await get_paper_by_id(url=url, proxy=self._proxy)
                resolved_paper = paper_to_dict(entry)
                resolved_paper_id = resolved_paper["paper_id"]
            except Exception as e:
                return f"Error fetching paper from {url}: {e}"
        else:
            resolved_paper = paper
            resolved_paper_id = paper.get("paper_id", "unknown")

        if not resolved_paper:
            return "Error: Could not resolve paper information"

        text_source = "abstract"
        if self._workspace and resolved_paper_id and resolved_paper_id != "unknown":
            extracted_path = self._resolve_path(f"literature/extracted/{resolved_paper_id}.txt")
            if extracted_path.exists():
                try:
                    with open(extracted_path, "r", encoding="utf-8") as f:
                        resolved_paper["extracted_text"] = f.read()
                    text_source = "full_text"
                except Exception:
                    pass

        existing_summary = resolved_paper.get("summary", {})
        if isinstance(existing_summary, dict):
            if existing_summary.get("one_sentence") and existing_summary.get("problem") and not overwrite:
                summary = existing_summary
                summary["text_source"] = "existing"
            else:
                summary = await self._generate_summary(resolved_paper, text_source)
                if overwrite:
                    summary["overwritten"] = True
        else:
            summary = await self._generate_summary(resolved_paper, text_source)

        saved_path = ""
        if save and resolved_paper_id and resolved_paper_id != "unknown":
            saved_path = self._save_summary(resolved_paper_id, summary, resolved_paper)

            # Update search index after saving summary
            search_index = self._get_search_index()
            index_status = "skipped (not configured)"
            if search_index is not None:
                try:
                    await search_index.initialize()
                    sync_result = await search_index.upsert_paper(resolved_paper)
                    search_index.close()
                    index_status = f"ok (content_changed={sync_result['content_changed']})"
                    if sync_result["graph_sync_status"] == "failed":
                        index_status = f"graph sync failed: {sync_result['graph_sync_error']}"
                        logger.warning(
                            f"Graph sync failed for {resolved_paper_id}: {sync_result['graph_sync_error']}"
                        )
                except Exception as e:
                    index_status = f"error: {e}"
                    logger.warning(f"Search index update failed for {resolved_paper_id}: {e}")

        return self._build_summary_result(
            summary, resolved_paper_id, saved_path, text_source, index_status
        )


def _generate_pdf_url(paper_id: str) -> str:
    """Generate arXiv PDF URL from paper ID."""
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


class PaperDownloadPdfTool(Tool):
    """Download arXiv paper PDF to local storage."""

    name = "paper_download_pdf"
    description = "Download arXiv paper PDF to local literature storage."

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID (e.g. '2401.12345' or '2401.12345v2')",
            },
            "url": {
                "type": "string",
                "description": "arXiv URL (e.g. 'https://arxiv.org/abs/2401.12345')",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite existing PDF",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        proxy: str | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._proxy = proxy

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load paper record from local JSON."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    async def execute(
        self,
        paper_id: str | None = None,
        url: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        if not paper_id and not url:
            return "Error: Either paper_id or url must be provided"

        if not self._workspace:
            return "Error: workspace not configured"

        pdfs_dir = self._resolve_path("literature/pdfs")
        self._ensure_dir(pdfs_dir)

        if paper_id:
            pdf_path = pdfs_dir / f"{paper_id}.pdf"
            pdf_url = _generate_pdf_url(paper_id)
        elif url:
            import re
            match = re.search(r"(\d+\.\d+[vV]\d+)", url)
            if not match:
                match = re.search(r"(\d+\.\d+)", url)
            if not match:
                return f"Error: Could not extract paper ID from URL: {url}"
            paper_id = match.group(1)
            pdf_path = pdfs_dir / f"{paper_id}.pdf"
            pdf_url = _generate_pdf_url(paper_id)

        if not overwrite and pdf_path.exists():
            existing_size = pdf_path.stat().st_size
            if existing_size > 0:
                local = self._load_local_paper(paper_id)
                if local:
                    local["pdf_path"] = str(pdf_path)
                    local["pdf_url"] = pdf_url
                    local.setdefault("status", {})
                    local["status"]["pdf_downloaded"] = True
                    json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(local, f, ensure_ascii=False, indent=2)
                return f"PDF already exists: {paper_id} ({existing_size} bytes)"

        try:
            async with httpx.AsyncClient(proxy=self._proxy) as client:
                response = await client.get(pdf_url, timeout=60.0, follow_redirects=True)
                response.raise_for_status()

            self._ensure_dir(pdf_path)
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            local = self._load_local_paper(paper_id)
            if local:
                local["pdf_path"] = str(pdf_path)
                local["pdf_url"] = pdf_url
                local.setdefault("status", {})
                local["status"]["pdf_downloaded"] = True
                json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(local, f, ensure_ascii=False, indent=2)

            file_size = len(response.content)
            return f"Downloaded PDF: {paper_id}\n  PDF: {pdf_path}\n  Size: {file_size} bytes\n  URL: {pdf_url}"

        except httpx.HTTPError as e:
            return f"Error downloading PDF: {e}"
        except Exception as e:
            return f"Error: {e}"


class PaperExtractTextTool(Tool):
    """Extract text from arXiv paper PDF."""

    name = "paper_extract_text"
    description = "Extract text content from a local arXiv paper PDF."

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID to find local PDF",
            },
            "pdf_path": {
                "type": "string",
                "description": "Direct path to PDF file",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite existing extracted text",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        proxy: str | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._proxy = proxy

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load paper record from local JSON."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _find_pdf_path(self, paper_id: str) -> Path | None:
        """Find local PDF path for paper_id."""
        if not self._workspace:
            return None
        pdf_path = self._resolve_path(f"literature/pdfs/{paper_id}.pdf")
        if pdf_path.exists():
            return pdf_path
        return None

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return "Error: pypdf library not installed. Run: pip install pypdf"

        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(text_parts)
        except Exception as e:
            return f"Error extracting text: {e}"

    async def execute(
        self,
        paper_id: str | None = None,
        pdf_path: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        if not paper_id and not pdf_path:
            return "Error: Either paper_id or pdf_path must be provided"

        if not self._workspace:
            return "Error: workspace not configured"

        if pdf_path:
            resolved_pdf = self._resolve_path(pdf_path)
            if paper_id is None:
                paper_id = resolved_pdf.stem
        elif paper_id:
            resolved_pdf = self._find_pdf_path(paper_id)
            if not resolved_pdf:
                local = self._load_local_paper(paper_id)
                if local and local.get("pdf_path"):
                    resolved_pdf = Path(local["pdf_path"])
                else:
                    return f"Error: No PDF found for paper_id: {paper_id}"
        else:
            return "Error: Could not resolve PDF path"

        if not resolved_pdf or not resolved_pdf.exists():
            return f"Error: PDF file not found: {resolved_pdf}"

        extracted_dir = self._resolve_path("literature/extracted")
        self._ensure_dir(extracted_dir)
        txt_path = extracted_dir / f"{paper_id}.txt"

        if not overwrite and txt_path.exists():
            existing_size = txt_path.stat().st_size
            if existing_size > 0:
                local = self._load_local_paper(paper_id)
                if local:
                    local["extracted_text_path"] = str(txt_path)
                    local.setdefault("status", {})
                    local["status"]["full_text_extracted"] = True
                    json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(local, f, ensure_ascii=False, indent=2)
                return f"Extracted text already exists: {paper_id} ({existing_size} chars)"

        text = self._extract_text_from_pdf(resolved_pdf)
        if text.startswith("Error:"):
            return text

        self._ensure_dir(txt_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        local = self._load_local_paper(paper_id)
        if local:
            local["extracted_text_path"] = str(txt_path)
            local.setdefault("status", {})
            local["status"]["full_text_extracted"] = True
            json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(local, f, ensure_ascii=False, indent=2)

        char_count = len(text)
        return f"Extracted text: {paper_id}\n  Text: {txt_path}\n  Characters: {char_count}"


# =============================================================================
# Prompt templates for paper comparison and review
# =============================================================================

COMPARE_PROMPT = """You are a research paper analyst. Given a list of academic papers with their summaries, generate a structured comparison.

## Topic (if provided)
{topic}

## Papers
{papers_json}

## Output Format
Please provide a structured comparison in JSON format:
{{
  "topic": "{topic}" or null,
  "papers": [
    {{"paper_id": "...", "title": "...", "year": "...", "key_takeaway": "..."}}
  ],
  "comparison_dimensions": {{
    "problem": [
      {{"paper_id": "...", "content": "How this paper addresses the problem (2-3 sentences)", "source": "paper_id or title"}}
    ],
    "method": [
      {{"paper_id": "...", "content": "The method used (2-3 sentences)", "source": "paper_id or title"}}
    ],
    "findings": [
      {{"paper_id": "...", "content": "Key findings (2-3 sentences)", "source": "paper_id or title"}}
    ],
    "limitations": [
      {{"paper_id": "...", "content": "Limitations (1-2 sentences)", "source": "paper_id or title"}}
    ],
    "categories": [
      {{"paper_id": "...", "content": "Categories/tags", "source": "paper_id or title"}}
    ]
  }},
  "common_patterns": [
    "Pattern 1 observed across multiple papers",
    "Pattern 2 observed across multiple papers"
  ],
  "major_differences": [
    "Key difference 1 between papers",
    "Key difference 2 between papers"
  ],
  "research_gaps": [
    "Gap 1: underexplored aspect",
    "Gap 2: conflicting findings that need resolution"
  ]
}}

## Guidelines
- Each comparison item MUST include a "source" field referencing a specific paper_id or title
- Be specific and analytical, not generic
- Identify concrete common patterns and major differences
- Research gaps should be actionable insights about what needs further investigation
- Return ONLY the JSON, no additional text"""

REVIEW_PROMPT = """You are a research paper analyst and technical writer. Given a collection of academic papers related to a topic, write a structured literature review.

## Topic
{topic}

## Papers
{papers_json}

## Output Format
Please provide a structured literature review in JSON format:
{{
  "topic": "{topic}",
  "introduction": "2-3 paragraph introduction to the research area and why it matters",
  "papers_summary": [
    {{
      "paper_id": "...",
      "title": "...",
      "year": "...",
      "contribution": "What this paper contributes to the field (2-3 sentences)",
      "method_summary": "Brief method overview",
      "key_findings": ["finding 1", "finding 2", "finding 3"]
    }}
  ],
  "thematic_analysis": [
    {{
      "theme": "Theme name (e.g., 'Optimization Methods', 'Evaluation Metrics')",
      "description": "2-3 sentences describing this theme across papers",
      "supporting_papers": ["paper_id 1", "paper_id 2"],
      "citations": ["According to [paper_id 1], ...", "In contrast, [paper_id 2] argues that ..."]
    }}
  ],
  "synthesis": {{
    "common_findings": ["Finding that appears consistently across papers"],
    "conflicting_findings": ["Where papers disagree and on what"],
    "methodological_approaches": ["Overview of main methodologies used"]
  }},
  "research_gaps": [
    {{
      "gap": "Description of the gap",
      "implication": "Why this gap matters for the field",
      "potential_directions": ["Possible research direction 1", "Possible research direction 2"]
    }}
  ],
  "conclusion": "2-3 paragraph conclusion summarizing the current state and future directions"
}}

## Guidelines
- Write in academic style but keep it accessible
- EVERY claim MUST be attributed to specific papers using paper_id citations like [paper_id]
- Do not write free-form prose that cannot be traced back to sources
- Group papers by themes when doing thematic analysis
- Be critical and identify both strengths and weaknesses
- Research gaps should be specific and actionable
- Return ONLY the JSON, no additional text"""


def _slugify(text: str) -> str:
    """Create a URL-safe slug from text."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:60]


class PaperCompareTool(Tool):
    """Compare multiple papers structurally across key dimensions."""

    name = "paper_compare"
    description = "Compare multiple academic papers across problem, method, findings, limitations, and categories dimensions. If a topic is provided without paper_ids, finds relevant papers from local literature storage."

    parameters = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paper IDs to compare (arXiv IDs without version)",
            },
            "papers": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of paper objects (with paper_id, title, summary fields) to compare",
            },
            "topic": {
                "type": "string",
                "description": "Topic to find relevant papers for comparison (used when paper_ids/papers not provided)",
            },
            "max_papers": {
                "type": "integer",
                "description": "Maximum number of papers to include (default: 5)",
                "default": 5,
            },
        },
    }

    def __init__(
        self,
        workspace: str | None = None,
        provider: Any = None,
        proxy: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._provider = provider
        self._proxy = proxy
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None

    def _resolve_path(self, relative: str) -> Path:
        if not self._workspace:
            return Path(relative)
        p = self._workspace / relative
        return p

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load a single paper from local storage."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_all_papers(self) -> list[dict[str, Any]]:
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

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._workspace is None or self._semantic_config is None:
            return None
        if self._search_index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._search_index = SearchIndex(db_path, self._semantic_config)
        return self._search_index

    def _find_relevant_papers(self, topic: str, max_papers: int = 5) -> list[dict[str, Any]]:
        """Find papers relevant to a topic from local storage.

        First tries semantic search if available, falls back to keyword matching.
        """
        # Try semantic search first
        search_index = self._get_search_index()
        if search_index is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't use await here
                    pass
                else:
                    results = loop.run_until_complete(search_index.initialize())
                    results = loop.run_until_complete(
                        search_index.search(topic, top_k=max_papers, rerank=False)
                    )
                    search_index.close()
                    if results:
                        # Load full paper data for each result
                        papers = []
                        for r in results:
                            paper_id = r.get("paper_id")
                            if paper_id:
                                paper = self._load_local_paper(paper_id)
                                if paper:
                                    papers.append(paper)
                        if papers:
                            return papers
            except Exception:
                pass

        # Fallback to keyword matching
        all_papers = self._load_all_papers()
        if not all_papers:
            return []

        topic_lower = topic.lower()
        scored = []
        for paper in all_papers:
            score = 0
            title = paper.get("title", "").lower()
            summary = paper.get("summary", {})
            if isinstance(summary, dict):
                keywords = " ".join(summary.get("keywords", []))
                topic_tags = " ".join(summary.get("topic_tags", []))
                one_sentence = summary.get("one_sentence", "")
            else:
                keywords = ""
                topic_tags = ""
                one_sentence = ""

            abstract = paper.get("abstract", "").lower()
            categories = " ".join(paper.get("categories", []))

            # Score based on keyword/tag matches
            for kw in topic_lower.split():
                if kw in title:
                    score += 5
                if kw in keywords.lower():
                    score += 3
                if kw in topic_tags.lower():
                    score += 3
                if kw in one_sentence.lower():
                    score += 2
                if kw in abstract:
                    score += 1
                if kw in categories.lower():
                    score += 1

            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored[:max_papers] if s > 0]

    def _build_papers_json(self, papers: list[dict[str, Any]]) -> str:
        """Build a JSON string representation of papers for the prompt."""
        papers_info = []
        for p in papers:
            paper_id = p.get("paper_id", "unknown")
            title = p.get("title", "Unknown")
            year = p.get("published", "")[:4] if p.get("published") else (p.get("year", ""))
            summary = p.get("summary", {})
            if isinstance(summary, dict):
                problem = summary.get("problem", "")
                method = summary.get("method", "")
                findings = summary.get("findings", "")
                limitations = summary.get("limitations", "")
                keywords = summary.get("keywords", [])
                topic_tags = summary.get("topic_tags", [])
            else:
                problem = method = findings = limitations = ""
                keywords = topic_tags = []

            abstract = p.get("abstract", p.get("summary", {}).get("problem", "") if isinstance(p.get("summary"), dict) else "")

            papers_info.append({
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "authors": p.get("authors", [])[:3],
                "abstract": (abstract[:500] + "...") if abstract and len(abstract) > 500 else abstract,
                "problem": problem,
                "method": method,
                "findings": findings,
                "limitations": limitations,
                "keywords": keywords,
                "topic_tags": topic_tags,
                "categories": p.get("categories", []),
            })
        return json.dumps(papers_info, ensure_ascii=False, indent=2)

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM to generate comparison."""
        if not self._provider:
            return {
                "error": "LLM provider not configured",
                "topic": None,
                "papers": [],
                "comparison_dimensions": {},
                "common_patterns": [],
                "major_differences": [],
                "research_gaps": [],
            }

        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self._provider.chat_with_retry(messages)
            content = response.content or "{}"
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            return {"error": str(e)}

        return {"error": "Failed to parse LLM response"}

    async def execute(
        self,
        paper_ids: list[str] | None = None,
        papers: list[dict[str, Any]] | None = None,
        topic: str | None = None,
        max_papers: int = 5,
        **kwargs: Any,
    ) -> str:
        # Validate: need either topic or explicit papers
        if not paper_ids and not papers and not topic:
            return "Error: Provide paper_ids, papers, or topic for comparison"

        if not self._workspace:
            return "Error: workspace not configured"

        # Collect papers to compare
        compare_papers: list[dict[str, Any]] = []

        if papers:
            compare_papers.extend(papers)

        if paper_ids:
            for pid in paper_ids:
                paper = self._load_local_paper(pid)
                if paper:
                    compare_papers.append(paper)

        if not compare_papers and topic:
            # Find relevant papers from local storage based on topic
            compare_papers = self._find_relevant_papers(topic, max_papers)

        if not compare_papers:
            return f"Error: No papers found for comparison. Topic: {topic}"

        # Limit number of papers
        compare_papers = compare_papers[:max_papers]

        # Build prompt
        papers_json = self._build_papers_json(compare_papers)
        prompt = COMPARE_PROMPT.format(
            topic=topic or "General comparison",
            papers_json=papers_json,
        )

        result = await self._call_llm(prompt)

        if "error" in result:
            return f"Error generating comparison: {result['error']}"

        # Build readable output
        output_parts = [f"# Paper Comparison: {topic or 'General'}\n"]

        if result.get("papers"):
            output_parts.append("\n## Papers Compared")
            for p in result["papers"]:
                output_parts.append(f"- **{p.get('paper_id', '?')}**: {p.get('title', '?')} ({p.get('year', '?')}) - {p.get('key_takeaway', '')}")

        if result.get("comparison_dimensions"):
            dims = result["comparison_dimensions"]
            output_parts.append("\n## Comparison Dimensions\n")

            for dim_name in ["problem", "method", "findings", "limitations", "categories"]:
                if dim_name in dims:
                    output_parts.append(f"\n### {dim_name.capitalize()}")
                    for item in dims[dim_name]:
                        source = item.get("source", "unknown")
                        content = item.get("content", "")
                        output_parts.append(f"- [{source}]: {content}")

        if result.get("common_patterns"):
            output_parts.append("\n## Common Patterns")
            for pattern in result["common_patterns"]:
                output_parts.append(f"- {pattern}")

        if result.get("major_differences"):
            output_parts.append("\n## Major Differences")
            for diff in result["major_differences"]:
                output_parts.append(f"- {diff}")

        if result.get("research_gaps"):
            output_parts.append("\n## Research Gaps")
            for gap in result["research_gaps"]:
                output_parts.append(f"- {gap}")

        return "\n".join(output_parts)

    def get_papers_for_comparison(self) -> list[dict[str, Any]]:
        """Return list of available papers in local storage for selection."""
        return self._load_all_papers()


class PaperReviewTool(Tool):
    """Generate a structured literature review for a topic based on locally saved papers."""

    name = "paper_review"
    description = "Generate a structured literature review for a research topic using locally saved papers. Saves review to literature/reviews/<topic_slug>.json and .md."

    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Research topic to generate literature review for",
            },
            "max_papers": {
                "type": "integer",
                "description": "Maximum number of papers to include (default: 10)",
                "default": 10,
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite existing review if it exists (default: False)",
                "default": False,
            },
        },
        "required": ["topic"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        provider: Any = None,
        proxy: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._provider = provider
        self._proxy = proxy
        self._semantic_config = semantic_config
        self._search_index: SearchIndex | None = None

    def _resolve_path(self, relative: str) -> Path:
        if not self._workspace:
            return Path(relative)
        return self._workspace / relative

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load a single paper from local storage."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_all_papers(self) -> list[dict[str, Any]]:
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

    def _get_search_index(self) -> SearchIndex | None:
        """Get or create the search index."""
        if self._workspace is None or self._semantic_config is None:
            return None
        if self._search_index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._search_index = SearchIndex(db_path, self._semantic_config)
        return self._search_index

    def _find_relevant_papers(self, topic: str, max_papers: int = 10) -> list[dict[str, Any]]:
        """Find papers relevant to a topic from local storage.

        First tries semantic search if available, falls back to keyword matching.
        """
        # Try semantic search first
        search_index = self._get_search_index()
        if search_index is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't use await here
                    pass
                else:
                    loop.run_until_complete(search_index.initialize())
                    results = loop.run_until_complete(
                        search_index.search(topic, top_k=max_papers, rerank=False)
                    )
                    search_index.close()
                    if results:
                        # Load full paper data for each result
                        papers = []
                        for r in results:
                            paper_id = r.get("paper_id")
                            if paper_id:
                                paper = self._load_local_paper(paper_id)
                                if paper:
                                    papers.append(paper)
                        if papers:
                            return papers
            except Exception:
                pass

        # Fallback to keyword matching
        all_papers = self._load_all_papers()
        if not all_papers:
            return []

        topic_lower = topic.lower()
        scored = []
        for paper in all_papers:
            score = 0
            title = paper.get("title", "").lower()
            summary = paper.get("summary", {})
            if isinstance(summary, dict):
                keywords = " ".join(summary.get("keywords", []))
                topic_tags = " ".join(summary.get("topic_tags", []))
                one_sentence = summary.get("one_sentence", "")
            else:
                keywords = ""
                topic_tags = ""
                one_sentence = ""

            abstract = paper.get("abstract", "").lower()
            categories = " ".join(paper.get("categories", []))

            # Score based on keyword/tag matches
            for kw in topic_lower.split():
                if kw in title:
                    score += 5
                if kw in keywords.lower():
                    score += 3
                if kw in topic_tags.lower():
                    score += 3
                if kw in one_sentence.lower():
                    score += 2
                if kw in abstract:
                    score += 1
                if kw in categories.lower():
                    score += 1

            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored[:max_papers] if s > 0]

    def _build_papers_json(self, papers: list[dict[str, Any]]) -> str:
        """Build a JSON string representation of papers for the prompt."""
        papers_info = []
        for p in papers:
            paper_id = p.get("paper_id", "unknown")
            title = p.get("title", "Unknown")
            year = p.get("published", "")[:4] if p.get("published") else (p.get("year", ""))
            summary = p.get("summary", {})
            if isinstance(summary, dict):
                problem = summary.get("problem", "")
                method = summary.get("method", "")
                findings = summary.get("findings", "")
                limitations = summary.get("limitations", "")
                keywords = summary.get("keywords", [])
                topic_tags = summary.get("topic_tags", [])
            else:
                problem = method = findings = limitations = ""
                keywords = topic_tags = []

            abstract = p.get("abstract", "")

            papers_info.append({
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "authors": p.get("authors", [])[:5],
                "abstract": (abstract[:800] + "...") if abstract and len(abstract) > 800 else abstract,
                "problem": problem,
                "method": method,
                "findings": findings,
                "limitations": limitations,
                "keywords": keywords,
                "topic_tags": topic_tags,
                "categories": p.get("categories", []),
            })
        return json.dumps(papers_info, ensure_ascii=False, indent=2)

    def _save_review(self, topic: str, review_data: dict[str, Any], papers: list[dict[str, Any]]) -> tuple[str, str]:
        """Save review to JSON and MD files."""
        if not self._workspace:
            return "", ""

        slug = _slugify(topic)
        reviews_dir = self._resolve_path("literature/reviews")
        reviews_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        review_data["topic"] = topic
        review_data["topic_slug"] = slug
        review_data["review_generated_at"] = datetime.now(timezone.utc).isoformat()
        review_data["papers_used"] = [
            {"paper_id": p.get("paper_id"), "title": p.get("title")}
            for p in papers
        ]

        # Save JSON
        json_path = reviews_dir / f"{slug}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)

        # Save Markdown
        md_path = reviews_dir / f"{slug}.md"
        self._generate_md(md_path, topic, review_data, papers)

        return str(json_path), str(md_path)

    def _generate_md(self, md_path: Path, topic: str, review: dict[str, Any], papers: list[dict[str, Any]]) -> None:
        """Generate markdown version of the review."""
        lines = [
            f"# Literature Review: {topic}\n",
            f"\n**Generated**: {review.get('review_generated_at', 'N/A')}\n",
            f"**Papers reviewed**: {len(papers)}\n",
        ]

        # Papers used section
        lines.append("\n## Papers Reviewed\n")
        for p in papers:
            pid = p.get("paper_id", "?")
            title = p.get("title", "?")
            year = p.get("published", "")[:4] if p.get("published") else (p.get("year", "?"))
            lines.append(f"- [{pid}] {title} ({year})\n")

        # Introduction
        if review.get("introduction"):
            lines.append(f"\n## Introduction\n\n{review['introduction']}\n")

        # Papers summary
        if review.get("papers_summary"):
            lines.append("\n## Papers Summary\n")
            for ps in review["papers_summary"]:
                pid = ps.get("paper_id", "?")
                title = ps.get("title", "?")
                year = ps.get("year", "?")
                lines.append(f"\n### [{pid}] {title} ({year})\n")
                if ps.get("contribution"):
                    lines.append(f"**Contribution**: {ps['contribution']}\n")
                if ps.get("method_summary"):
                    lines.append(f"**Method**: {ps['method_summary']}\n")
                if ps.get("key_findings"):
                    lines.append("**Key Findings**:\n")
                    for f in ps["key_findings"]:
                        lines.append(f"- {f}\n")

        # Thematic analysis
        if review.get("thematic_analysis"):
            lines.append("\n## Thematic Analysis\n")
            for theme in review["thematic_analysis"]:
                theme_name = theme.get("theme", "?")
                description = theme.get("description", "")
                supporting = theme.get("supporting_papers", [])
                citations = theme.get("citations", [])

                lines.append(f"\n### {theme_name}\n")
                lines.append(f"{description}\n")
                if supporting:
                    lines.append(f"**Supporting papers**: {', '.join(supporting)}\n")
                for cit in citations:
                    lines.append(f"- {cit}\n")

        # Synthesis
        if review.get("synthesis"):
            synth = review["synthesis"]
            lines.append("\n## Synthesis\n")

            if synth.get("common_findings"):
                lines.append("\n**Common Findings**:\n")
                for f in synth["common_findings"]:
                    lines.append(f"- {f}\n")

            if synth.get("conflicting_findings"):
                lines.append("\n**Conflicting Findings**:\n")
                for f in synth["conflicting_findings"]:
                    lines.append(f"- {f}\n")

            if synth.get("methodological_approaches"):
                lines.append("\n**Methodological Approaches**:\n")
                for m in synth["methodological_approaches"]:
                    lines.append(f"- {m}\n")

        # Research gaps
        if review.get("research_gaps"):
            lines.append("\n## Research Gaps\n")
            for i, gap in enumerate(review["research_gaps"], 1):
                lines.append(f"\n### Gap {i}: {gap.get('gap', '?')}\n")
                lines.append(f"**Why it matters**: {gap.get('implication', 'N/A')}\n")
                if gap.get("potential_directions"):
                    lines.append("**Potential Directions**:\n")
                    for d in gap["potential_directions"]:
                        lines.append(f"- {d}\n")

        # Conclusion
        if review.get("conclusion"):
            lines.append(f"\n## Conclusion\n\n{review['conclusion']}\n")

        # References
        lines.append("\n## References\n")
        for p in papers:
            pid = p.get("paper_id", "?")
            title = p.get("title", "?")
            authors = p.get("authors", [])
            year = p.get("published", "")[:4] if p.get("published") else (p.get("year", "?"))
            authors_str = ", ".join(authors[:5]) if authors else "Unknown"
            if len(authors) > 5:
                authors_str += " et al."
            lines.append(f"- [{pid}] {authors_str}. \"{title}\". {year}.\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM to generate review."""
        if not self._provider:
            return {
                "error": "LLM provider not configured",
                "topic": None,
                "introduction": "",
                "papers_summary": [],
                "thematic_analysis": [],
                "synthesis": {},
                "research_gaps": [],
                "conclusion": "",
            }

        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self._provider.chat_with_retry(messages)
            content = response.content or "{}"
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            return {"error": str(e)}

        return {"error": "Failed to parse LLM response"}

    async def execute(
        self,
        topic: str,
        max_papers: int = 10,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        if not topic:
            return "Error: topic is required"

        if not self._workspace:
            return "Error: workspace not configured"

        slug = _slugify(topic)

        # Check if review already exists
        if not overwrite:
            existing = self._resolve_path(f"literature/reviews/{slug}.json")
            if existing.exists():
                return f"Review already exists for topic: {topic} at {existing}. Use overwrite=True to regenerate."

        # Find relevant papers
        papers = self._find_relevant_papers(topic, max_papers)

        if not papers:
            return f"Error: No papers found in local literature storage for topic: {topic}. Please save some papers first using paper_save."

        # Build prompt
        papers_json = self._build_papers_json(papers)
        prompt = REVIEW_PROMPT.format(
            topic=topic,
            papers_json=papers_json,
        )

        result = await self._call_llm(prompt)

        if "error" in result:
            return f"Error generating review: {result['error']}"

        # Save review
        json_path, md_path = self._save_review(topic, result, papers)

        if not json_path:
            return "Error: Could not save review (workspace may not be configured)"

        # Build output
        output_parts = [
            f"# Literature Review: {topic}\n",
            f"\n**Papers used**: {len(papers)}\n",
            f"\n**Saved to**:\n- JSON: {json_path}\n- Markdown: {md_path}\n",
        ]

        if result.get("introduction"):
            output_parts.append(f"\n## Introduction\n{result['introduction'][:500]}...\n")

        if result.get("synthesis"):
            synth = result["synthesis"]
            if synth.get("common_findings"):
                output_parts.append(f"\n**Common Findings** ({len(synth['common_findings'])} items)")
            if synth.get("research_gaps"):
                output_parts.append(f"\n**Research Gaps** identified: {len(result['research_gaps'])}")

        if result.get("conclusion"):
            output_parts.append(f"\n## Conclusion\n{result['conclusion'][:300]}...\n")

        return "\n".join(output_parts)
