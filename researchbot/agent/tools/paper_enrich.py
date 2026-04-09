"""Paper enrichment tool - enriches existing papers with Crossref/OpenAlex metadata."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.crossref_client import get_crossref_work, search_crossref
from researchbot.agent.tools.openalex_client import get_openalex_work, search_openalex
from researchbot.agent.tools.metadata_merge import (
    create_standard_paper,
    merge_papers,
    normalize_from_arxiv,
    normalize_from_crossref,
    normalize_from_openalex,
)
from researchbot.config.schema import SemanticSearchConfig
from researchbot.knowledge_graph import KnowledgeGraph
from researchbot.search_index import SearchIndex
from researchbot.utils.helpers import safe_filename

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


def _make_safe_paper_id(paper: dict[str, Any]) -> str:
    """Generate a stable, filesystem-safe paper_id for saving.

    Priority:
    1. Existing paper_id (if safe)
    2. arXiv ID
    3. OpenAlex ID
    4. DOI (safe version)
    5. title (safe version)

    Returns a filesystem-safe string with no /, \\, :, or control chars.
    """
    # Priority 1: existing paper_id
    existing_id = paper.get("paper_id", "")
    if existing_id and existing_id == safe_filename(existing_id):
        return existing_id

    # Priority 2: arXiv ID
    arxiv_id = (paper.get("external_ids") or {}).get("arxiv", "")
    if arxiv_id and arxiv_id == safe_filename(arxiv_id):
        return arxiv_id

    # Priority 3: OpenAlex ID
    openalex_id = (paper.get("external_ids") or {}).get("openalex", "")
    if openalex_id and openalex_id == safe_filename(openalex_id):
        return openalex_id

    # Priority 4: DOI (safe version)
    doi = (paper.get("external_ids") or {}).get("doi", "")
    if doi:
        safe_doi = _safe_filename(doi)
        if safe_doi:
            return safe_doi

    # Priority 5: title (safe version) - use first meaningful words
    title = paper.get("title", "")
    if title:
        safe_title = _safe_filename(title)
        if safe_title and len(safe_title) >= 3:
            # Truncate to 100 chars to avoid overly long filenames
            return safe_title[:100]

    # Fallback: use a timestamp-based ID
    return f"paper_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _safe_filename(s: str) -> str:
    """Convert a string to a filesystem-safe filename."""
    if not s:
        return ""
    # Replace unsafe characters with underscore
    s = re.sub(r'[/\\:*?"<>|]', '_', s)
    # Replace whitespace with single underscore
    s = re.sub(r'\s+', '_', s)
    # Remove any remaining non-printable characters
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    # Strip leading/trailing underscores
    s = s.strip('_')
    return s


def _paper_to_enriched_dict(paper: dict[str, Any]) -> dict[str, Any]:
    """Convert paper (possibly from arXiv) to standardized enriched dict."""
    # Check if it's already in normalized format (external_ids must be a non-empty dict)
    if paper.get("external_ids"):
        return paper

    # Convert from arXiv-style paper_save format
    normalized = create_standard_paper()
    normalized["paper_id"] = paper.get("paper_id", "")
    normalized["title"] = paper.get("title", "")
    normalized["authors"] = paper.get("authors", [])
    normalized["year"] = paper.get("year", "")
    normalized["abstract"] = paper.get("abstract", "")
    normalized["url"] = paper.get("url", "")
    normalized["pdf_url"] = paper.get("pdf_url", "")

    # Handle topic/tags
    if "tags" in paper:
        normalized["topic_tags"] = paper["tags"]
    if "topic" in paper:
        normalized["topic_tags"].append(paper["topic"])

    # arXiv papers
    arxiv_id = paper.get("paper_id", "")
    if arxiv_id and not normalized["external_ids"]["arxiv"]:
        normalized["external_ids"]["arxiv"] = arxiv_id
        normalized["source"] = "arxiv"
        normalized["publication_type"] = "preprint"

    return normalized


async def enrich_paper_by_doi(
    doi: str,
    existing_paper: dict[str, Any] | None = None,
    crossref_mailto: str | None = None,
    openalex_api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> dict[str, Any]:
    """Enrich a paper by DOI.

    Tries Crossref first (better for DOI resolution), then OpenAlex.
    If existing_paper provided, merges metadata into it.
    """
    result = _paper_to_enriched_dict(existing_paper) if existing_paper else create_standard_paper()
    result["external_ids"]["doi"] = doi

    errors: list[str] = []

    # Try Crossref first
    crossref_paper = None
    try:
        crossref_work = await get_crossref_work(
            doi=doi,
            mailto=crossref_mailto,
            timeout=timeout,
            proxy=proxy,
        )
        if crossref_work:
            crossref_paper = normalize_from_crossref(crossref_work)
            result = merge_papers(result, crossref_paper)
    except Exception as e:
        errors.append(f"Crossref: {e}")

    # Also try OpenAlex (it has concepts/citations that Crossref lacks)
    try:
        openalex_work = await get_openalex_work(
            identifier=doi,
            api_key=openalex_api_key,
            timeout=timeout,
            proxy=proxy,
        )
        if openalex_work:
            openalex_paper = normalize_from_openalex(openalex_work)
            result = merge_papers(result, openalex_paper)
    except Exception as e:
        errors.append(f"OpenAlex: {e}")

    if not result.get("title"):
        raise ValueError(f"Could not enrich DOI {doi}. Errors: {', '.join(errors)}")

    return result


async def enrich_paper_by_title(
    title: str,
    existing_paper: dict[str, Any] | None = None,
    year: str | None = None,
    authors: list[str] | None = None,
    crossref_mailto: str | None = None,
    openalex_api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> dict[str, Any]:
    """Enrich a paper by title.

    Searches both Crossref and OpenAlex, merges results.
    """
    result = _paper_to_enriched_dict(existing_paper) if existing_paper else create_standard_paper()
    if title:
        result["title"] = title
    if year:
        result["year"] = year
    if authors:
        result["authors"] = authors

    year_int = int(year) if year and year.isdigit() else None
    author_str = authors[0] if authors else None

    all_candidates: list[dict[str, Any]] = []

    # Try Crossref
    try:
        crossref_results = await search_crossref(
            query=title,
            max_results=5,
            year=year_int,
            author=author_str,
            mailto=crossref_mailto,
            timeout=timeout,
            proxy=proxy,
        )
        for work in crossref_results:
            all_candidates.append(normalize_from_crossref(work))
    except Exception:
        pass  # Graceful degradation

    # Try OpenAlex
    try:
        openalex_results = await search_openalex(
            query=title,
            max_results=5,
            year=year_int,
            author=author_str,
            api_key=openalex_api_key,
            timeout=timeout,
            proxy=proxy,
        )
        for work in openalex_results:
            all_candidates.append(normalize_from_openalex(work))
    except Exception:
        pass  # Graceful degradation

    # Merge all candidates into result
    for candidate in all_candidates:
        result = merge_papers(result, candidate)

    return result


async def enrich_arxiv_paper(
    paper_id: str,
    existing_paper: dict[str, Any] | None = None,
    crossref_mailto: str | None = None,
    openalex_api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> dict[str, Any]:
    """Enrich an arXiv paper by looking up its DOI in Crossref/OpenAlex.

    Workflow:
    1. If existing_paper has DOI, use that directly
    2. Otherwise search Crossref by title+author to find DOI
    3. Then enrich with both Crossref and OpenAlex
    """
    # Start with existing paper or create new
    if existing_paper:
        result = _paper_to_enriched_dict(existing_paper)
    else:
        result = create_standard_paper()
        result["paper_id"] = paper_id
        result["external_ids"]["arxiv"] = paper_id
        result["source"] = "arxiv"

    # Check if we already have a DOI
    doi = (result.get("external_ids") or {}).get("doi", "")
    title = result.get("title", "")
    authors = result.get("authors", [])
    year = result.get("year", "")

    # Validate existing DOI: if title/authors are available, verify the DOI resolves to a matching paper
    if doi and title:
        try:
            validated = await enrich_paper_by_doi(
                doi=doi,
                existing_paper=None,
                crossref_mailto=crossref_mailto,
                openalex_api_key=openalex_api_key,
                timeout=timeout,
                proxy=proxy,
            )
            if validated.get("title"):
                # Check title word overlap
                title_words = set(title.lower().split()) - {"the", "a", "an", "of", "and", "in", "for", "to", "by"}
                validated_words = set(validated["title"].lower().split()) - {"the", "a", "an", "of", "and", "in", "for", "to", "by"}
                overlap = title_words & validated_words
                # If fewer than 30% of title words match, the DOI is likely wrong
                if len(title_words) > 0 and len(overlap) / len(title_words) < 0.3:
                    doi = ""  # Discard wrong DOI, will re-search
        except Exception:
            doi = ""  # Discard invalid DOI on error

    if not doi and title:
        # Try to find DOI via Crossref search
        try:
            year_int = int(year) if year and year.isdigit() else None
            author_str = authors[0] if authors else None

            crossref_results = await search_crossref(
                query=title,
                max_results=3,
                year=year_int,
                author=author_str,
                mailto=crossref_mailto,
                timeout=timeout,
                proxy=proxy,
            )

            for work in crossref_results:
                if work.doi:
                    # Validate: if we have author info, check for overlap with Crossref result
                    if authors and work.authors:
                        existing_author_last = authors[0].split()[-1].lower()
                        matched = any(
                            a.split()[-1].lower() == existing_author_last
                            for a in work.authors
                        )
                        if not matched:
                            continue  # Skip this result, keep looking
                    result["external_ids"]["doi"] = work.doi
                    break
        except Exception:
            pass

    # Enrich via OpenAlex using arXiv DOI first (guaranteed to resolve)
    arxiv_id = (result.get("external_ids") or {}).get("arxiv", paper_id or "").replace("arXiv:", "")
    # Strip version suffix (e.g. "1706.03762v7" -> "1706.03762")
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    if arxiv_id:
        arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
        try:
            result = await enrich_paper_by_doi(
                doi=arxiv_doi,
                existing_paper=result,
                crossref_mailto=crossref_mailto,
                openalex_api_key=openalex_api_key,
                timeout=timeout,
                proxy=proxy,
            )
        except Exception:
            pass

    # If still no concepts/refs, fall back to Crossref DOI or title search
    if not result.get("concepts") and not result.get("referenced_works"):
        doi = (result.get("external_ids") or {}).get("doi", "")
        if doi:
            try:
                result = await enrich_paper_by_doi(
                    doi=doi,
                    existing_paper=result,
                    crossref_mailto=crossref_mailto,
                    openalex_api_key=openalex_api_key,
                    timeout=timeout,
                    proxy=proxy,
                )
            except Exception:
                pass

    if not result.get("concepts") and not result.get("referenced_works"):
        try:
            result = await enrich_paper_by_title(
                title=title,
                existing_paper=result,
                year=year,
                authors=authors,
                crossref_mailto=crossref_mailto,
                openalex_api_key=openalex_api_key,
                timeout=timeout,
                proxy=proxy,
            )
        except Exception:
            pass

    result["status"]["enriched"] = True
    result["provenance"]["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return result


class PaperEnrichTool(Tool):
    """Enrich paper metadata using Crossref and OpenAlex.

    Takes an existing paper (by ID, DOI, or object) and enriches it with:
    - DOI (if not present)
    - Journal/venue information
    - Citation count
    - Referenced works
    - Concepts and topics
    - Open access status
    """

    name = "paper_enrich"
    description = (
        "Enrich paper metadata using Crossref and OpenAlex. "
        "Looks up DOI, journal info, citation count, references, and more. "
        "Can work with arXiv IDs, DOIs, or paper objects."
    )

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "arXiv paper ID to enrich",
            },
            "doi": {
                "type": "string",
                "description": "DOI to look up and use for enrichment",
            },
            "title": {
                "type": "string",
                "description": "Paper title to search for",
            },
            "paper": {
                "type": "object",
                "description": "Paper object (standardized dict) to enrich",
            },
            "save": {
                "type": "boolean",
                "description": "Whether to save enriched metadata back to local literature storage",
                "default": False,
            },
            "workspace": {
                "type": "string",
                "description": "Workspace path for local literature storage",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        crossref_mailto: str | None = None,
        openalex_api_key: str | None = None,
        workspace: str | None = None,
        proxy: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self.crossref_mailto = crossref_mailto
        self.openalex_api_key = openalex_api_key
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

    def _load_local_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load paper from local storage."""
        if not self._workspace:
            return None
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_enriched(self, paper: dict[str, Any]) -> str:
        """Save enriched paper back to local storage.

        Writes the full standardized paper dict, merging with any existing record
        to preserve fields not explicitly tracked by enrichment.
        """
        if not self._workspace:
            return ""

        # Generate safe paper_id
        paper_id = _make_safe_paper_id(paper)
        if not paper_id:
            # Last resort fallback - should never happen
            paper_id = f"paper_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        literature_dir = self._resolve_path("literature")
        json_path = literature_dir / "papers" / f"{paper_id}.json"

        # Load existing record if it exists (for merging)
        existing_record: dict[str, Any] = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_record = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_record = {}

        # Build the full record to save
        # Start with existing record to preserve old fields
        record = dict(existing_record)

        # Update with all fields from the enriched paper
        record.update({
            # Core identification fields - always preserve from enriched if present
            "paper_id": paper.get("paper_id", "") or paper_id,
            "title": paper.get("title", "") or existing_record.get("title", ""),
            "authors": paper.get("authors", []) or existing_record.get("authors", []),
            "year": paper.get("year", "") or existing_record.get("year", ""),
            "abstract": paper.get("abstract", "") or existing_record.get("abstract", ""),
            # External IDs
            "doi": paper.get("external_ids", {}).get("doi", "") or existing_record.get("doi", ""),
            # Enrichment-specific fields
            "venue": paper.get("venue", "") or existing_record.get("venue", ""),
            "publication_type": paper.get("publication_type", "") or existing_record.get("publication_type", ""),
            "url": paper.get("url", "") or existing_record.get("url", ""),
            "pdf_url": paper.get("pdf_url", "") or existing_record.get("pdf_url", ""),
            "citation_count": paper.get("citation_count", 0) or existing_record.get("citation_count", 0),
            "referenced_works": paper.get("referenced_works", []) or existing_record.get("referenced_works", []),
            "concepts": paper.get("concepts", []) or existing_record.get("concepts", []),
            "keywords": paper.get("keywords", []) or existing_record.get("keywords", []),
            "topic_tags": paper.get("topic_tags", []) or existing_record.get("topic_tags", []),
            # Metadata
            "enriched_at": datetime.now(timezone.utc).isoformat(),
            "enriched": True,
        })

        # If we generated a new safe paper_id and it differs from existing, update references
        if paper_id != existing_record.get("paper_id", "") and existing_record.get("paper_id"):
            # Keep original paper_id for reference
            record["original_paper_id"] = existing_record.get("paper_id", "")

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return str(json_path)

    async def execute(
        self,
        paper_id: str | None = None,
        doi: str | None = None,
        title: str | None = None,
        paper: dict[str, Any] | None = None,
        save: bool = False,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Enrich paper metadata."""
        ws = Path(workspace) if workspace else self._workspace

        # Try to load existing paper if we have an ID
        existing = None
        if paper_id:
            if ws:
                existing = self._load_local_paper(paper_id)
            if not existing:
                existing = paper

        # Determine what to enrich
        if doi:
            result = await enrich_paper_by_doi(
                doi=doi,
                existing_paper=existing,
                crossref_mailto=self.crossref_mailto,
                openalex_api_key=self.openalex_api_key,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
        elif paper_id and existing and (existing.get("external_ids") or {}).get("arxiv"):
            result = await enrich_arxiv_paper(
                paper_id=paper_id,
                existing_paper=existing,
                crossref_mailto=self.crossref_mailto,
                openalex_api_key=self.openalex_api_key,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
        elif title or (existing and existing.get("title")):
            enrich_title = title or (existing.get("title") if existing else None)
            result = await enrich_paper_by_title(
                title=enrich_title,
                existing_paper=existing,
                year=str(existing.get("year")) if existing and existing.get("year") else None,
                authors=existing.get("authors") if existing else None,
                crossref_mailto=self.crossref_mailto,
                openalex_api_key=self.openalex_api_key,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )
        elif paper:
            # Just enrich the paper object directly
            result = _paper_to_enriched_dict(paper)
        else:
            return "Error: Must provide paper_id, doi, title, or paper object"

        # Save if requested
        saved_path = ""
        index_status = "skipped (not configured)"
        if save and ws:
            # Ensure result has a paper_id for consistent display
            if not result.get("paper_id"):
                result["paper_id"] = _make_safe_paper_id(result)
            saved_path = self._save_enriched(result)

            # Update search index (includes knowledge graph sync)
            search_index = self._get_search_index()
            if search_index is not None:
                try:
                    await search_index.initialize()
                    sync_result = await search_index.upsert_paper(result)
                    search_index.close()
                    index_status = f"{'ok' if sync_result['graph_sync_status'] == 'ok' else 'failed'}"
                    if sync_result["graph_sync_status"] == "failed":
                        index_status += f": {sync_result['graph_sync_error']}"
                except Exception as e:
                    index_status = f"failed: {e}"
                    logger.warning(f"Search index update failed for {result.get('paper_id')}: {e}")

        # Format output
        lines = [
            f"Enriched Paper",
            f"Paper ID: {result.get('paper_id', 'N/A')}",
            f"Title: {result.get('title', 'N/A')}",
            f"Source: {result.get('source', 'N/A')}",
            "",
            "External IDs:",
            f"  arXiv: {result.get('external_ids', {}).get('arxiv', 'N/A')}",
            f"  DOI: {result.get('external_ids', {}).get('doi', 'N/A')}",
            f"  OpenAlex: {result.get('external_ids', {}).get('openalex', 'N/A')}",
            "",
            f"Authors: {', '.join(result.get('authors', [])[:5])}" +
            (f" et al. ({len(result['authors'])} total)" if len(result.get('authors', [])) > 5 else ""),
            f"Year: {result.get('year', 'N/A')}",
            f"Venue: {result.get('venue', 'N/A')}",
            f"Publication Type: {result.get('publication_type', 'N/A')}",
            f"Citation Count: {result.get('citation_count', 0)}",
            "",
            f"Concepts: {', '.join(result.get('concepts', [])[:10]) if result.get('concepts') else 'None'}",
            f"Topic Tags: {', '.join(result.get('topic_tags', [])[:10]) if result.get('topic_tags') else 'None'}",
            "",
            f"Enriched: {result.get('status', {}).get('enriched', False)}",
        ]

        if saved_path:
            lines.append(f"\nSaved to: {saved_path}")
            lines.append(f"Search index sync: {index_status}")

        return "\n".join(lines)


class CrossrefSearchTool(Tool):
    """Search Crossref for academic papers."""

    name = "crossref_search"
    description = (
        "Search Crossref for academic papers by query, author, year, or DOI. "
        "Crossref is best for finding published journal articles and their metadata."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-100)",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "year": {
                "type": "integer",
                "description": "Filter by publication year",
            },
            "author": {
                "type": "string",
                "description": "Filter by author name",
            },
            "doi": {
                "type": "string",
                "description": "Filter by DOI",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        mailto: str | None = None,
        user_agent: str | None = None,
        proxy: str | None = None,
    ):
        self.mailto = mailto
        self.user_agent = user_agent
        self._proxy = proxy

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        year: int | None = None,
        author: str | None = None,
        doi: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Search Crossref for papers."""
        try:
            results = await search_crossref(
                query=query,
                max_results=max_results,
                year=year,
                author=author,
                doi=doi,
                mailto=self.mailto,
                user_agent=self.user_agent,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )

            if not results:
                return f"No results found for: {query}"

            lines = [f"Crossref search results for: {query}\n"]
            lines.append(f"Found {len(results)} result(s):\n")

            for i, work in enumerate(results, 1):
                lines.append(f"[{i}] Title: {work.title}")
                lines.append(f"    Authors: {', '.join(work.authors[:5])}" +
                    (f" et al. ({len(work.authors)} total)" if len(work.authors) > 5 else ""))
                lines.append(f"    Year: {work.year}")
                lines.append(f"    Journal: {work.journal or 'N/A'}")
                lines.append(f"    DOI: {work.doi}")
                lines.append(f"    Citations: {work.cited_by_count}")
                lines.append(f"    URL: {work.url}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"Crossref search error: {e}"


class OpenAlexSearchTool(Tool):
    """Search OpenAlex for academic papers."""

    name = "openalex_search"
    description = (
        "Search OpenAlex for academic papers by query, author, year, or DOI. "
        "OpenAlex provides citation counts, concepts, and open access information."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-100)",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "year": {
                "type": "integer",
                "description": "Filter by publication year",
            },
            "author": {
                "type": "string",
                "description": "Filter by author name",
            },
            "doi": {
                "type": "string",
                "description": "Filter by DOI",
            },
            "title": {
                "type": "string",
                "description": "Filter by title",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        proxy: str | None = None,
    ):
        self.api_key = api_key
        self._proxy = proxy

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        year: int | None = None,
        author: str | None = None,
        doi: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Search OpenAlex for papers."""
        try:
            results = await search_openalex(
                query=query,
                max_results=max_results,
                year=year,
                author=author,
                doi=doi,
                title=title,
                api_key=self.api_key,
                timeout=DEFAULT_TIMEOUT,
                proxy=self._proxy,
            )

            if not results:
                return f"No results found for: {query}"

            lines = [f"OpenAlex search results for: {query}\n"]
            lines.append(f"Found {len(results)} result(s):\n")

            for i, work in enumerate(results, 1):
                lines.append(f"[{i}] Title: {work.title}")
                lines.append(f"    Authors: {', '.join(work.authors[:5])}" +
                    (f" et al. ({len(work.authors)} total)" if len(work.authors) > 5 else ""))
                lines.append(f"    Year: {work.year}")
                lines.append(f"    Journal: {work.journal or 'N/A'}")
                lines.append(f"    DOI: {work.doi}")
                lines.append(f"    Citations: {work.cited_by_count}")
                lines.append(f"    OA Status: {work.oa_status or 'N/A'}")
                lines.append(f"    Concepts: {', '.join(work.concepts[:5]) if work.concepts else 'None'}")
                lines.append(f"    URL: {work.url}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"OpenAlex search error: {e}"
