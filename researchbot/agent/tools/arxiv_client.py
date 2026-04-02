"""Minimal arXiv API client for querying papers."""

from __future__ import annotations

import httpx
from dataclasses import dataclass
from typing import Any

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"
ARXIV_SCHEMAS_NS = "http://arxiv.org/schemas/atom"
DEFAULT_TIMEOUT = 15.0


@dataclass
class PaperEntry:
    """Normalized paper entry from arXiv."""
    paper_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    summary: str
    primary_category: str
    categories: list[str]
    doi: str
    journal_ref: str
    abs_url: str
    pdf_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "summary": self.summary,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "source": "arxiv",
        }


def _extract_text(element: Any, tag: str) -> str:
    """Extract text content from an XML element in the default Atom namespace."""
    child = element.find(f"{{{ARXIV_NS}}}{tag}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _extract_authors(entry: Any) -> list[str]:
    """Extract author names from entry."""
    authors = []
    for author in entry.findall(f"{{{ARXIV_NS}}}author"):
        name_el = author.find(f"{{{ARXIV_NS}}}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    return authors


def _extract_arxiv_element(entry: Any, local_name: str) -> str:
    """Extract text from an arxiv-namespaced element (e.g. doi, journal_ref)."""
    child = entry.find(f"{{{ARXIV_SCHEMAS_NS}}}{local_name}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _extract_primary_category(entry: Any) -> str:
    """Extract primary category from arxiv:primary_category element (preferred) or category with primary='true' (fallback)."""
    primary_cat = entry.find(f"{{{ARXIV_SCHEMAS_NS}}}primary_category")
    if primary_cat is not None:
        term = primary_cat.get("term", "")
        if term:
            return term
    for cat in entry.findall(f"{{{ARXIV_NS}}}category"):
        if cat.get("primary") == "true":
            term = cat.get("term", "")
            if term:
                return term
    return ""


def _extract_categories(entry: Any) -> list[str]:
    """Extract all categories from entry."""
    categories = []
    for cat in entry.findall(f"{{{ARXIV_NS}}}category"):
        term = cat.get("term", "")
        if term:
            categories.append(term)
    return categories


def _parse_entry(entry: Any) -> PaperEntry | None:
    """Parse a single arXiv entry element into a PaperEntry."""
    try:
        id_text = _extract_text(entry, "id")
        if not id_text:
            return None
        paper_id = id_text.rstrip("/").split("/")[-1]

        title = _extract_text(entry, "title")
        summary = _extract_text(entry, "summary")
        published = _extract_text(entry, "published")
        updated = _extract_text(entry, "updated")
        authors = _extract_authors(entry)
        primary_category = _extract_primary_category(entry)
        categories = _extract_categories(entry)

        doi = _extract_arxiv_element(entry, "doi")
        journal_ref = _extract_arxiv_element(entry, "journal_ref")

        abs_url = f"https://arxiv.org/abs/{paper_id}"
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        return PaperEntry(
            paper_id=paper_id,
            title=title,
            authors=authors,
            published=published,
            updated=updated,
            summary=summary,
            primary_category=primary_category,
            categories=categories,
            doi=doi,
            journal_ref=journal_ref,
            abs_url=abs_url,
            pdf_url=pdf_url,
        )
    except Exception:
        return None


def _parse_feed(xml_content: bytes) -> list[PaperEntry]:
    """Parse arXiv Atom feed and return list of PaperEntry objects."""
    import xml.etree.ElementTree as ET

    entries = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}")

    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        parsed = _parse_entry(entry)
        if parsed is not None:
            entries.append(parsed)

    return entries


async def search_arxiv(
    query: str,
    start: int = 0,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> list[PaperEntry]:
    """
    Search arXiv for papers matching the query.

    Args:
        query: arXiv search query (e.g. "ti:transformer AND au:hinton")
        start: Start index for pagination
        max_results: Number of results to return (1-20)
        sort_by: Sort by "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order: "ascending" or "descending"
        timeout: Request timeout in seconds
        proxy: Optional proxy URL

    Returns:
        List of PaperEntry objects

    Raises:
        httpx.HTTPError: On network errors
        ValueError: On XML parsing errors or invalid parameters
    """
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            response = await client.get(
                ARXIV_API_URL,
                params=params,
                timeout=timeout,
                headers={"Accept": "application/atom+xml"},
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise httpx.HTTPError(f"arXiv API request failed: {e}")

    if not response.text or response.text.strip() == "":
        return []

    try:
        return _parse_feed(response.content)
    except ValueError:
        raise


def _normalize_paper_id(paper_id_or_url: str) -> str:
    """Normalize paper ID from various formats to arXiv ID."""
    import re
    paper_id_or_url = paper_id_or_url.strip()

    url_patterns = [
        r"https?://arxiv\.org/abs/(\d+\.\d+[vV]\d+)",
        r"https?://arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/abs/(\d+\.\d+[vV]\d+)",
        r"(\d+\.\d+[vV]\d+)",
        r"(\d+\.\d+)",
    ]

    for pattern in url_patterns:
        match = re.search(pattern, paper_id_or_url)
        if match:
            return match.group(1)

    return paper_id_or_url


async def get_paper_by_id(
    paper_id: str | None = None,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> PaperEntry:
    """
    Get a single paper from arXiv by ID or URL.

    Args:
        paper_id: arXiv paper ID (e.g. "2401.12345" or "2401.12345v2")
        url: arXiv URL (e.g. "https://arxiv.org/abs/2401.12345")
        timeout: Request timeout in seconds
        proxy: Optional proxy URL

    Returns:
        PaperEntry object

    Raises:
        ValueError: If neither paper_id nor url is provided, or paper not found
        httpx.HTTPError: On network errors
    """
    if not paper_id and not url:
        raise ValueError("Either paper_id or url must be provided")

    normalized_id = _normalize_paper_id(paper_id or url or "")

    params = {
        "id_list": normalized_id,
    }

    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            response = await client.get(
                ARXIV_API_URL,
                params=params,
                timeout=timeout,
                headers={"Accept": "application/atom+xml"},
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise httpx.HTTPError(f"arXiv API request failed: {e}")

    if not response.text or response.text.strip() == "":
        raise ValueError(f"Paper not found: {normalized_id}")

    try:
        entries = _parse_feed(response.content)
    except ValueError:
        raise

    if not entries:
        raise ValueError(f"Paper not found: {normalized_id}")

    return entries[0]
