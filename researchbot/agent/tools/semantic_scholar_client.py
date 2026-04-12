"""Semantic Scholar API client for fetching paper metadata."""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any


S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_FIELDS = "title,authors,year,venue,citationCount,externalIds,abstract"


@dataclass
class SemanticScholarWork:
    """Standardized paper metadata from Semantic Scholar."""

    paper_id: str  # Semantic Scholar ID
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_semantic_scholar_work(item: dict[str, Any]) -> SemanticScholarWork:
    """Parse a Semantic Scholar paper item into a SemanticScholarWork."""
    external_ids = item.get("externalIds", {})
    authors_data = item.get("authors", []) or []

    # Extract first 20 authors to avoid too long lists
    authors = [a.get("name", "") for a in authors_data[:20] if a.get("name")]

    # Parse year
    year = item.get("year")
    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None

    return SemanticScholarWork(
        paper_id=item.get("paperId", ""),
        title=item.get("title", "") or "",
        authors=authors,
        year=year,
        venue=item.get("venue", "") or "",
        citation_count=item.get("citationCount", 0) or 0,
        doi=external_ids.get("DOI", "") or "",
        arxiv_id=external_ids.get("ArXiv", "") or "",
        abstract=item.get("abstract", "") or "",
        raw=item,
    )


async def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    year: int | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> list[SemanticScholarWork]:
    """Search Semantic Scholar for papers.

    Args:
        query: Search query string
        max_results: Maximum number of results (1-100)
        year: Filter by publication year
        api_key: Semantic Scholar API key (optional)
        timeout: Request timeout in seconds
        proxy: HTTP proxy URL

    Returns:
        List of SemanticScholarWork objects
    """
    params: dict[str, Any] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": DEFAULT_FIELDS,
    }

    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_API_BASE}/paper/search"

    proxies = proxy or None

    async with httpx.AsyncClient(timeout=timeout, proxy=proxies) as client:
        response = await client.post(url, json=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    results = data.get("data", [])
    works = [_parse_semantic_scholar_work(item) for item in results]

    # Filter by year if specified (Semantic Scholar search API doesn't support year filter directly)
    if year:
        works = [w for w in works if w.year == year]

    return works
