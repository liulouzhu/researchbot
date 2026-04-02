"""Crossref API client for fetching paper metadata."""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any


CROSSREF_API_BASE = "https://api.crossref.org"
DEFAULT_TIMEOUT = 30.0
DEFAULT_USER_AGENT = "researchbot/1.0 (mailto:researchbot@example.com)"


@dataclass
class CrossrefWork:
    """Standardized paper metadata from Crossref."""

    doi: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    abstract: str = ""
    url: str = ""
    cited_by_count: int = 0
    is_open_access: bool = False
    license_url: str = ""
    referenced_works: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_author(author_data: dict[str, Any]) -> str:
    """Parse author name from Crossref author data."""
    given = author_data.get("given", "")
    family = author_data.get("family", "")
    name = author_data.get("name", "")
    if name:
        return name
    if given and family:
        return f"{given} {family}"
    return family or given or "Unknown"


def _extract_year(crossref_data: dict[str, Any]) -> str:
    """Extract year from Crossref date data."""
    date_parts = crossref_data.get("published-print", crossref_data.get("published-online", {}))
    if not date_parts:
        date_parts = crossref_data.get("created", {})
    if date_parts:
        date_parts_list = date_parts.get("date-parts", [[]])
        if date_parts_list and date_parts_list[0]:
            year = date_parts_list[0][0]
            if year:
                return str(year)
    return ""


def _extract_abstract(crossref_data: dict[str, Any]) -> str:
    """Extract abstract from Crossref data."""
    abstract = crossref_data.get("abstract", "")
    if abstract:
        # Crossref sometimes wraps abstract in <jats:p> tags
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract)
    return abstract


def _parse_crossref_work(item: dict[str, Any]) -> CrossrefWork:
    """Parse a single Crossref work/item into CrossrefWork."""
    msg = item.get("message", item)

    doi = msg.get("DOI", "")
    title_list = msg.get("title", [])
    title = title_list[0] if title_list else ""

    authors = [_parse_author(a) for a in msg.get("author", [])]

    year = _extract_year(msg)

    container_title = msg.get("container-title", [])
    journal = container_title[0] if container_title else ""

    volume = msg.get("volume", "")
    issue = msg.get("issue", "")
    pages = msg.get("page", "")
    publisher = msg.get("publisher", "")

    abstract = _extract_abstract(msg)

    url = msg.get("URL", f"https://doi.org/{doi}" if doi else "")

    cited_by_count = msg.get("is-referenced-by-count", 0) or 0

    is_oa = bool(msg.get("license", []))
    license_url = ""
    if msg.get("license"):
        license_url = msg["license"][0].get("URL", "")

    referenced_works = msg.get("reference", [])
    referenced_works = [ref.get("DOI", "") for ref in referenced_works if ref.get("DOI")]

    subjects = msg.get("subject", [])

    return CrossrefWork(
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        volume=volume,
        issue=issue,
        pages=pages,
        publisher=publisher,
        abstract=abstract,
        url=url,
        cited_by_count=cited_by_count,
        is_open_access=is_oa,
        license_url=license_url,
        referenced_works=referenced_works,
        subjects=subjects,
        raw=msg,
    )


def _build_headers(mailto: str | None = None, user_agent: str | None = None) -> dict[str, str]:
    """Build headers for Crossref API requests."""
    ua = user_agent or DEFAULT_USER_AGENT
    if mailto and "mailto" not in ua:
        ua = f"{ua} (mailto:{mailto})"
    return {
        "User-Agent": ua,
        "Accept": "application/json",
    }


async def search_crossref(
    query: str,
    max_results: int = 10,
    year: int | None = None,
    author: str | None = None,
    doi: str | None = None,
    mailto: str | None = None,
    user_agent: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> list[CrossrefWork]:
    """Search Crossref for papers.

    Args:
        query: Search query string
        max_results: Maximum number of results (1-100)
        year: Filter by publication year
        author: Filter by author name
        doi: Filter by DOI
        mailto: Email for Crossref polite pool
        user_agent: Custom User-Agent string
        timeout: Request timeout in seconds
        proxy: HTTP proxy URL

    Returns:
        List of CrossrefWork objects
    """
    params: dict[str, Any] = {
        "query": query,
        "rows": min(max_results, 100),
        "select": "DOI,title,author,published-print,published-online,created,container-title,volume,issue,page,publisher,abstract,URL,is-referenced-by-count,license,reference,subject",
    }

    if year:
        params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
    if author:
        params["query.author"] = author
    if doi:
        params["query.container-title"] = doi

    headers = _build_headers(mailto, user_agent)

    url = f"{CROSSREF_API_BASE}/works"

    proxies = proxy or None

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params, headers=headers, proxies=proxies)
        response.raise_for_status()
        data = response.json()

    items = data.get("message", {}).get("items", [])
    return [_parse_crossref_work(item) for item in items]


async def get_crossref_work(
    doi: str,
    mailto: str | None = None,
    user_agent: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> CrossrefWork | None:
    """Get a single Crossref work by DOI.

    Args:
        doi: The DOI to look up
        mailto: Email for Crossref polite pool
        user_agent: Custom User-Agent string
        timeout: Request timeout in seconds
        proxy: HTTP proxy URL

    Returns:
        CrossrefWork object or None if not found
    """
    headers = _build_headers(mailto, user_agent)
    proxies = proxy or None

    url = f"{CROSSREF_API_BASE}/works/{doi}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, headers=headers, proxies=proxies)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return _parse_crossref_work(data.get("message", {}))
        except httpx.HTTPStatusError:
            return None
