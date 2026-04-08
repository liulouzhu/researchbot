"""OpenAlex API client for fetching paper metadata."""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any


OPENALEX_API_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT = 30.0


@dataclass
class OpenAlexWork:
    """Standardized paper metadata from OpenAlex."""

    id: str  # OpenAlex ID (e.g., "W1234567890")
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
    oa_status: str = ""
    referenced_works: list[str] = field(default_factory=list)
    related_works: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    topics: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _decode_abstract_inverted_index(inverted_index: Any | None) -> str:
    """Decode OpenAlex abstract_inverted_index back to a string.

    The real OpenAlex format is dict[str, list[int]], e.g.:
        {"word": [0, 3], "another": [1]}
    where each integer is a word position in the abstract.

    Some entries may use (position, count) tuples for backward compatibility.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    word_positions: list[tuple[int, str]] = []

    for word, positions in inverted_index.items():
        if not isinstance(positions, (list, tuple)):
            continue

        # Handle list[int] format (correct OpenAlex format)
        # Each item is just a position integer
        for item in positions:
            try:
                if isinstance(item, (int, float)):
                    pos = int(item)
                    word_positions.append((pos, word))
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    # Handle (position, count) tuple format for compatibility
                    pos = int(item[0])
                    word_positions.append((pos, word))
            except (ValueError, TypeError):
                # Fault tolerance: skip malformed entries rather than crashing
                continue

    # Sort by position
    word_positions.sort(key=lambda x: x[0])

    # Join words in order
    return " ".join(word for _, word in word_positions)


def _parse_authorhips(authorships: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Parse OpenAlex authorships to extract author names and institutions.

    Returns:
        Tuple of (author_names, institutions)
    """
    authors = []
    institutions = []
    for auth in authorships:
        author_info = auth.get("author", {})
        display_name = author_info.get("display_name", "")
        if display_name:
            authors.append(display_name)

        # Extract institutions
        for inst in auth.get("institutions", []):
            inst_name = inst.get("display_name", "")
            if inst_name and inst_name not in institutions:
                institutions.append(inst_name)

    return authors, institutions


def _parse_openalex_work(item: dict[str, Any]) -> OpenAlexWork:
    """Parse a single OpenAlex work entity into OpenAlexWork."""
    id_ = item.get("id", "").replace("https://openalex.org/", "")
    doi = item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else ""

    title = item.get("title", "") or ""

    authors, _ = _parse_authorhips(item.get("authorships", []))

    year = ""
    date = item.get("publication_date", item.get("created_date", ""))
    if date:
        year = date[:4]

    # Journal info
    primary_location = item.get("primary_location") or {}
    source_info = primary_location.get("source") or {}
    journal = source_info.get("display_name", "")
    if not journal:
        best_oa = item.get("best_oa_location") or {}
        source_info = best_oa.get("source") or {}
        journal = source_info.get("display_name", "")

    volume = source_info.get("volume", "") or ""
    issue = source_info.get("issue", "") or ""
    pages = source_info.get("page", "") or ""

    publisher = item.get("publisher", "") or ""

    # Abstract
    abstract_inverted_index = item.get("abstract_inverted_index")
    abstract = _decode_abstract_inverted_index(abstract_inverted_index)

    url = item.get("doi", "") or item.get("id", "")
    if url and not url.startswith("http"):
        url = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{id_}"

    cited_by_count = item.get("cited_by_count", 0) or 0

    is_oa = item.get("open_access", {}).get("is_oa", False)
    oa_status = item.get("open_access", {}).get("oa_status", "")

    referenced_works = [
        ref.replace("https://openalex.org/", "")
        for ref in item.get("referenced_works", [])
        if ref
    ]

    related_works = [
        rel.replace("https://openalex.org/", "")
        for rel in item.get("related_works", [])
        if rel
    ]

    # Concepts (topics/subjects)
    concepts = [
        c.get("display_name", "")
        for c in item.get("concepts", [])
        if c.get("display_name")
    ]

    topics = item.get("topics", [])

    return OpenAlexWork(
        id=id_,
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
        oa_status=oa_status,
        referenced_works=referenced_works,
        related_works=related_works,
        concepts=concepts,
        topics=topics,
        raw=item,
    )


async def search_openalex(
    query: str,
    max_results: int = 10,
    year: int | None = None,
    author: str | None = None,
    doi: str | None = None,
    title: str | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> list[OpenAlexWork]:
    """Search OpenAlex for papers.

    Args:
        query: Search query string
        max_results: Maximum number of results (1-100)
        year: Filter by publication year
        author: Filter by author name
        doi: Filter by DOI
        title: Filter by title (used as additional search filter)
        api_key: OpenAlex API key (required after 2026-02-13)
        timeout: Request timeout in seconds
        proxy: HTTP proxy URL

    Returns:
        List of OpenAlexWork objects
    """
    params: dict[str, Any] = {
        "per-page": min(max_results, 100),
    }

    # Build search query - use search= for full-text search
    if query:
        params["search"] = query

    # Build filter list
    filters: list[str] = []

    if year:
        filters.append(f"from_publication_date:{year}-01-01")
        filters.append(f"to_publication_date:{year}-12-31")

    if author:
        # Use OpenAlex structured filter for author name
        filters.append(f"authorships.author.display_name.search:{author}")

    if doi:
        filters.append(f"doi:{doi}")

    if title:
        # Use OpenAlex structured filter for title
        filters.append(f"title.search:{title}")

    if filters:
        params["filter"] = ",".join(filters)

    # mailto is for rate limiting - just a contact email, not api_key
    params["mailto"] = "researchbot@example.com"

    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{OPENALEX_API_BASE}/works"

    proxies = proxy or None

    async with httpx.AsyncClient(timeout=timeout, proxy=proxies) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    return [_parse_openalex_work(item) for item in results]


async def get_openalex_work(
    identifier: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> OpenAlexWork | None:
    """Get a single OpenAlex work by ID or DOI.

    Args:
        identifier: OpenAlex ID (e.g., "W1234567890") or DOI
        api_key: OpenAlex API key
        timeout: Request timeout in seconds
        proxy: HTTP proxy URL

    Returns:
        OpenAlexWork object or None if not found
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # DOI lookups via /works/{doi} use path parameter, no filter needed
    params: dict[str, str] = {"mailto": "researchbot@example.com"}

    proxies = proxy or None

    # Check if it's a DOI (starts with 10.)
    clean_id = identifier.replace("https://doi.org/", "").replace("https://openalex.org/", "")
    if not clean_id.startswith("W"):
        identifier = clean_id

    url = f"{OPENALEX_API_BASE}/works/{identifier}"

    async with httpx.AsyncClient(timeout=timeout, proxy=proxies) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return _parse_openalex_work(data)
        except httpx.HTTPStatusError:
            return None
