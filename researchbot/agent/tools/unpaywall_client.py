"""Unpaywall API client for resolving open-access PDF URLs from DOIs."""

from __future__ import annotations

import httpx
from dataclasses import dataclass
from typing import Any

UNPAYWALL_API_BASE = "https://api.unpaywall.org"
DEFAULT_TIMEOUT = 15.0


@dataclass
class UnpaywallResult:
    """Result from Unpaywall OA lookup."""

    doi: str
    is_oa: bool
    oa_url: str  # Best OA landing page
    pdf_url: str  # Direct PDF URL if available
    oa_status: str  # gold, green, hybrid, bronze, closed
    host_type: str  # publisher, repository
    version: str  # publishedVersion, acceptedVersion, submittedVersion
    raw: dict[str, Any]


def _extract_pdf_from_oa_locations(locations: list[dict[str, Any]]) -> str:
    """Find the best direct PDF URL from OA locations.

    Priority: pdf_url from any OA location, preferring 'publishedVersion'.
    """
    if not locations:
        return ""

    # First pass: look for publishedVersion with a pdf_url
    for loc in locations:
        if loc.get("pdf_url") and loc.get("version_for_pdf") == "publishedVersion":
            return loc["pdf_url"]

    # Second pass: any pdf_url
    for loc in locations:
        if loc.get("pdf_url"):
            return loc["pdf_url"]

    return ""


async def get_unpaywall_oa(
    doi: str,
    email: str,
    timeout: float = DEFAULT_TIMEOUT,
    proxy: str | None = None,
) -> UnpaywallResult | None:
    """Look up OA status for a DOI via Unpaywall.

    Args:
        doi: The DOI to look up.
        email: Required email for Unpaywall polite pool.
        timeout: Request timeout in seconds.
        proxy: Optional proxy URL.

    Returns:
        UnpaywallResult or None if lookup fails.
    """
    if not doi or not email:
        return None

    # Clean DOI
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    url = f"{UNPAYWALL_API_BASE}/v2/{doi}"
    params = {"email": email}

    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            response = await client.get(url, params=params, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return None

    is_oa = data.get("is_oa", False)
    oa_status = data.get("oa_status", "closed")

    # Best OA location
    best = data.get("best_oa_location") or {}
    oa_url = best.get("url_for_landing_page", "") or ""
    pdf_url = best.get("url_for_pdf", "") or ""
    host_type = best.get("host_type", "") or ""
    version = best.get("version", "") or ""

    # If best location has no direct PDF, try all OA locations
    if not pdf_url and is_oa:
        oa_locations = data.get("oa_locations") or data.get("oa_locations_from_best_to_worst") or []
        pdf_url = _extract_pdf_from_oa_locations(oa_locations)

    return UnpaywallResult(
        doi=doi,
        is_oa=is_oa,
        oa_url=oa_url,
        pdf_url=pdf_url,
        oa_status=oa_status,
        host_type=host_type,
        version=version,
        raw=data,
    )
