"""Unified full-text fetching for papers from any source.

Provides ensure_full_text() which tries multiple strategies in order:
  1. If local PDF / extracted text already exists → use it
  2. arXiv papers → download PDF from arXiv
  3. Papers with DOI → try Unpaywall OA PDF
  4. Papers with pdf_url → direct download
  5. Fallback → return None (caller uses abstract only)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Regex for extracting arXiv ID from paper_id or URL
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _resolve_path(workspace: Path, rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = workspace / p
    return p


def _load_paper_json(workspace: Path, paper_id: str) -> dict[str, Any] | None:
    json_path = workspace / "literature" / "papers" / f"{paper_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_paper_json(workspace: Path, paper_id: str, record: dict[str, Any]) -> None:
    json_path = workspace / "literature" / "papers" / f"{paper_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


async def _download_pdf(
    url: str,
    dest: Path,
    proxy: str | None = None,
    timeout: float = 60.0,
) -> bool:
    """Download a PDF from *url* to *dest*. Returns True on success."""
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            # Sanity check: response should look like a PDF
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and len(resp.content) < 10_000:
                # Likely an HTML landing page, not a PDF
                return False
            if len(resp.content) < 500:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as exc:
        logger.debug("PDF download failed from %s: %s", url, exc)
        return False


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — cannot extract text from PDF")
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                parts.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning("Text extraction failed for %s: %s", pdf_path, exc)
        return ""


def _is_arxiv_paper(paper: dict[str, Any]) -> bool:
    """Check whether a paper dict represents an arXiv paper."""
    paper_id = paper.get("paper_id", "")
    source = paper.get("source", "")
    if source == "arxiv":
        return True
    if _ARXIV_ID_RE.search(paper_id):
        return True
    url = paper.get("url", "") or paper.get("pdf_url", "")
    if "arxiv.org" in url:
        return True
    return False


def _extract_arxiv_id(paper: dict[str, Any]) -> str:
    """Extract arXiv numeric ID from a paper dict."""
    paper_id = paper.get("paper_id", "")
    m = _ARXIV_ID_RE.search(paper_id)
    if m:
        return m.group(1)
    url = paper.get("url", "") or paper.get("pdf_url", "")
    m = _ARXIV_ID_RE.search(url)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# OA PDF URL resolution strategies
# ---------------------------------------------------------------------------


async def _resolve_pdf_url_from_openalex(
    doi: str,
    proxy: str | None = None,
) -> str:
    """Try to get a direct PDF URL from OpenAlex best_oa_location."""
    if not doi:
        return ""
    try:
        from researchbot.agent.tools.openalex_client import get_openalex_work
        work = await get_openalex_work(doi, proxy=proxy)
        if work:
            # OpenAlex doesn't expose pdf_url directly in OpenAlexWork dataclass,
            # but we can check is_open_access and construct a DOI-based URL.
            # The real PDF URL comes from best_oa_location in the raw data.
            raw = work.raw or {}
            best_oa = raw.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf", "") or ""
            if pdf_url:
                return pdf_url
            # Fallback: try landing page
            landing = best_oa.get("url_for_landing_page", "") or ""
            if landing:
                return landing
    except Exception:
        pass
    return ""


async def _resolve_pdf_url_from_unpaywall(
    doi: str,
    unpaywall_email: str,
    proxy: str | None = None,
) -> str:
    """Try to get a direct PDF URL via Unpaywall."""
    if not doi or not unpaywall_email:
        return ""
    try:
        from researchbot.agent.tools.unpaywall_client import get_unpaywall_oa
        result = await get_unpaywall_oa(doi, email=unpaywall_email, proxy=proxy)
        if result and result.is_oa and result.pdf_url:
            return result.pdf_url
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_full_text(
    paper: dict[str, Any],
    workspace: Path | None = None,
    proxy: str | None = None,
    unpaywall_email: str = "",
) -> dict[str, Any]:
    """Ensure *paper* has downloadable PDF and extracted text available.

    This function mutates *paper* in-place and returns it.  The following
    fields are set / updated:

    - ``pdf_url``     – resolved direct PDF download URL (if found)
    - ``pdf_path``    – local filesystem path to downloaded PDF (if downloaded)
    - ``text_source`` – one of ``arxiv``, ``openalex_oa``, ``unpaywall``,
                        ``doi_pdf``, ``direct_url``, ``none``

    On failure at every strategy the function simply returns without setting
    ``pdf_path`` — callers should fall back to abstract.
    """
    paper_id = paper.get("paper_id", "")
    doi = paper.get("doi", "") or (paper.get("external_ids") or {}).get("doi", "")

    if not workspace:
        return paper

    pdfs_dir = workspace / "literature" / "pdfs"
    extracted_dir = workspace / "literature" / "extracted"

    # ---- 1. Already have extracted text? Done. ----------------------------
    extracted_path = extracted_dir / f"{paper_id}.txt"
    if extracted_path.exists() and extracted_path.stat().st_size > 0:
        paper.setdefault("pdf_path", str(pdfs_dir / f"{paper_id}.pdf"))
        paper["text_source"] = "local"
        return paper

    # ---- 2. Already have a local PDF? Extract text. -----------------------
    pdf_path = pdfs_dir / f"{paper_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        text = _extract_text_from_pdf(pdf_path)
        if text:
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            with open(extracted_path, "w", encoding="utf-8") as f:
                f.write(text)
            paper["pdf_path"] = str(pdf_path)
            paper["text_source"] = "local"
            _update_paper_json(workspace, paper_id, pdf_path, extracted_path)
            return paper

    # ---- 3. Try to download a PDF using various strategies ----------------
    pdf_url = ""
    text_source = "none"

    # Strategy A: arXiv direct download
    if _is_arxiv_paper(paper):
        arxiv_id = _extract_arxiv_id(paper)
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            text_source = "arxiv"

    # Strategy B: OpenAlex best_oa_location (if DOI available)
    if not pdf_url and doi:
        pdf_url = await _resolve_pdf_url_from_openalex(doi, proxy=proxy)
        if pdf_url:
            text_source = "openalex_oa"

    # Strategy C: Unpaywall (if DOI and email available)
    if not pdf_url and doi and unpaywall_email:
        pdf_url = await _resolve_pdf_url_from_unpaywall(doi, unpaywall_email, proxy=proxy)
        if pdf_url:
            text_source = "unpaywall"

    # Strategy D: Direct pdf_url already on the paper dict
    if not pdf_url:
        existing_pdf_url = paper.get("pdf_url", "")
        if existing_pdf_url and existing_pdf_url.startswith("http"):
            pdf_url = existing_pdf_url
            text_source = "direct_url"

    # ---- 4. Download the PDF if we found a URL ----------------------------
    if pdf_url:
        success = await _download_pdf(pdf_url, pdf_path, proxy=proxy)
        if success:
            text = _extract_text_from_pdf(pdf_path)
            if text:
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                with open(extracted_path, "w", encoding="utf-8") as f:
                    f.write(text)
            paper["pdf_path"] = str(pdf_path)
            paper["pdf_url"] = pdf_url
            paper["text_source"] = text_source
            _update_paper_json(workspace, paper_id, pdf_path, extracted_path)
            return paper

    # ---- 5. Nothing worked ------------------------------------------------
    paper["text_source"] = "none"
    return paper


def _update_paper_json(
    workspace: Path,
    paper_id: str,
    pdf_path: Path,
    extracted_path: Path,
) -> None:
    """Update the paper JSON record with pdf_path and extracted_text_path."""
    record = _load_paper_json(workspace, paper_id)
    if record is None:
        return
    record["pdf_path"] = str(pdf_path)
    record["extracted_text_path"] = str(extracted_path)
    record.setdefault("status", {})
    record["status"]["pdf_downloaded"] = True
    record["status"]["full_text_extracted"] = True
    _save_paper_json(workspace, paper_id, record)
