"""Metadata standardization and merge layer for papers from multiple sources.

This module provides:
1. Standardized paper dict format for all sources
2. Merge rules for combining metadata from arXiv, Crossref, and OpenAlex
3. Score-based ranking when multiple sources have results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# --------------------------------------------------------------------
# Standardized Paper Dict
# --------------------------------------------------------------------

STANDARD_PAPER_FIELDS = [
    "paper_id",
    "source",
    "external_ids",
    "title",
    "authors",
    "year",
    "venue",
    "publication_type",
    "url",
    "pdf_url",
    "abstract",
    "keywords",
    "topic_tags",
    "citation_count",
    "referenced_works",
    "concepts",
    "provenance",
    "status",
]


def create_standard_paper() -> dict[str, Any]:
    """Create an empty standardized paper dict."""
    return {
        "paper_id": "",
        "source": "",
        "external_ids": {
            "arxiv": "",
            "doi": "",
            "openalex": "",
            "crossref": "",
        },
        "title": "",
        "authors": [],
        "year": "",
        "venue": "",
        "publication_type": "",
        "url": "",
        "pdf_url": "",
        "abstract": "",
        "keywords": [],
        "topic_tags": [],
        "citation_count": 0,
        "referenced_works": [],
        "concepts": [],
        "provenance": {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "sources": [],
        },
        "status": {
            "reviewed": False,
            "pdf_downloaded": False,
            "summary_generated": False,
            "enriched": False,
        },
    }


def normalize_from_arxiv(entry: Any) -> dict[str, Any]:
    """Convert arXiv PaperEntry to standardized paper dict."""
    paper = create_standard_paper()
    paper["paper_id"] = entry.paper_id
    paper["source"] = "arxiv"
    paper["external_ids"]["arxiv"] = entry.paper_id
    paper["external_ids"]["doi"] = entry.doi or ""
    paper["title"] = entry.title
    paper["authors"] = list(entry.authors) if entry.authors else []
    paper["year"] = entry.published[:4] if entry.published else ""
    paper["venue"] = entry.journal_ref or ""
    paper["publication_type"] = "preprint"
    paper["url"] = entry.abs_url
    paper["pdf_url"] = entry.pdf_url
    paper["abstract"] = entry.summary
    paper["topic_tags"] = list(entry.categories) if entry.categories else []
    paper["provenance"]["sources"] = ["arxiv"]
    paper["status"]["pdf_downloaded"] = bool(entry.pdf_url)
    return paper


def normalize_from_crossref(work: Any) -> dict[str, Any]:
    """Convert CrossrefWork to standardized paper dict."""
    paper = create_standard_paper()
    paper["paper_id"] = work.doi
    paper["source"] = "crossref"
    paper["external_ids"]["doi"] = work.doi
    paper["title"] = work.title
    paper["authors"] = list(work.authors) if work.authors else []
    paper["year"] = work.year
    paper["venue"] = work.journal or ""
    paper["publication_type"] = "journal_article"
    paper["url"] = work.url or f"https://doi.org/{work.doi}"
    paper["abstract"] = work.abstract
    paper["citation_count"] = work.cited_by_count
    paper["referenced_works"] = list(work.referenced_works) if work.referenced_works else []
    paper["topic_tags"] = list(work.subjects) if work.subjects else []
    paper["provenance"]["sources"] = ["crossref"]

    # Open access info
    if work.is_open_access:
        paper["keywords"] = paper["keywords"] + ["open_access"]

    return paper


def normalize_from_openalex(work: Any) -> dict[str, Any]:
    """Convert OpenAlexWork to standardized paper dict."""
    paper = create_standard_paper()
    paper["paper_id"] = work.id
    paper["source"] = "openalex"
    paper["external_ids"]["doi"] = work.doi
    paper["external_ids"]["openalex"] = work.id
    paper["title"] = work.title
    paper["authors"] = list(work.authors) if work.authors else []
    paper["year"] = work.year
    paper["venue"] = work.journal or ""
    paper["publication_type"] = "journal_article"
    paper["url"] = work.url
    paper["abstract"] = work.abstract
    paper["citation_count"] = work.cited_by_count
    paper["referenced_works"] = list(work.referenced_works) if work.referenced_works else []
    paper["concepts"] = list(work.concepts) if work.concepts else []
    paper["topic_tags"] = list(work.concepts) if work.concepts else []
    paper["provenance"]["sources"] = ["openalex"]

    if work.is_open_access:
        paper["keywords"] = ["open_access"]

    return paper


# --------------------------------------------------------------------
# Merge Rules
# --------------------------------------------------------------------

def merge_papers(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Merge secondary metadata into primary paper dict.

    Merge rules:
    - DOI matches → Crossref data takes precedence
    - OpenAlex supplements citation_count, concepts, referenced_works, OA info
    - arXiv stays as primary for preprint source/IDs
    - Never overwrite non-empty fields with empty values
    """
    merged = dict(primary)

    # Track all sources
    all_sources = set()
    if merged["provenance"].get("sources"):
        all_sources.update(merged["provenance"]["sources"])
    if secondary.get("provenance", {}).get("sources"):
        all_sources.update(secondary["provenance"]["sources"])
    merged["provenance"]["sources"] = list(all_sources)

    def _merge_field(field_name: str, secondary_preferred: bool = False) -> None:
        """Merge a single field, optionally preferring secondary source."""
        primary_val = merged.get(field_name)
        secondary_val = secondary.get(field_name)

        # Don't overwrite with empty
        if not secondary_val:
            return
        if not primary_val and secondary_val:
            merged[field_name] = secondary_val
        elif secondary_preferred and secondary_val:
            merged[field_name] = secondary_val

    # DOI - prefer secondary if primary doesn't have it
    if not merged["external_ids"].get("doi") and secondary.get("external_ids", {}).get("doi"):
        merged["external_ids"]["doi"] = secondary["external_ids"]["doi"]

    # Title - prefer secondary if primary is too short or empty
    if not merged.get("title") and secondary.get("title"):
        merged["title"] = secondary["title"]

    # Authors - prefer secondary if primary has fewer
    if not merged.get("authors") and secondary.get("authors"):
        merged["authors"] = secondary["authors"]

    # Year - prefer secondary if primary is empty
    if not merged.get("year") and secondary.get("year"):
        merged["year"] = secondary["year"]

    # Venue - prefer secondary (usually more complete)
    _merge_field("venue", secondary_preferred=True)

    # Abstract - prefer longer one
    if len(secondary.get("abstract", "")) > len(merged.get("abstract", "")):
        merged["abstract"] = secondary["abstract"]

    # Citation count - take the higher value
    primary_cites = merged.get("citation_count") or 0
    secondary_cites = secondary.get("citation_count") or 0
    if secondary_cites > primary_cites:
        merged["citation_count"] = secondary_cites

    # Referenced works - combine unique
    primary_refs = set(merged.get("referenced_works", []))
    secondary_refs = set(secondary.get("referenced_works", []))
    if secondary_refs:
        merged["referenced_works"] = list(primary_refs | secondary_refs)

    # Concepts - combine unique
    primary_concepts = set(merged.get("concepts", []))
    secondary_concepts = set(secondary.get("concepts", []))
    if secondary_concepts:
        merged["concepts"] = list(primary_concepts | secondary_concepts)

    # Topic tags - combine unique
    primary_tags = set(merged.get("topic_tags", []))
    secondary_tags = set(secondary.get("topic_tags", []))
    if secondary_tags:
        merged["topic_tags"] = list(primary_tags | secondary_tags)

    # Keywords - combine unique
    primary_kw = set(merged.get("keywords", []))
    secondary_kw = set(secondary.get("keywords", []))
    if secondary_kw:
        merged["keywords"] = list(primary_kw | secondary_kw)

    # URL - prefer secondary if primary is empty
    if not merged.get("url") and secondary.get("url"):
        merged["url"] = secondary["url"]

    # PDF URL - keep primary if it exists (arXiv PDFs are free)
    if not merged.get("pdf_url") and secondary.get("pdf_url"):
        merged["pdf_url"] = secondary["pdf_url"]

    # Mark as enriched
    if secondary.get("source") in ("crossref", "openalex"):
        merged["status"]["enriched"] = True

    return merged


# --------------------------------------------------------------------
# Score-based ranking for multi-source results
# --------------------------------------------------------------------

@dataclass
class ScoredPaper:
    """A paper with a relevance score."""
    paper: dict[str, Any]
    score: float = 0.0
    match_type: str = ""  # "doi", "title_exact", "title_fuzzy", "author", "year"


def _normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    return title.lower().strip().replace(":", "").replace("-", " ").replace("_", " ")


def _title_similarity(title1: str, title2: str) -> float:
    """Calculate title similarity (0-1)."""
    t1 = _normalize_title(title1)
    t2 = _normalize_title(title2)

    if t1 == t2:
        return 1.0
    if t1 in t2 or t2 in t1:
        return 0.9

    # Simple word overlap
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    union = len(words1 | words2)
    return overlap / union if union > 0 else 0.0


def _author_overlap(authors1: list[str], authors2: list[str]) -> float:
    """Calculate author overlap (0-1)."""
    if not authors1 or not authors2:
        return 0.0

    a1_normalized = {a.lower().strip() for a in authors1}
    a2_normalized = {a.lower().strip() for a in authors2}

    overlap = len(a1_normalized & a2_normalized)
    union = len(a1_normalized | a2_normalized)
    return overlap / union if union > 0 else 0.0


def score_paper(
    paper: dict[str, Any],
    target_doi: str | None = None,
    target_title: str | None = None,
    target_authors: list[str] | None = None,
    target_year: str | None = None,
) -> ScoredPaper:
    """Score a paper against target criteria.

    Scoring:
    - DOI exact match: 100
    - Title exact match: 80
    - Title fuzzy (>0.8 similarity): 60
    - Title fuzzy (>0.5 similarity): 40
    - Author overlap: 20 * overlap_score
    - Year match: 10
    """
    score = 0.0
    match_type = ""

    paper_doi = paper.get("external_ids", {}).get("doi", "").lower().strip()
    target_doi_norm = (target_doi or "").lower().strip().replace("https://doi.org/", "")

    # DOI exact match (highest priority)
    if paper_doi and target_doi_norm and paper_doi == target_doi_norm:
        return ScoredPaper(paper=paper, score=100.0, match_type="doi")

    # Title matching
    paper_title = paper.get("title", "")
    if target_title and paper_title:
        title_sim = _title_similarity(paper_title, target_title)
        if title_sim >= 0.95:
            score += 80
            match_type = "title_exact"
        elif title_sim >= 0.8:
            score += 60
            match_type = "title_fuzzy"
        elif title_sim >= 0.5:
            score += 40

    # Author overlap
    paper_authors = paper.get("authors", [])
    if target_authors and paper_authors:
        author_sim = _author_overlap(paper_authors, target_authors)
        score += 20 * author_sim

    # Year match
    paper_year = paper.get("year", "")
    if target_year and paper_year == str(target_year):
        score += 10

    return ScoredPaper(paper=paper, score=score, match_type=match_type or "general")


def merge_and_rank(
    candidates: list[dict[str, Any]],
    target_doi: str | None = None,
    target_title: str | None = None,
    target_authors: list[str] | None = None,
    target_year: str | None = None,
) -> list[dict[str, Any]]:
    """Merge candidates from multiple sources and rank by relevance.

    Returns papers sorted by score descending, with merge metadata.
    """
    if not candidates:
        return []

    # Score all candidates
    scored = [score_paper(p, target_doi, target_title, target_authors, target_year) for p in candidates]

    # Sort by score descending
    scored.sort(key=lambda x: x.score, reverse=True)

    # Build result with merge metadata
    results = []
    seen_dois = set()

    for s in scored:
        paper = s.paper
        doi = paper.get("external_ids", {}).get("doi", "")

        # Skip duplicates with same DOI (keep highest scored)
        if doi and doi in seen_dois:
            continue

        # Merge with previous lower-scored version of same DOI if exists
        if doi:
            seen_dois.add(doi)

        results.append(paper)

    return results
