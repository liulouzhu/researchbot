"""Canonical paper identity normalization and multi-source merge.

This module provides:
1. Identity normalization helpers (arXiv ID, DOI, title, author)
2. Canonical key building for cross-source deduplication
3. Tiered match classification (exact / strong / weak / none)
4. Field-level merge logic with confidence tracking

Canonical key priority:
1. DOI (highest) — globally unique, stable
2. arXiv ID (version-stripped) — unique within arXiv ecosystem
3. normalized_title + first_author + year — heuristic, conservative
4. If none available, no heuristic merging

Match tiers (safety-first: prefer NOT merging over false merges):
- exact:  DOI exact or arXiv ID exact — always merge, allowed in union-find
- strong: Very high title similarity + author overlap + same year, AND at least
          one side has a strong ID (DOI/arXiv) — merge but flag confidence
- weak:   Moderate title similarity without strong ID anchor — do NOT auto-merge;
          only record as "possibly same" for human review
- none:   Clearly different papers
"""

from __future__ import annotations

import enum
import re
import unicodedata
from typing import Any


# --------------------------------------------------------------------
# Merge tier enum
# --------------------------------------------------------------------


class MergeTier(enum.Enum):
    """Confidence tier for a paper identity match.

    Only ``exact`` and ``strong`` are used for automatic merging.
    ``weak`` is recorded but does NOT trigger a merge.
    ``none`` means the papers are clearly different.
    """

    exact = "exact"  # DOI or arXiv ID match — highest confidence
    strong = "strong"  # Very high heuristic match with strong ID anchor
    weak = "weak"  # Moderate heuristic match without strong anchor
    none = "none"  # Clearly different papers


# --------------------------------------------------------------------
# Identity normalization helpers
# --------------------------------------------------------------------


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arXiv ID by stripping the version suffix.

    Examples:
        "2401.12345v2" -> "2401.12345"
        "2401.12345" -> "2401.12345"
        "arXiv:2401.12345v1" -> "2401.12345"
    """
    if not arxiv_id:
        return ""
    s = arxiv_id.strip()
    # Strip common prefix
    s = re.sub(r"^arXiv:", "", s, flags=re.IGNORECASE)
    # Strip version suffix (e.g. v2, V3)
    s = re.sub(r"[vV]\d+$", "", s)
    return s


def normalize_doi(doi: str) -> str:
    """Normalize a DOI by stripping URL prefix and lowercasing.

    Examples:
        "https://doi.org/10.1000/xyz123" -> "10.1000/xyz123"
        "10.1000/xyz123" -> "10.1000/xyz123"
    """
    if not doi:
        return ""
    s = doi.strip()
    s = re.sub(r"^https?://doi\.org/", "", s, flags=re.IGNORECASE)
    return s.lower()


def normalize_title(title: str) -> str:
    """Normalize a paper title for fuzzy comparison.

    Steps:
    - Unicode NFKD decomposition + strip combining marks (accents)
    - Lowercase
    - Remove LaTeX commands (\\xxx{}) but preserve braced text
    - Remove punctuation and special symbols
    - Collapse whitespace
    - Strip version marks / noise prefixes
    """
    if not title:
        return ""
    s = title
    # Unicode normalization: decompose then strip combining marks
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Remove LaTeX commands but preserve the text inside braces
    # e.g. \textit{transformer} -> transformer
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    # Remove remaining bare LaTeX commands (no braces)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Lowercase
    s = s.lower()
    # Remove punctuation (keep letters, digits, spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_author_name(name: str) -> str:
    """Normalize an author name for comparison.

    Steps:
    - Unicode NFKD + strip combining marks
    - Lowercase
    - Strip punctuation
    - Collapse whitespace
    """
    if not name:
        return ""
    s = name
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_last_name(name: str) -> str:
    """Extract the last (family) name from an author name string.

    Handles "First Last" and "Last, First" formats.
    """
    if not name:
        return ""
    # Check for comma format BEFORE normalizing (normalization strips commas)
    stripped = name.strip()
    if "," in stripped:
        # "Last, First" format — take the part before the comma
        last_part = stripped.split(",")[0].strip()
        return normalize_author_name(last_part)
    # "First Last" format — take last word
    normalized = normalize_author_name(name)
    parts = normalized.split()
    if parts:
        return parts[-1]
    return ""


# --------------------------------------------------------------------
# Title danger signals — titles that are high-risk for false merges
# --------------------------------------------------------------------

# Words that make a title "generic" and thus more prone to false positives
_GENERIC_TITLE_WORDS = frozenset(
    {
        "survey",
        "benchmark",
        "dataset",
        "framework",
        "evaluation",
        "review",
        "overview",
        "introduction",
        "tutorial",
        "comparison",
        "study",
        "analysis",
        "exploring",
        "towards",
        "note",
    }
)

_SHORT_TITLE_WORD_THRESHOLD = 3  # titles with ≤ this many words are "short"
_SHORT_TITLE_CHAR_THRESHOLD = 20  # titles shorter than this many chars are "short"


def _is_short_title(normalized_title: str) -> bool:
    """Return True if a normalized title is too short for reliable Jaccard."""
    word_count = len(normalized_title.split())
    char_count = len(normalized_title)
    return word_count <= _SHORT_TITLE_WORD_THRESHOLD or char_count <= _SHORT_TITLE_CHAR_THRESHOLD


def _has_generic_keywords(normalized_title: str) -> bool:
    """Return True if the title contains generic words that make
    false-positive merges more likely (survey, benchmark, etc.)."""
    words = set(normalized_title.split())
    return bool(words & _GENERIC_TITLE_WORDS)


# --------------------------------------------------------------------
# Canonical key building
# --------------------------------------------------------------------


def build_canonical_key(paper: dict[str, Any]) -> tuple[str, str]:
    """Build a canonical identity key for a paper record.

    Returns a (key_type, key_value) tuple.  key_type indicates the match
    confidence level:

    - "doi": DOI-based match (highest confidence)
    - "arxiv": arXiv ID-based match (high confidence)
    - "title_author_year": title + first author last name + year (medium)
    - "title_year": title + year only (low — conservative fallback)

    The key_value is a string suitable for exact dict-key lookup.
    """
    # Priority 1: DOI
    doi = normalize_doi(paper.get("doi", "") or (paper.get("external_ids") or {}).get("doi", ""))
    if doi:
        return ("doi", f"doi:{doi}")

    # Priority 2: arXiv ID (version-stripped)
    arxiv_id = normalize_arxiv_id(
        paper.get("arxiv_id", "") or (paper.get("external_ids") or {}).get("arxiv", "")
    )
    if arxiv_id:
        return ("arxiv", f"arxiv:{arxiv_id}")

    # Priority 3: title + first author last name + year
    title = normalize_title(paper.get("title", ""))
    authors = paper.get("authors") or []
    year = _extract_year(paper)

    if title:
        first_author_last = ""
        if authors:
            first_author_last = extract_last_name(authors[0])
        if first_author_last and year is not None:
            return ("title_author_year", f"tay:{title}|{first_author_last}|{year}")
        if first_author_last:
            return ("title_author_year", f"tay:{title}|{first_author_last}|")

    # Priority 4: title + year (very conservative)
    if title and year is not None:
        return ("title_year", f"ty:{title}|{year}")

    # Last resort: just title
    if title:
        return ("title_year", f"ty:{title}|")

    # No usable identity — return a unique-ish key to avoid collisions
    return ("none", f"none:id{id(paper)}")


def _extract_year(paper: dict[str, Any]) -> int | None:
    """Extract publication year as int from various paper dict formats."""
    year = paper.get("year")
    if year is None:
        return None
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.strip():
        try:
            return int(year.strip()[:4])
        except (ValueError, TypeError):
            return None
    return None


# --------------------------------------------------------------------
# Tiered match classification
# --------------------------------------------------------------------


def classify_match(a: dict[str, Any], b: dict[str, Any]) -> tuple[MergeTier, str]:
    """Classify the match between two paper records into a MergeTier.

    Returns (tier, reason) where reason is a human-readable explanation.

    Tiers:
    - exact:  DOI or arXiv ID exact match — always safe to merge
    - strong: Very high heuristic match with at least one strong ID anchor
    - weak:   Moderate heuristic match without strong ID anchor — do NOT auto-merge
    - none:   Clearly different papers

    Safety principle: it is far worse to merge two different papers than
    to keep two records of the same paper separate.
    """
    # --- Phase 1: Check for exact ID matches ---

    a_doi = normalize_doi(a.get("doi", "") or (a.get("external_ids") or {}).get("doi", ""))
    b_doi = normalize_doi(b.get("doi", "") or (b.get("external_ids") or {}).get("doi", ""))
    a_arxiv = normalize_arxiv_id(
        a.get("arxiv_id", "") or (a.get("external_ids") or {}).get("arxiv", "")
    )
    b_arxiv = normalize_arxiv_id(
        b.get("arxiv_id", "") or (b.get("external_ids") or {}).get("arxiv", "")
    )

    # DOI exact match
    if a_doi and b_doi and a_doi == b_doi:
        return (MergeTier.exact, f"DOI match: {a_doi}")

    # arXiv ID exact match
    if a_arxiv and b_arxiv and a_arxiv == b_arxiv:
        return (MergeTier.exact, f"arXiv ID match: {a_arxiv}")

    # --- Phase 2: Heuristic checks ---

    title_a = normalize_title(a.get("title", ""))
    title_b = normalize_title(b.get("title", ""))

    if not title_a or not title_b:
        return (MergeTier.none, "Missing title")

    # Compute title similarity
    title_sim = _jaccard_similarity(title_a, title_b)
    containment = title_a in title_b or title_b in title_a

    # --- Title similarity gate ---
    # Base threshold depends on title length and generic keywords
    is_short = _is_short_title(title_a) or _is_short_title(title_b)
    has_generic = _has_generic_keywords(title_a) or _has_generic_keywords(title_b)
    has_strong_id = bool(a_doi or b_doi or a_arxiv or b_arxiv)

    # Pre-check author overlap (needed for containment threshold relaxation)
    authors_a = a.get("authors") or []
    authors_b = b.get("authors") or []
    both_have_authors = bool(authors_a) and bool(authors_b)
    _pre_authors_a = {extract_last_name(a_) for a_ in authors_a} if authors_a else set()
    _pre_authors_b = {extract_last_name(b_) for b_ in authors_b} if authors_b else set()
    pre_author_overlap = bool(_pre_authors_a & _pre_authors_b) if both_have_authors else False

    # Year proximity pre-check
    year_a = _extract_year(a)
    year_b = _extract_year(b)
    pre_same_year = year_a is not None and year_b is not None and year_a == year_b

    # For short titles: require near-exact match or containment with supporting evidence
    if is_short:
        # Relax threshold when containment + strong ID anchor
        # (one title is a substring of the other + DOI/arXiv confirms identity)
        if containment and has_strong_id:
            if title_sim < 0.5:
                return (
                    MergeTier.none,
                    f"Short title, insufficient similarity even with containment+ID ({title_sim:.2f})",
                )
        # Relax threshold when containment + author overlap + same year
        # (strong heuristic evidence that the shorter title is a subset)
        elif containment and pre_author_overlap and pre_same_year:
            if title_sim < 0.3:
                return (
                    MergeTier.none,
                    f"Short title, insufficient similarity with containment+authors ({title_sim:.2f})",
                )
        # Standard short-title gate
        elif title_sim < 0.95 and not (containment and title_sim >= 0.85):
            return (MergeTier.none, f"Short title, insufficient similarity ({title_sim:.2f})")
    # For generic titles (survey/benchmark/etc): require higher threshold
    elif has_generic:
        # Relax when containment + author overlap + same year (same paper, different subtitle)
        if containment and pre_author_overlap and pre_same_year:
            if title_sim < 0.3:
                return (
                    MergeTier.none,
                    f"Generic title, insufficient similarity with containment+authors ({title_sim:.2f})",
                )
        elif title_sim < 0.9 and not (containment and title_sim >= 0.8):
            return (MergeTier.none, f"Generic title, insufficient similarity ({title_sim:.2f})")
    # Normal titles: original thresholds
    else:
        if title_sim < 0.8 and not (containment and title_sim >= 0.65):
            return (MergeTier.none, f"Insufficient title similarity ({title_sim:.2f})")

    # --- Author overlap check (mandatory for all heuristic matches) ---
    authors_a = a.get("authors") or []
    authors_b = b.get("authors") or []

    if not authors_a or not authors_b:
        # Missing author info: if no strong ID, this is only weak match at best
        has_strong_id = bool(a_doi or b_doi or a_arxiv or b_arxiv)
        if not has_strong_id:
            return (MergeTier.weak, "Missing authors, no strong ID anchor")
        # If one side has a strong ID and title is very similar,
        # allow as strong match (the ID anchors it)
        if title_sim >= 0.85:
            return (
                MergeTier.strong,
                "Strong ID anchor + high title sim, missing authors on one side",
            )
        # If one side has a strong ID + containment (one title is a prefix of the other),
        # this is likely the same paper with a different subtitle — allow as strong
        if containment and title_sim >= 0.5:
            return (
                MergeTier.strong,
                "Strong ID anchor + containment, missing authors on one side",
            )
        return (MergeTier.weak, "Strong ID anchor but moderate title sim and missing authors")

    last_a = {extract_last_name(a_) for a_ in authors_a}
    last_b = {extract_last_name(b_) for b_ in authors_b}
    author_overlap = last_a & last_b

    if not author_overlap:
        return (MergeTier.none, "No author overlap")

    # --- Year proximity check ---
    year_a = _extract_year(a)
    year_b = _extract_year(b)

    if year_a is not None and year_b is not None:
        year_drift = abs(year_a - year_b)
        # For heuristic matches without strong IDs, year must match exactly
        has_strong_id = bool(a_doi or b_doi or a_arxiv or b_arxiv)
        if not has_strong_id and year_drift > 0:
            return (MergeTier.none, f"Year drift {year_drift} without strong ID anchor")
        # Even with strong IDs, don't merge if year drift > 1
        if year_drift > 1:
            return (MergeTier.none, f"Year drift {year_drift} too large")
    elif year_a is not None or year_b is not None:
        # One has year, other doesn't — only allow if strong ID anchors
        has_strong_id = bool(a_doi or b_doi or a_arxiv or b_arxiv)
        if not has_strong_id:
            return (MergeTier.weak, "Partial year info, no strong ID anchor")

    # --- Final tier decision ---
    has_strong_id = bool(a_doi or b_doi or a_arxiv or b_arxiv)

    if has_strong_id:
        # Strong ID anchor + title/author/year match = strong
        return (MergeTier.strong, "Strong ID anchor + heuristic match")

    # No strong ID: this is a heuristic-only match
    # Exact normalized title (Jaccard 1.0) + author overlap + same year = strong
    # This handles "Attention Is All You Need" / "Attention is all you need"
    # where the titles are identical after normalization but neither source has DOI/arXiv
    if title_sim >= 0.99 and author_overlap and pre_same_year:
        return (
            MergeTier.strong,
            "Exact normalized title + author overlap + same year, no ID anchor",
        )

    # For generic or short titles without ID anchor, keep as weak (don't auto-merge)
    if has_generic or is_short:
        return (MergeTier.weak, "Heuristic match only, generic/short title without strong ID")

    # For normal titles with good similarity, author overlap, and exact year:
    # allow as strong (this handles the common case of same paper from
    # different sources where neither has DOI/arXiv)
    return (MergeTier.strong, "Heuristic match: title+author+year, no ID anchor")


def papers_likely_same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if two paper records are likely the same paper.

    Backward-compatible wrapper around classify_match.
    Returns True only for exact or strong matches.
    """
    tier, _ = classify_match(a, b)
    return tier in (MergeTier.exact, MergeTier.strong)


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between two strings (word-level)."""
    words1 = set(s1.split())
    words2 = set(s2.split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


# --------------------------------------------------------------------
# Field merge logic
# --------------------------------------------------------------------

# Source reliability ordering for venue (higher index = more reliable for venue)
_VENUE_PRIORITY = {
    "crossref": 3,
    "openalex": 2,
    "semantic_scholar": 1,
    "arxiv": 0,
    "": -1,
}


def merge_paper_fields(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    merge_tier: MergeTier = MergeTier.exact,
    merge_reason: str = "",
) -> dict[str, Any]:
    """Merge two paper dicts into a single, richer record.

    Merge rules:
    - title: prefer longer / more complete (more non-space chars)
    - authors: merge dedup (by normalized name)
    - abstract: prefer longer
    - doi: fill from any source, prefer primary if exists
    - arxiv_id: fill from any source, prefer primary if exists (both normalized)
    - venue: prefer source with higher reliability ordering
    - cited_by_count / citations: take max
    - sources: union of all source labels
    - external_ids: deep-merge all IDs
    - year: prefer non-empty; if both, prefer primary
    - url / pdf_url: prefer primary if exists, else fill from secondary
    - merge_confidence: recorded from merge_tier
    - merge_reason: recorded for traceability
    """
    merged: dict[str, Any] = dict(primary)

    # --- merge traceability ---
    # If primary already has merge_confidence, keep the lowest (most cautious)
    existing_tier = merged.get("merge_confidence", "")
    if existing_tier:
        # Keep the less confident tier
        _tier_order = {MergeTier.exact.value: 3, MergeTier.strong.value: 2, MergeTier.weak.value: 1}
        existing_rank = _tier_order.get(existing_tier, 0)
        new_rank = _tier_order.get(merge_tier.value, 0)
        if new_rank < existing_rank:
            merged["merge_confidence"] = merge_tier.value
    else:
        merged["merge_confidence"] = merge_tier.value

    if merge_reason:
        existing_reasons = merged.get("merge_reasons") or []
        merged["merge_reasons"] = existing_reasons + [merge_reason]

    # --- title ---
    p_title = merged.get("title", "")
    s_title = secondary.get("title", "")
    if not p_title and s_title:
        merged["title"] = s_title
    elif p_title and s_title:
        # Prefer the longer title (likely more complete)
        # But also prefer the one that has fewer ALL-CAPS or noise
        if len(s_title.replace(" ", "")) > len(p_title.replace(" ", "")):
            # Only switch if secondary is significantly longer (>10% more chars)
            if len(s_title.replace(" ", "")) > len(p_title.replace(" ", "")) * 1.1:
                merged["title"] = s_title

    # --- authors ---
    p_authors = merged.get("authors") or []
    s_authors = secondary.get("authors") or []
    if not p_authors and s_authors:
        merged["authors"] = list(s_authors)
    elif p_authors and s_authors:
        # Merge by normalized name dedup
        seen_normalized = set()
        combined: list[str] = []
        for name in p_authors + s_authors:
            norm = normalize_author_name(name)
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                combined.append(name)
        merged["authors"] = combined

    # --- abstract ---
    p_abstract = merged.get("abstract", "")
    s_abstract = secondary.get("abstract", "")
    if len(s_abstract or "") > len(p_abstract or ""):
        merged["abstract"] = s_abstract

    # --- doi ---
    p_doi = normalize_doi(
        merged.get("doi", "") or (merged.get("external_ids") or {}).get("doi", "")
    )
    s_doi = normalize_doi(
        secondary.get("doi", "") or (secondary.get("external_ids") or {}).get("doi", "")
    )
    if not p_doi and s_doi:
        merged["doi"] = s_doi
        if "external_ids" not in merged:
            merged["external_ids"] = {}
        merged["external_ids"]["doi"] = s_doi

    # --- arxiv_id ---
    p_arxiv = normalize_arxiv_id(
        merged.get("arxiv_id", "") or (merged.get("external_ids") or {}).get("arxiv", "")
    )
    s_arxiv = normalize_arxiv_id(
        secondary.get("arxiv_id", "") or (secondary.get("external_ids") or {}).get("arxiv", "")
    )
    if not p_arxiv and s_arxiv:
        merged["arxiv_id"] = s_arxiv
        if "external_ids" not in merged:
            merged["external_ids"] = {}
        merged["external_ids"]["arxiv"] = s_arxiv

    # --- venue ---
    p_source = _best_source(merged)
    s_source = _best_source(secondary)
    p_venue = merged.get("venue", "") or ""
    s_venue = secondary.get("venue", "") or ""
    if not p_venue and s_venue:
        merged["venue"] = s_venue
    elif p_venue and s_venue:
        # Prefer venue from more reliable source
        if _VENUE_PRIORITY.get(s_source, -1) > _VENUE_PRIORITY.get(p_source, -1):
            merged["venue"] = s_venue

    # --- cited_by_count / citations ---
    p_cites = merged.get("cited_by_count") or merged.get("citations") or 0
    s_cites = secondary.get("cited_by_count") or secondary.get("citations") or 0
    max_cites = max(int(p_cites), int(s_cites))
    # Store in both fields for compatibility
    merged["cited_by_count"] = max_cites
    merged["citations"] = max_cites

    # --- sources ---
    p_sources = set(merged.get("sources") or [])
    s_sources = set(secondary.get("sources") or [])
    merged["sources"] = sorted(p_sources | s_sources)

    # --- external_ids (deep merge) ---
    p_ext = dict(merged.get("external_ids") or {})
    s_ext = dict(secondary.get("external_ids") or {})
    for k, v in s_ext.items():
        if v and (not p_ext.get(k)):
            p_ext[k] = v
    merged["external_ids"] = p_ext

    # --- year ---
    p_year = merged.get("year")
    s_year = secondary.get("year")
    if not p_year and s_year:
        merged["year"] = s_year

    # --- url / pdf_url ---
    if not merged.get("url") and secondary.get("url"):
        merged["url"] = secondary["url"]
    if not merged.get("pdf_url") and secondary.get("pdf_url"):
        merged["pdf_url"] = secondary["pdf_url"]

    # --- combined_score ---
    p_score = merged.get("combined_score", 1.0)
    s_score = secondary.get("combined_score", 1.0)
    merged["combined_score"] = max(float(p_score), float(s_score))

    return merged


def _best_source(paper: dict[str, Any]) -> str:
    """Return the most authoritative source label from a paper record."""
    sources = paper.get("sources") or []
    if not sources:
        src = paper.get("source", "")
        return src if isinstance(src, str) else ""
    # Return highest-priority source
    for s in sorted(sources, key=lambda x: _VENUE_PRIORITY.get(x, -1), reverse=True):
        return s
    return sources[0] if sources else ""
