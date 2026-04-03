"""Citation data model, citekey generation, and format exporters.

This module provides:
1. Unified CitationEntry data model from heterogeneous paper dicts
2. Deterministic citekey generation
3. Export renderers for BibTeX, RIS, CSL-JSON, APA, MLA, GB/T 7714
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# ────────────────────────────────────────────────────────────────────
# Unified Citation Entry
# ────────────────────────────────────────────────────────────────────

@dataclass
class CitationEntry:
    """Unified citation record normalized from heterogeneous paper dicts."""

    title: str = ""
    authors: list[str] = field(default_factory=list)  # each "Family, Given" or "Name"
    year: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    crossref_id: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    language: str = ""
    publication_type: str = ""  # journal_article, conference_paper, preprint, etc.
    citekey: str = ""

    # ---------- factory helpers ----------

    @classmethod
    def from_paper(cls, paper: dict[str, Any]) -> CitationEntry:
        """Create CitationEntry from a standard paper dict."""
        ext = paper.get("external_ids", {}) or {}

        raw_authors = paper.get("authors", []) or []
        authors: list[str] = []
        for a in raw_authors:
            if isinstance(a, str):
                authors.append(a.strip())
            elif isinstance(a, dict):
                name = a.get("name", "").strip()
                if name:
                    authors.append(name)

        venue = paper.get("venue", "") or ""
        pub_type = paper.get("publication_type", "") or ""

        volume = ""
        issue = ""
        pages = ""
        for key in ("volume", "issue", "pages", "page"):
            val = paper.get(key, "")
            if val:
                if key == "volume":
                    volume = str(val)
                elif key == "issue":
                    issue = str(val)
                elif key in ("pages", "page"):
                    pages = str(val)

        doi = ext.get("doi", "") or paper.get("doi", "") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        doi = doi.strip()

        url = paper.get("url", "") or ""
        if not url and doi:
            url = f"https://doi.org/{doi}"

        arxiv_id = ext.get("arxiv", "") or ""
        if not arxiv_id:
            raw_id = paper.get("paper_id", "")
            # Heuristic: arXiv IDs look like 2401.12345
            if re.match(r"\d{4}\.\d{4,5}(v\d+)?$", str(raw_id)):
                arxiv_id = str(raw_id)

        keywords_raw = paper.get("keywords", []) or []
        keywords = [k for k in keywords_raw if isinstance(k, str)]

        entry = cls(
            title=paper.get("title", "") or "",
            authors=authors,
            year=str(paper.get("year", "") or ""),
            journal=venue,
            volume=volume,
            issue=issue,
            pages=pages,
            publisher=paper.get("publisher", "") or "",
            doi=doi,
            url=url,
            arxiv_id=arxiv_id,
            openalex_id=ext.get("openalex", "") or "",
            crossref_id=ext.get("crossref", "") or "",
            abstract=paper.get("abstract", "") or "",
            keywords=keywords,
            language=paper.get("language", "") or "",
            publication_type=pub_type,
        )
        entry.citekey = generate_citekey(entry)
        return entry


# ────────────────────────────────────────────────────────────────────
# Citekey Generation
# ────────────────────────────────────────────────────────────────────

def _ascii_slug(text: str, max_len: int = 40) -> str:
    """Convert text to a safe ASCII slug."""
    # Normalize unicode → ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    # Remove non-alphanumeric
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "", text)
    return text[:max_len]


def _first_author_last_name(entry: CitationEntry) -> str:
    """Extract last name from first author."""
    if not entry.authors:
        return ""
    first = entry.authors[0]
    # Handle "Last, First" format
    if "," in first:
        return first.split(",")[0].strip()
    # Handle "First Last" format — take last word
    parts = first.strip().split()
    if parts:
        return parts[-1]
    return ""


def _short_title(title: str, max_words: int = 3) -> str:
    """Create a short slug from the title."""
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", title).split()
    # Skip very common short words
    skip = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "is", "are"}
    filtered = [w for w in words if w.lower() not in skip]
    return _ascii_slug(" ".join(filtered[:max_words]))


def generate_citekey(entry: CitationEntry) -> str:
    """Generate a deterministic citation key.

    Strategy (priority):
    1. firstAuthorLastName + year + shortTitle
    2. If DOI exists, append DOI suffix for uniqueness
    3. If authors missing, fall back to title slug
    """
    author_part = _ascii_slug(_first_author_last_name(entry), max_len=20)
    year_part = entry.year[:4] if entry.year else ""
    title_part = _short_title(entry.title)

    if author_part and year_part and title_part:
        base = f"{author_part}{year_part}{title_part}"
    elif title_part:
        base = f"{title_part}{year_part}"
    elif author_part:
        base = f"{author_part}{year_part}"
    else:
        base = f"unknown{year_part}"

    # Use DOI suffix for extra uniqueness if available
    if entry.doi:
        doi_suffix = _ascii_slug(entry.doi.replace("/", "_").replace(".", "_"), max_len=15)
        if doi_suffix and doi_suffix not in base:
            base = f"{base}_{doi_suffix}"

    # Sanitize
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "untitled"


# ────────────────────────────────────────────────────────────────────
# BibTeX Helpers
# ────────────────────────────────────────────────────────────────────

def _bibtex_escape(text: str) -> str:
    """Escape special characters for BibTeX values."""
    # Protect already-braced content, then escape
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    text = text.replace("~", "\\~{}")
    text = text.replace("^", "\\^{}")
    return text


def _bibtex_entry_type(entry: CitationEntry) -> str:
    """Determine BibTeX entry type."""
    pt = entry.publication_type.lower()
    if "journal" in pt or "article" in pt:
        return "article"
    if "conference" in pt or "proceedings" in pt or "inproceedings" in pt:
        return "inproceedings"
    if entry.arxiv_id and not entry.journal:
        return "misc"
    if entry.journal:
        return "article"
    return "misc"


# ────────────────────────────────────────────────────────────────────
# Renderers
# ────────────────────────────────────────────────────────────────────

def render_bibtex(entry: CitationEntry) -> str:
    """Render a single BibTeX entry."""
    etype = _bibtex_entry_type(entry)
    fields: list[tuple[str, str]] = []

    fields.append(("title", _bibtex_escape(entry.title)))
    if entry.authors:
        fields.append(("author", " and ".join(entry.authors)))
    if entry.journal:
        key = "booktitle" if etype == "inproceedings" else "journal"
        fields.append((key, _bibtex_escape(entry.journal)))
    if entry.year:
        fields.append(("year", entry.year[:4]))
    if entry.volume:
        fields.append(("volume", entry.volume))
    if entry.issue:
        fields.append(("number", entry.issue))
    if entry.pages:
        fields.append(("pages", entry.pages.replace("--", "-").replace("—", "-")))
    if entry.doi:
        fields.append(("doi", entry.doi))
    if entry.url:
        fields.append(("url", entry.url))
    if entry.publisher:
        fields.append(("publisher", _bibtex_escape(entry.publisher)))
    if entry.abstract:
        fields.append(("abstract", _bibtex_escape(entry.abstract[:500])))
    if entry.keywords:
        fields.append(("keywords", "; ".join(entry.keywords)))
    if entry.arxiv_id and etype == "misc":
        fields.append(("note", f"arXiv:{entry.arxiv_id}"))
        if not entry.url:
            fields.append(("url", f"https://arxiv.org/abs/{entry.arxiv_id}"))

    lines = [f"@{etype}{{{entry.citekey},"]
    for key, val in fields:
        lines.append(f"  {key} = {{{val}}},")
    lines.append("}")
    return "\n".join(lines)


def render_bibtex_batch(entries: list[CitationEntry]) -> str:
    """Render multiple BibTeX entries."""
    return "\n\n".join(render_bibtex(e) for e in entries)


def _ris_type(entry: CitationEntry) -> str:
    """Map publication type to RIS TY code."""
    pt = entry.publication_type.lower()
    if "journal" in pt or "article" in pt:
        return "JOUR"
    if "conference" in pt or "proceedings" in pt:
        return "CONF"
    if "book" in pt:
        return "BOOK"
    return "GEN"


def render_ris(entry: CitationEntry) -> str:
    """Render a single RIS entry."""
    lines: list[str] = []
    lines.append(f"TY  - {_ris_type(entry)}")
    if entry.title:
        lines.append(f"TI  - {entry.title}")
    for a in entry.authors:
        lines.append(f"AU  - {a}")
    if entry.year:
        lines.append(f"PY  - {entry.year[:4]}")
    if entry.journal:
        lines.append(f"JO  - {entry.journal}")
    if entry.volume:
        lines.append(f"VL  - {entry.volume}")
    if entry.issue:
        lines.append(f"IS  - {entry.issue}")
    if entry.pages:
        pages = entry.pages.replace("--", "-").replace("—", "-")
        parts = pages.split("-", 1)
        lines.append(f"SP  - {parts[0]}")
        if len(parts) > 1:
            lines.append(f"EP  - {parts[1]}")
    if entry.doi:
        lines.append(f"DO  - {entry.doi}")
    if entry.url:
        lines.append(f"UR  - {entry.url}")
    if entry.abstract:
        lines.append(f"AB  - {entry.abstract}")
    for kw in entry.keywords:
        lines.append(f"KW  - {kw}")
    lines.append("ER  - ")
    return "\n".join(lines)


def render_ris_batch(entries: list[CitationEntry]) -> str:
    """Render multiple RIS entries."""
    return "\n\n".join(render_ris(e) for e in entries)


def _split_author_name(name: str) -> dict[str, str]:
    """Split author name into family/given for CSL-JSON."""
    if "," in name:
        parts = name.split(",", 1)
        return {"family": parts[0].strip(), "given": parts[1].strip()}
    parts = name.strip().split()
    if len(parts) >= 2:
        return {"family": parts[-1], "given": " ".join(parts[:-1])}
    return {"family": name, "given": ""}


def _csl_json_dict(entry: CitationEntry) -> dict[str, Any]:
    """Build a CSL-JSON dict for a single entry."""
    csl_type = "article-journal"
    pt = entry.publication_type.lower()
    if "conference" in pt or "proceedings" in pt:
        csl_type = "paper-conference"
    elif not entry.journal and entry.arxiv_id:
        csl_type = "manuscript"  # preprint fallback

    item: dict[str, Any] = {
        "id": entry.citekey,
        "type": csl_type,
        "title": entry.title,
    }

    if entry.authors:
        item["author"] = [_split_author_name(a) for a in entry.authors]

    if entry.year:
        try:
            y = int(entry.year[:4])
            item["issued"] = {"date-parts": [[y]]}
        except ValueError:
            pass

    if entry.journal:
        item["container-title"] = entry.journal
    if entry.volume:
        item["volume"] = entry.volume
    if entry.issue:
        item["issue"] = entry.issue
    if entry.pages:
        item["page"] = entry.pages.replace("--", "-").replace("—", "-")
    if entry.doi:
        item["DOI"] = entry.doi
    if entry.url:
        item["URL"] = entry.url
    if entry.publisher:
        item["publisher"] = entry.publisher
    if entry.abstract:
        item["abstract"] = entry.abstract
    if entry.keywords:
        item["keyword"] = ", ".join(entry.keywords)
    if entry.language:
        item["language"] = entry.language

    return item


def render_csl_json(entry: CitationEntry) -> str:
    """Render a single CSL-JSON entry as a JSON string."""
    return json.dumps(_csl_json_dict(entry), ensure_ascii=False, indent=2)


def render_csl_json_batch(entries: list[CitationEntry]) -> str:
    """Render multiple CSL-JSON entries as a JSON array."""
    items = [_csl_json_dict(e) for e in entries]
    return json.dumps(items, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────────────────────────────
# Human-readable styles: APA, MLA, GB/T 7714
# ────────────────────────────────────────────────────────────────────

def _format_authors_apa(authors: list[str]) -> str:
    """Format authors per APA style."""
    if not authors:
        return ""
    formatted = []
    for a in authors:
        if "," in a:
            formatted.append(a.strip())
        else:
            parts = a.strip().split()
            if len(parts) >= 2:
                family = parts[-1]
                given = " ".join(parts[:-1])
                initials = ". ".join(g[0] for g in given.split() if g) + "."
                formatted.append(f"{family}, {initials}")
            else:
                formatted.append(a)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    # >20 authors: first 19 ... last
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def _format_authors_mla(authors: list[str]) -> str:
    """Format authors per MLA style."""
    if not authors:
        return ""
    if len(authors) == 1:
        a = authors[0]
        if "," not in a:
            parts = a.strip().split()
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return a
    if len(authors) == 2:
        first = authors[0]
        if "," not in first:
            parts = first.split()
            if len(parts) >= 2:
                first = f"{parts[-1]}, {' '.join(parts[:-1])}"
        return f"{first}, and {authors[1]}"
    # 3+: first author et al.
    first = authors[0]
    if "," not in first:
        parts = first.split()
        if len(parts) >= 2:
            first = f"{parts[-1]}, {' '.join(parts[:-1])}"
    return f"{first}, et al."


def render_apa(entry: CitationEntry) -> str:
    """Render in APA 7th edition style."""
    parts: list[str] = []

    author_str = _format_authors_apa(entry.authors)
    if author_str:
        parts.append(f"{author_str}.")

    if entry.year:
        parts.append(f"({entry.year[:4]}).")
    else:
        parts.append("(n.d.).")

    if entry.title:
        title = entry.title
        if not title.endswith("."):
            title += "."
        parts.append(title)

    if entry.journal:
        journal_part = f"*{entry.journal}*"
        if entry.volume:
            journal_part += f", *{entry.volume}*"
            if entry.issue:
                journal_part += f"({entry.issue})"
        if entry.pages:
            journal_part += f", {entry.pages}"
        journal_part += "."
        parts.append(journal_part)

    if entry.doi:
        parts.append(f"https://doi.org/{entry.doi}")
    elif entry.url:
        parts.append(entry.url)

    return " ".join(parts)


def render_mla(entry: CitationEntry) -> str:
    """Render in MLA 9th edition style."""
    parts: list[str] = []

    author_str = _format_authors_mla(entry.authors)
    if author_str:
        parts.append(f"{author_str}.")

    if entry.title:
        title = entry.title
        if not title.endswith("."):
            title += "."
        parts.append(f'"{title}"')

    if entry.journal:
        journal = f"*{entry.journal}*"
        if entry.volume:
            journal += f", vol. {entry.volume}"
        if entry.issue:
            journal += f", no. {entry.issue}"
        parts.append(f"{journal},")

    if entry.year:
        parts.append(f"{entry.year[:4]},")

    if entry.pages:
        parts.append(f"pp. {entry.pages}.")

    if entry.doi:
        parts.append(f"doi:{entry.doi}.")
    elif entry.url:
        parts.append(entry.url + ".")

    return " ".join(parts)


def _format_authors_gbt(authors: list[str]) -> str:
    """Format authors for GB/T 7714 style."""
    if not authors:
        return ""
    formatted = []
    for a in authors:
        # GB/T keeps original name order; for Chinese names keep as-is
        formatted.append(a.strip())
    if len(formatted) <= 3:
        return ", ".join(formatted)
    return ", ".join(formatted[:3]) + ", 等"


def render_gbt7714(entry: CitationEntry) -> str:
    """Render in GB/T 7714-2015 style (simplified practical version).

    Format patterns:
    - Journal: [N] AUTHORS. TITLE[J]. JOURNAL, YEAR, VOLUME(ISSUE): PAGES.
    - Conference: [C] AUTHORS. TITLE[C]//CONFERENCE. PLACE: PUBLISHER, YEAR: PAGES.
    - Preprint/Online: [EB/OL] AUTHORS. TITLE[EB/OL]. (YEAR)[引用日期]. URL.
    """
    parts: list[str] = []

    author_str = _format_authors_gbt(entry.authors)
    if author_str:
        parts.append(author_str + ".")

    if entry.title:
        title = entry.title
        if not title.endswith("."):
            title += "."

        pt = entry.publication_type.lower()
        if "conference" in pt or "proceedings" in pt:
            parts.append(f"{title}[C]")
        elif entry.journal:
            parts.append(f"{title}[J]")
        elif entry.arxiv_id and not entry.journal:
            parts.append(f"{title}[EB/OL]")
        else:
            parts.append(f"{title}[J]")

    if entry.journal:
        journal_part = f" {entry.journal},"
        if entry.year:
            journal_part += f" {entry.year[:4]},"
        if entry.volume:
            journal_part += f" {entry.volume}"
            if entry.issue:
                journal_part += f"({entry.issue})"
        if entry.pages:
            journal_part += f": {entry.pages}."
        elif not entry.volume and entry.year:
            journal_part = journal_part.rstrip(",") + "."
        parts.append(journal_part)
    elif entry.arxiv_id and not entry.journal:
        url = entry.url or f"https://arxiv.org/abs/{entry.arxiv_id}"
        year_str = entry.year[:4] if entry.year else ""
        parts.append(f" ({year_str})[EB/OL]. {url}.")
    elif entry.url:
        year_str = entry.year[:4] if entry.year else ""
        parts.append(f" ({year_str}). {entry.url}.")

    return "".join(parts)


# ────────────────────────────────────────────────────────────────────
# Unified render dispatch
# ────────────────────────────────────────────────────────────────────

_RENDERERS = {
    "bibtex": (render_bibtex, render_bibtex_batch),
    "ris": (render_ris, render_ris_batch),
    "csl-json": (render_csl_json, render_csl_json_batch),
    "apa": (render_apa, None),
    "mla": (render_mla, None),
    "gbt7714": (render_gbt7714, None),
}

SUPPORTED_FORMATS = frozenset(_RENDERERS.keys())


def render_citation(entry: CitationEntry, fmt: str) -> str:
    """Render a single citation in the given format."""
    fmt = fmt.lower().strip()
    if fmt not in _RENDERERS:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}")
    renderer = _RENDERERS[fmt][0]
    return renderer(entry)


def render_citations(entries: list[CitationEntry], fmt: str) -> str:
    """Render multiple citations. For text styles, entries are separated by blank lines."""
    fmt = fmt.lower().strip()
    if fmt not in _RENDERERS:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}")
    single_fn, batch_fn = _RENDERERS[fmt]
    if batch_fn:
        return batch_fn(entries)
    # For text styles: join with double newlines
    return "\n\n".join(single_fn(e) for e in entries)


def papers_to_entries(papers: list[dict[str, Any]]) -> list[CitationEntry]:
    """Convert a list of paper dicts to CitationEntry list, with deduplication by citekey."""
    entries = [CitationEntry.from_paper(p) for p in papers]
    _dedupe_citekeys(entries)
    return entries


def _dedupe_citekeys(entries: list[CitationEntry]) -> None:
    """Deduplicate citekeys in-place.

    First occurrence keeps the original citekey.  Second, third, … occurrences
    of the *same* original citekey get ``_1``, ``_2``, ``_3``, … suffixes so
    every entry is guaranteed unique within the batch.
    """
    counts: dict[str, int] = {}  # original_citekey → how many seen so far
    for e in entries:
        base = e.citekey
        n = counts.get(base, 0)
        if n > 0:
            e.citekey = f"{base}_{n}"
        counts[base] = n + 1
