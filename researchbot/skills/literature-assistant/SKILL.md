---
name: literature-assistant
description: Research literature assistant for searching, retrieving, and managing academic papers from arXiv.
homepage: https://arxiv.org
metadata: {"researchbot":{"emoji":"📚"}}
---

# Literature Assistant

Tools for searching, retrieving, downloading, extracting text, summarizing, and managing academic research papers from arXiv.

## When to use (trigger phrases)

Use this skill immediately when the user asks any of:
- "paper", "arxiv", "research paper"
- "find me papers on", "search for papers"
- "帮我找论文", "搜索论文"
- "related work", "literature review"
- "survey on", "综述"
- "这篇论文", "this paper"
- "save this paper", "收藏论文"
- "帮我总结", "总结这篇论文"
- "download pdf", "下载PDF"
- "详细总结", "深入理解"
- "提炼方法", "实验结论"
- "compare papers", "对比论文"
- "compare these two papers"
- "literature review for", "主题综述"
- "related work on", "相关工作"
- "research gaps", "研究空白"

## Available Tools

### 1. paper_search

Search arXiv for papers by topic.

```python
paper_search(query="ti:transformer AND au:hinton", max_results=5, sort_by="relevance")
```

**Query syntax examples:**
- `ti:transformer` - title contains "transformer"
- `au:hinton` - author is "hinton"
- `abs:reinforcement learning` - abstract contains "reinforcement learning"
- `cat:cs.CL` - category is "cs.CL"
- `all:attention` - search everywhere

### 2. paper_get

Get detailed information about a specific paper.

```python
paper_get(paper_id="2401.12345v2")
paper_get(url="https://arxiv.org/abs/2401.12345")
```

### 3. paper_download_pdf

Download arXiv paper PDF to local storage.

```python
paper_download_pdf(paper_id="2401.12345v2")
paper_download_pdf(url="https://arxiv.org/abs/2401.12345")
paper_download_pdf(paper_id="2401.12345", overwrite=True)
```

**Note:** If PDF already exists locally and overwrite=False, will reuse existing file.

### 4. paper_extract_text

Extract text content from a local PDF.

```python
paper_extract_text(paper_id="2401.12345")
paper_extract_text(pdf_path="/path/to/paper.pdf")
paper_extract_text(paper_id="2401.12345", overwrite=True)
```

**Note:** Requires PDF to be downloaded first. If text already extracted and overwrite=False, will reuse.

### 5. paper_save

Save a paper to local literature knowledge base.

```python
paper_save(paper={...paper_object...}, topic="NLP", tags=["transformer", "attention"])
```

### 6. paper_summarize

Generate structured summary for a paper. **Enhanced to use full text when available.**

```python
paper_summarize(paper_id="2401.12345")
paper_summarize(paper_id="2401.12345", save=True)
paper_summarize(paper_id="2401.12345", overwrite=True)
```

**Summary Priority:**
1. If extracted full text exists → uses full text for better summary
2. Otherwise falls back to abstract only
3. `overwrite=True` forces regeneration even if summary exists

### 7. paper_compare

Compare multiple papers across key dimensions (problem, method, findings, limitations). Can find relevant papers automatically by topic.

```python
# Compare specific papers by ID
paper_compare(paper_ids=["2401.12345", "2301.45678", "2212.34567"])

# Compare papers given as objects
paper_compare(papers=[{...paper_object...}, {...}])

# Find relevant papers from local storage by topic and compare
paper_compare(topic="retrieval augmented generation", max_papers=5)
```

**Behavior:**
- If `paper_ids` or `papers` provided → compares those directly
- If only `topic` provided → searches local `literature/` for relevant papers
- Returns structured comparison with common patterns, differences, and research gaps
- Each claim is attributed to specific papers

### 8. paper_review

Generate a structured literature review for a research topic using locally saved papers. Saves to `literature/reviews/<topic_slug>.json` and `.md`.

```python
# Generate literature review for a topic
paper_review(topic="retrieval augmented generation", max_papers=10)

# Overwrite existing review
paper_review(topic="RAG", overwrite=True)
```

**Output includes:**
- Introduction to the research area
- Per-paper summary with key contributions
- Thematic analysis grouping papers by themes
- Synthesis of common/conflicting findings
- Research gaps with potential directions
- Proper citations with [paper_id] references

**Note:** Only uses papers saved in local `literature/` storage. Will not fetch new papers.

## Workflow for Multi-Paper Analysis

### Quick Comparison
1. **Find papers**: Use `paper_search` or direct IDs
2. **Compare**: Use `paper_compare(topic="...")` to auto-find and compare from local storage
3. **Review**: Use `paper_review(topic="...")` to generate full literature review

### Deep Comparison
1. **Search & Save**: Find papers with `paper_search` and save with `paper_save`
2. **Download & Extract**: Get PDFs with `paper_download_pdf` and extract with `paper_extract_text`
3. **Summarize**: Generate summaries with `paper_summarize` (uses full text)
4. **Compare**: Use `paper_compare(paper_ids=[...])` for specific papers
5. **Review**: Use `paper_review(topic="...")` for comprehensive literature review

## Workflow for Deep Paper Understanding

1. **Search**: Use `paper_search` to find relevant papers
2. **Get details**: Use `paper_get` for full metadata
3. **Download PDF**: Use `paper_download_pdf` to get local PDF
4. **Extract text**: Use `paper_extract_text` to get full text content
5. **Generate summary**: Use `paper_summarize` - now enhanced with full text
6. **Save**: Use `paper_save` to persist everything

## Output Format

All tools return structured text with:
- Paper ID, title, authors
- Publication date and category
- Abstract/summary
- Links to arXiv pages and PDFs
- Local paths when downloaded/extracted

## Guidelines

1. **Prioritize literature tools** over general web search for research papers
2. **Deep understanding workflow**: When user asks for "detailed summary", "understand paper", or "methods and experiments":
   - First download PDF: `paper_download_pdf`
   - Then extract text: `paper_extract_text`
   - Then summarize: `paper_summarize` (uses full text)
3. **Flag information depth**: If using only abstract, say "Note: Summary based on abstract only. For deeper understanding, PDF download recommended."
4. **Always cite sources**: Include arXiv ID and URL when referencing findings
5. **Structure responses**: Use headers, bullet points, and tables when appropriate
6. **Flag uncertainty**: If information is incomplete, say "I'm not sure" rather than guessing
7. **Suggest next steps**: Offer to save papers, download PDF, or generate detailed summary

## Data Sources

ResearchBot now supports multiple paper metadata sources:

### arXiv
- **Best for**: Preprints, recent papers, CS/ML papers
- **Provides**: Titles, abstracts, authors, categories, PDF links
- **URL**: https://arxiv.org

### Crossref
- **Best for**: Published journal articles, DOI resolution, citation counts
- **Provides**: DOI, journal info, volume/issue/pages, publisher, references
- **API**: https://api.crossref.org
- **Requires**: Optional mailto for polite pool (recommended)

### OpenAlex
- **Best for**: Comprehensive paper metadata, concepts/topics, open access status
- **Provides**: Citation counts, concepts, referenced works, OA status
- **API**: https://api.openalex.org
- **Requires**: API key (mandatory since 2026-02-13)

## Automatic Enrichment

When you save an arXiv paper or search for papers, ResearchBot can automatically enrich the metadata:

1. If the paper has a DOI, Crossref is queried first for journal info
2. OpenAlex provides citation counts, concepts, and references
3. The enriched metadata is merged into the local paper record

To enable automatic enrichment, configure your API keys (see Configuration section).

## New Tools

### 9. paper_enrich

Enrich existing paper metadata with Crossref and OpenAlex data.

```python
# Enrich by arXiv ID
paper_enrich(paper_id="2401.12345")

# Enrich by DOI
paper_enrich(doi="10.1000/xyz123")

# Enrich by title
paper_enrich(title="Attention is All You Need")

# Enrich and save back to literature storage
paper_enrich(paper_id="2401.12345", save=True)
```

**What it enriches:**
- DOI (if not present)
- Journal/venue information
- Citation count
- Referenced works
- Concepts and topic tags
- Open access status

### 10. crossref_search

Search Crossref directly for published papers.

```python
# Search by query
crossref_search(query="transformer architecture")

# Search with filters
crossref_search(query="machine learning", year=2023, author="Hinton")

# Search by DOI
crossref_search(query="", doi="10.1000/xyz123")
```

**Crossref is best for:**
- Finding published journal articles
- Resolving DOIs to full metadata
- Getting reference lists
- Journal volume/issue/page info

### 11. openalex_search

Search OpenAlex for papers with rich metadata.

```python
# Search by query
openalex_search(query="attention mechanism")

# Search with filters
openalex_search(query="deep learning", year=2023, author="Bengio")

# Search by title
openalex_search(query="language models", title="BERT")
```

**OpenAlex is best for:**
- Getting citation counts
- Finding related works
- Getting concept/topic classifications
- Open access status information

## Enrichment Workflow

For existing arXiv papers in your literature storage:

1. **Enrich**: `paper_enrich(paper_id="2401.12345", save=True)`
2. **View**: Check `literature/papers/2401.12345.json` for enriched fields
3. **Update**: Re-enrich anytime to get latest citation counts

## Limitations

- OCR not supported for scanned PDFs
- No citation formatting built-in
- Summary quality depends on extracted text quality
- OpenAlex API key required for production use (without key, rate limited)

## Tips

- Use `max_results=10` for broader discovery, `max_results=3` for focused search
- Sort by `submittedDate` for latest papers, `relevance` for most related
- After finding a paper, offer to `paper_save` it with relevant topic/tags
- For important papers, always offer to download PDF and generate full-text enhanced summary
- Local storage structure:
  - `literature/papers/<paper_id>.json` - metadata
  - `literature/papers/<paper_id>.md` - readable markdown
  - `literature/pdfs/<paper_id>.pdf` - downloaded PDF
  - `literature/extracted/<paper_id>.txt` - extracted text
  - `literature/reviews/<topic_slug>.json` - literature review (structured)
  - `literature/reviews/<topic_slug>.md` - literature review (markdown)
