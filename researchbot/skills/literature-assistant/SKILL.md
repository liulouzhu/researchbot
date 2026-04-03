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
- "citation", "cite this", "bibtex", "bibliography"
- "引用", "导出引用", "参考文献格式", "参考文献"

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

## Citation Export

### 14. paper_cite

Export paper citations in standard academic formats for use in manuscripts, reference managers, and bibliographies.

```python
# Export single paper as BibTeX
paper_cite(paper_id="2401.12345", format="bibtex")

# Export multiple papers as RIS (for Zotero/EndNote)
paper_cite(paper_ids=["2401.12345", "2301.45678"], format="ris")

# Export all local papers as CSL-JSON
paper_cite(format="csl-json")

# Export to file
paper_cite(paper_id="2401.12345", format="bibtex", output="file", path="refs.bib")

# Export a paper dict directly (without loading from local storage)
paper_cite(paper={...paper_object...}, format="apa")
```

**Supported formats:**

| Format | Description | Typical Use |
|--------|-------------|-------------|
| `bibtex` | BibTeX entries | LaTeX manuscripts |
| `ris` | RIS tag format | EndNote, Zotero, Mendeley |
| `csl-json` | CSL-JSON objects | Pandoc, Citeproc |
| `apa` | APA 7th edition | Social science papers |
| `mla` | MLA 9th edition | Humanities papers |
| `gbt7714` | GB/T 7714-2015 | Chinese academic papers |

**Parameters:**
- `paper_id`: Export a single paper by ID
- `paper_ids`: Export multiple papers by ID list
- `paper`: Export a paper dict directly
- `format`: Output format (default: `bibtex`)
- `output`: `text` (default) or `file`
- `path`: File path for `file` mode (defaults to `literature/citations/export.<ext>`)

**Citekey format:** Deterministic keys like `vaswani2017attention` (first author last name + year + title words). DOI suffix appended for uniqueness.

**Workflow:** When user asks for citations, references, bibliography, or export:
1. If papers are not yet saved, search and save first
2. Use `paper_cite` with the desired format
3. For manuscript preparation, export to file with `output="file"`

## Limitations

- OCR not supported for scanned PDFs
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
  - `literature/indexes/search.sqlite3` - semantic search index (SQLite)

## Local Semantic Search

ResearchBot supports **local semantic search** over your saved papers. This enables:
- Semantic similarity search (find papers by meaning, not just keywords)
- Hybrid search combining keyword + vector matching
- Filtering by topic, tags, year, categories, source
- Automatic reranking with LLM for better relevance

### How It Works

1. **Automatic Indexing**: Papers are indexed automatically when saved/summarized/enriched
2. **FTS5 Keyword Search**: Traditional keyword matching with BM25 ranking
3. **Vector Search**: Using sqlite-vec for semantic similarity (if available)
4. **Hybrid Fusion**: Combines keyword and vector scores using RRF
5. **LLM Rerank**: Re-ranks top candidates using a language model

### 12. paper_search_local

Search local papers using semantic search.

```python
# Basic semantic search
paper_search_local(query="machine learning security")

# With filters
paper_search_local(
    query="deep learning for graphs",
    topic="graph neural networks",
    tags=["GNN", "attention"],
    year_from=2022,
    year_to=2024,
    top_k=10,
    rerank=True
)

# Search by source
paper_search_local(query="transformer", source="arxiv")
```

**Parameters:**
- `query` (required): Search query string
- `top_k`: Maximum results to return (default: 10)
- `topic`: Filter by topic (substring match)
- `tags`: Filter by tags (any match)
- `year`: Filter by exact year
- `year_from` / `year_to`: Year range filter
- `categories`: Filter by categories
- `source`: Filter by source (arxiv, crossref, openalex)
- `rerank`: Apply LLM reranking (default: True)

### 13. paper_index

Manage the local search index.

```python
# Rebuild entire index from all local papers
paper_index(rebuild=True)

# Index a specific paper
paper_index(paper_id="2401.12345")
```

### Index Updates

The search index is **automatically updated** when you:
- Save a paper (`paper_save`)
- Generate a summary (`paper_summarize`)
- Enrich paper metadata (`paper_enrich`)

You normally don't need to manually rebuild the index unless:
- You've added papers before semantic search was configured
- The index file was corrupted
- You want to re-index with new embedding settings

### Fallback Behavior

The system distinguishes between "disabled" and "unavailable":

**embedding 未配置（`embeddingApiKey` 为空）：**
- 完全不调用 embedding 服务，不报错
- 纯 FTS5 关键词检索，滤波功能正常工作
- 检索质量略有下降（无语义相似度）

**sqlite-vec 配置关闭（`enableSqliteVec=false`）：**
- 完全不尝试加载 sqlite-vec 扩展
- 纯 FTS5 检索

**sqlite-vec 环境不可用（但 `enableSqliteVec=true`）：**
- 尝试加载失败后自动降级到纯 FTS5
- 不报错，不影响基础功能

**embedding 服务运行时失败：**
- runtime 状态标记为不可用
- 已有结果不受影响，不报错

### Configuration

The semantic search is configured via the `literature.semanticSearch` section:

```json
{
  "literature": {
    "semanticSearch": {
      "sqliteDbPath": "literature/indexes/search.sqlite3",
      "embeddingModel": "text-embedding-v4",
      "embeddingProvider": "dashscope",
      "embeddingApiKey": "your-api-key",
      "embeddingApiBase": "",
      "enableSqliteVec": true,
      "enableRerank": true,
      "rerankTopK": 20,
      "hybridSearchRrfK": 60,
      "lexicalWeight": 0.3,
      "vectorWeight": 0.7
    }
  }
}
```

**配置说明：**
- `embeddingApiKey` **不配置则不启用向量检索**，FTS5 检索仍然正常工作
- `enableSqliteVec` 设为 `false` 则完全不加载 sqlite-vec 扩展
- `rerankTopK`: LLM rerank 的候选数量
- `hybridSearchRrfK`: RRF 融合的 k 值

**阿里云 dashscope 配置示例：**
```bash
export RESEARCHBOT_LITERATURE__SEMANTIC_SEARCH__EMBEDDING_PROVIDER="dashscope"
export RESEARCHBOT_LITERATURE__SEMANTIC_SEARCH__EMBEDDING_API_KEY="your-dashscope-api-key"
export RESEARCHBOT_LITERATURE__SEMANTIC_SEARCH__EMBEDDING_MODEL="text-embedding-v4"
```

**最小配置（仅 FTS5）：**
```json
{
  "literature": {
    "semanticSearch": {
      "sqliteDbPath": "literature/indexes/search.sqlite3"
    }
  }
}
```

### SQLite 文件位置

- 路径：`workspace/literature/indexes/search.sqlite3`
- 数据库包含表：`papers`, `papers_fts`, `paper_embeddings`, `paper_vectors` (如果 sqlite-vec 可用)
