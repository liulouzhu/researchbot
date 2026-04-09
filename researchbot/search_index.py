"""SQLite-based local semantic search index.

Supports:
- Hybrid search: FTS5 keyword search + sqlite-vec vector search
- Fallback to FTS5-only when sqlite-vec is unavailable
- Incremental updates via content_hash
- Filtering by topic, tags, year, categories

Schema:
  papers          - paper metadata
  papers_fts      - FTS5 virtual table on title, abstract, summary, keywords, topic_tags
  paper_embeddings - vector embeddings for each paper
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from researchbot.config.schema import SemanticSearchConfig
from researchbot.embedding import EmbeddingClient, bytes_to_vector, vector_to_bytes
from researchbot.knowledge_graph import KnowledgeGraph


def _compute_content_hash(paper: dict[str, Any]) -> str:
    """Compute a hash of paper content that determines when to re-index."""
    parts = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        _summary_text(paper.get("summary", {})),
        ",".join(sorted(paper.get("keywords", []))),
        ",".join(sorted(paper.get("topic_tags", []))),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _summary_text(summary: dict[str, Any] | str) -> str:
    """Extract searchable text from summary dict or string."""
    if isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        parts = [
            summary.get("one_sentence", ""),
            summary.get("problem", ""),
            summary.get("method", ""),
            summary.get("findings", ""),
        ]
        return " ".join(parts)
    return ""


def _build_search_text(paper: dict[str, Any]) -> str:
    """Build the combined search text for FTS5 indexing."""
    parts = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        _summary_text(paper.get("summary", {})),
        " ".join(paper.get("keywords", [])),
        " ".join(paper.get("topic_tags", [])),
    ]
    return " ".join(parts)


class SearchIndex:
    """SQLite-based local semantic search index.

    Usage:
        index = SearchIndex(workspace_path / "literature/indexes/search.sqlite3")
        await index.initialize()

        # After saving a paper
        await index.upsert_paper(paper_dict)

        # Search
        results = await index.search("machine learning", top_k=10, topic="AI")
    """

    def __init__(
        self,
        db_path: str | Path,
        config: SemanticSearchConfig | None = None,
    ):
        self._db_path = Path(db_path)
        self._config = config or SemanticSearchConfig()
        self._conn: sqlite3.Connection | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._sqlite_vec_available: bool = False
        self._graph_initialized: bool = False

    @property
    def embedding_client(self) -> EmbeddingClient:
        """Get or create the embedding client."""
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._config)
        return self._embedding_client

    @property
    def sqlite_vec_available(self) -> bool:
        """Whether sqlite-vec extension is available."""
        return self._sqlite_vec_available

    def get_graph(self) -> KnowledgeGraph:
        """Get the KnowledgeGraph instance sharing this SearchIndex's connection."""
        return KnowledgeGraph(self)

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the SQLite connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def initialize(self) -> None:
        """Initialize the database schema."""
        conn = self._get_conn()

        # Create papers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                authors_json TEXT NOT NULL DEFAULT '[]',
                year TEXT NOT NULL DEFAULT '',
                venue TEXT NOT NULL DEFAULT '',
                publication_type TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL DEFAULT '',
                pdf_url TEXT NOT NULL DEFAULT '',
                abstract TEXT NOT NULL DEFAULT '',
                summary_json TEXT NOT NULL DEFAULT '{}',
                topic TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                categories_json TEXT NOT NULL DEFAULT '[]',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                doi TEXT NOT NULL DEFAULT '',
                openalex_id TEXT NOT NULL DEFAULT '',
                crossref_id TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                graph_sync_status TEXT NOT NULL DEFAULT 'ok',
                graph_sync_error TEXT NOT NULL DEFAULT ''
            )
        """)

        # Migration: add keywords_json if missing (existing DBs before this column)
        try:
            conn.execute("ALTER TABLE papers ADD COLUMN keywords_json TEXT NOT NULL DEFAULT '[]'")
        except Exception:
            pass  # Column already exists (new DB)

        # Create FTS5 virtual table for keyword search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                paper_id,
                title,
                abstract,
                summary,
                keywords,
                topic_tags
            )
        """)

        # Create triggers to keep FTS5 in sync
        # Note: Using simpler triggers without the content= option for reliability
        try:
            conn.execute("DROP TRIGGER IF EXISTS papers_fts_insert")
        except Exception:
            pass
        try:
            conn.execute("DROP TRIGGER IF EXISTS papers_fts_update")
        except Exception:
            pass
        try:
            conn.execute("DROP TRIGGER IF EXISTS papers_fts_delete")
        except Exception:
            pass

        conn.execute("""
            CREATE TRIGGER papers_fts_insert AFTER INSERT ON papers BEGIN
                INSERT INTO papers_fts(paper_id, title, abstract, summary, keywords, topic_tags)
                VALUES (
                    NEW.paper_id,
                    NEW.title,
                    NEW.abstract,
                    json_extract(NEW.summary_json, '$.one_sentence') || ' ' ||
                    json_extract(NEW.summary_json, '$.problem') || ' ' ||
                    json_extract(NEW.summary_json, '$.method') || ' ' ||
                    json_extract(NEW.summary_json, '$.findings'),
                    REPLACE(REPLACE(json_extract(NEW.keywords_json, '$'), '["', ''), '"]', ''),
                    NEW.topic || ' ' || json_extract(NEW.tags_json, '$')
                );
            END
        """)

        # Create paper_embeddings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_embeddings (
                paper_id TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Try to load sqlite-vec only if explicitly enabled in config
        self._sqlite_vec_available = False
        self._vector_dim: int | None = None
        if self._config.enable_sqlite_vec:
            try:
                # Get path to vec0.dll from sqlite-vec package
                try:
                    import sqlite_vec as _sv
                    vec0_dll = os.path.join(os.path.dirname(_sv.__file__), "vec0.dll")
                except Exception:
                    vec0_dll = "vec0.dll"  # Fallback to searching PATH

                conn.enable_load_extension(True)
                # Try loading with full path first (needed on Windows)
                loaded = False
                for entry_point in [None, "sqlite3_vec_init"]:
                    try:
                        if entry_point:
                            conn.execute(f"SELECT load_extension('{vec0_dll}', '{entry_point}')")
                        else:
                            conn.execute(f"SELECT load_extension('{vec0_dll}')")
                        loaded = True
                        logger.info("sqlite-vec loaded with entry point: {}", entry_point or "default")
                        break
                    except Exception:
                        continue

                if not loaded:
                    raise Exception("Could not load vec0 extension")

                # Check version - sqlite-vec 0.1.x uses vec_version(), newer versions use vec0_version()
                try:
                    result = conn.execute("SELECT vec0_version()").fetchone()
                    logger.info("sqlite-vec loaded (vec0_version): {}", result)
                except Exception:
                    try:
                        result = conn.execute("SELECT vec_version()").fetchone()
                        logger.info("sqlite-vec loaded (vec_version): {}", result)
                    except Exception:
                        raise Exception("No version function found")
                self._sqlite_vec_available = True
                logger.info("sqlite-vec enabled and available, vector search enabled")
            except Exception as e:
                logger.warning(
                    "sqlite-vec enabled but not available ({}), falling back to FTS5-only search", e
                )

        # Create sqlite-vec virtual table if available and enabled
        # Note: sqlite-vec 0.1.x requires explicit dimension (e.g. float[1536]), newer versions support float[0]
        if self._sqlite_vec_available:
            try:
                # Use configured dimension, or default based on model
                dim = self._config.embedding_dimension
                if dim <= 0:
                    # Auto-detect: text-embedding-v4 standard = 1536, compatible-mode = 1024
                    # Use 1536 as safe default
                    dim = 1536

                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS paper_vectors USING vec0(
                        paper_id TEXT,
                        embedding float[{dim}]
                    )
                """)
                self._vector_dim = dim
                logger.info("paper_vectors table created with dimension {}", dim)
            except Exception as e:
                logger.warning("Could not create paper_vectors table: {}", e)
                self._sqlite_vec_available = False

        conn.commit()

        # Initialize knowledge graph tables
        kg = KnowledgeGraph(self)
        kg.initialize()
        self._graph_initialized = True

    def _ensure_graph_initialized(self) -> None:
        """Synchronously ensure graph tables exist.

        Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
        Bypasses async initialize() to avoid event-loop issues in sync contexts.
        """
        if getattr(self, "_graph_initialized", False):
            return
        conn = self._get_conn()
        kg = KnowledgeGraph(self)
        kg.initialize()
        self._graph_initialized = True

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def upsert_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a paper in the index.

        Returns a dict with:
        - content_changed: bool — True if new or changed (FTS/embedding rebuilt)
        - graph_sync_status: str — 'ok' or 'failed'
        - graph_sync_error: str — error message if failed, else ''

        Metadata field update semantics (consistent with knowledge graph edges):
        - key absent from paper dict → preserve existing value in papers table
        - key present as [] / {}     → overwrite with empty collection
        - key present with value    → overwrite with new value

        Fields covered by this semantics: authors, summary, tags/topic_tags,
        categories. All other fields (title, abstract, url, year, etc.) are
        always written from the paper dict (missing → empty-string default).

        An "effective paper" is constructed by merging the incoming paper with
        preserved values from the existing database row. This effective paper is
        used for content_hash computation, FTS rebuilding, and embedding generation,
        ensuring that partial updates do not spuriously trigger re-indexing.

        When a partial paper object (e.g. from PaperSaveTool) is passed,
        fields not present in the dict are preserved, so enrichment data
        (summary, categories, etc.) is not accidentally wiped.
        """
        paper_id = paper.get("paper_id")
        if not paper_id:
            return {"content_changed": False, "graph_sync_status": "skipped", "graph_sync_error": "no paper_id"}

        conn = self._get_conn()

        # Check existing row for change detection and for preserving values
        existing_row = conn.execute(
            "SELECT content_hash, authors_json, tags_json, categories_json, summary_json, abstract, title, keywords_json FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()

        # ------------------------------------------------------------------
        # Parse nested fields using presence-aware logic.
        # These become the serialized forms written to the DB.
        # ------------------------------------------------------------------

        # authors_json
        has_authors = "authors" in paper
        if has_authors:
            authors = paper["authors"]
            authors_json = json.dumps(authors if isinstance(authors, list) else [], ensure_ascii=False)
        else:
            authors_json = existing_row["authors_json"] if existing_row else "[]"

        # tags / topic_tags
        has_tags = "tags" in paper or "topic_tags" in paper
        if has_tags:
            tags = paper.get("topic_tags", paper.get("tags", []))
            tags_json = json.dumps(tags if isinstance(tags, list) else [], ensure_ascii=False)
        else:
            tags_json = existing_row["tags_json"] if existing_row else "[]"

        # categories
        has_categories = "categories" in paper
        if has_categories:
            categories = paper["categories"]
            categories_json = json.dumps(categories if isinstance(categories, list) else [], ensure_ascii=False)
        else:
            categories_json = existing_row["categories_json"] if existing_row else "[]"

        # summary
        has_summary = "summary" in paper
        if has_summary:
            summary = paper["summary"]
            if isinstance(summary, dict):
                summary_json = json.dumps(summary, ensure_ascii=False)
            elif isinstance(summary, str):
                summary_json = json.dumps({"text": summary}, ensure_ascii=False)
            else:
                summary_json = "{}"
        else:
            summary_json = existing_row["summary_json"] if existing_row else "{}"

        # abstract — preserve from existing row when not in input; used in
        # effective_paper so that _compute_content_hash is stable across
        # partial updates that omit the abstract field.
        has_abstract = "abstract" in paper
        if has_abstract:
            eff_abstract = paper["abstract"]
        else:
            eff_abstract = existing_row["abstract"] if existing_row else ""

        # title — preserve from existing row when not in input; same reasoning
        # as abstract: partial updates must not corrupt the stored title.
        has_title = "title" in paper
        if has_title:
            eff_title = paper["title"]
        else:
            eff_title = existing_row["title"] if existing_row else ""

        # keywords — preserve from existing row; keywords affect content_hash
        # and _build_search_text, so omitting them in a partial update must
        # not spuriously trigger content_changed.
        has_keywords = "keywords" in paper
        if has_keywords:
            keywords = paper["keywords"]
            keywords_json = json.dumps(keywords if isinstance(keywords, list) else [], ensure_ascii=False)
        else:
            keywords_json = existing_row["keywords_json"] if existing_row else "[]"

        # ------------------------------------------------------------------
        # Build effective_paper: used for all downstream indexing operations.
        # Missing list-field keys are filled in from the existing DB row
        # so that content_hash, FTS text, and embedding all reflect the
        # true stored state — not the partial input.
        # ------------------------------------------------------------------
        try:
            eff_authors = json.loads(authors_json)
        except Exception:
            eff_authors = []
        try:
            eff_tags = json.loads(tags_json)
        except Exception:
            eff_tags = []
        try:
            eff_categories = json.loads(categories_json)
        except Exception:
            eff_categories = []
        try:
            eff_summary = json.loads(summary_json)
        except Exception:
            eff_summary = {}
        try:
            eff_keywords = json.loads(keywords_json)
        except Exception:
            eff_keywords = []

        effective_paper = dict(paper)  # shallow copy — start with all scalar fields
        effective_paper["authors"] = eff_authors
        effective_paper["tags"] = eff_tags
        effective_paper["topic_tags"] = eff_tags
        effective_paper["categories"] = eff_categories
        effective_paper["summary"] = eff_summary
        effective_paper["abstract"] = eff_abstract
        effective_paper["title"] = eff_title
        effective_paper["keywords"] = eff_keywords
        # topic: derive from effective tags if not explicitly set
        topic = paper.get("topic", "")
        if not topic and eff_tags:
            topic = eff_tags[0]

        # ------------------------------------------------------------------
        # Content hash and change detection use the effective paper so that
        # partial updates (missing list fields) do not produce a different
        # hash and do not spuriously trigger re-indexing.
        # ------------------------------------------------------------------
        content_hash = _compute_content_hash(effective_paper)
        content_changed = not existing_row or existing_row["content_hash"] != content_hash

        external_ids = paper.get("external_ids", {})
        doi = external_ids.get("doi", paper.get("doi", ""))
        openalex_id = external_ids.get("openalex", "")
        crossref_id = external_ids.get("crossref", "")

        now = datetime.now(timezone.utc).isoformat()

        # Upsert into papers — all parsed values written
        conn.execute(
            """
            INSERT INTO papers (
                paper_id, source, title, authors_json, year, venue, publication_type,
                url, pdf_url, abstract, summary_json, topic, tags_json, categories_json,
                keywords_json, doi, openalex_id, crossref_id, content_hash, updated_at,
                graph_sync_status, graph_sync_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                source=excluded.source,
                title=excluded.title,
                authors_json=excluded.authors_json,
                year=excluded.year,
                venue=excluded.venue,
                publication_type=excluded.publication_type,
                url=excluded.url,
                pdf_url=excluded.pdf_url,
                abstract=excluded.abstract,
                summary_json=excluded.summary_json,
                topic=excluded.topic,
                tags_json=excluded.tags_json,
                categories_json=excluded.categories_json,
                keywords_json=excluded.keywords_json,
                doi=excluded.doi,
                openalex_id=excluded.openalex_id,
                crossref_id=excluded.crossref_id,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at,
                graph_sync_status=excluded.graph_sync_status,
                graph_sync_error=excluded.graph_sync_error
            """,
            (
                paper_id,
                paper.get("source", ""),
                eff_title,
                authors_json,
                paper.get("year", ""),
                paper.get("venue", ""),
                paper.get("publication_type", ""),
                paper.get("url", ""),
                paper.get("pdf_url", ""),
                eff_abstract,
                summary_json,
                topic,
                tags_json,
                categories_json,
                keywords_json,
                doi,
                openalex_id,
                crossref_id,
                content_hash,
                now,
                "ok",       # graph_sync_status
                "",         # graph_sync_error
            ),
        )

        # Only rebuild FTS/embeddings when text content changed.
        # All indexing operations use effective_paper so they reflect
        # the actual stored state, not the partial input.
        if content_changed:
            # Rebuild FTS entry — uses effective_paper for search text
            self._rebuild_fts_entry(conn, paper_id, effective_paper, summary_json, topic, tags_json, eff_summary)

            # Generate and store embedding using effective_paper
            if self._sqlite_vec_available and self.embedding_client.is_enabled():
                await self._upsert_embedding(conn, paper_id, effective_paper, content_hash, now)
            elif self._sqlite_vec_available and not self.embedding_client.is_enabled():
                # Embedding was previously available but now unavailable
                # Remove stale embeddings
                conn.execute(
                    "DELETE FROM paper_embeddings WHERE paper_id = ?", (paper_id,)
                )
                if self._sqlite_vec_available:
                    try:
                        conn.execute(
                            "DELETE FROM paper_vectors WHERE paper_id = ?", (paper_id,)
                        )
                    except Exception:
                        pass

        # Sync to knowledge graph — uses the original partial paper dict
        # so that missing graph-edge fields correctly trigger "preserve" semantics.
        graph_sync_status = "ok"
        graph_sync_error = ""
        try:
            self._ensure_graph_initialized()
            kg = KnowledgeGraph(self)
            kg.upsert_paper(paper, commit=False)
        except Exception as e:
            logger.warning(f"Failed to sync paper {paper_id} to knowledge graph: {e}")
            graph_sync_status = "failed"
            graph_sync_error = str(e)[:500]
            # Record failure in papers table so it is observable
            conn.execute(
                "UPDATE papers SET graph_sync_status = 'failed', graph_sync_error = ? WHERE paper_id = ?",
                (graph_sync_error, paper_id),
            )

        conn.commit()

        return {
            "content_changed": content_changed,
            "graph_sync_status": graph_sync_status,
            "graph_sync_error": graph_sync_error,
        }

    def _rebuild_fts_entry(
        self,
        conn: sqlite3.Connection,
        paper_id: str,
        paper: dict[str, Any],
        summary_json: str,
        topic: str,
        tags_json: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Rebuild FTS entry for a paper.

        Args:
            summary: pre-parsed summary dict. If None, parsed from summary_json.
                     Internal callers (upsert_paper) pass this to avoid re-parsing.
        """
        if summary is None:
            try:
                summary = json.loads(summary_json) if summary_json != "{}" else {}
            except Exception:
                summary = {}

        summary_text = " ".join([
            summary.get("one_sentence", ""),
            summary.get("problem", ""),
            summary.get("method", ""),
            summary.get("findings", ""),
        ])
        keywords = ",".join(paper.get("keywords", []))
        topic_tags = topic + " " + ",".join(json.loads(tags_json) if isinstance(tags_json, str) else tags_json)

        # Delete old FTS entry by paper_id
        conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,))

        # Insert new FTS entry directly
        conn.execute(
            "INSERT INTO papers_fts(paper_id, title, abstract, summary, keywords, topic_tags) VALUES (?, ?, ?, ?, ?, ?)",
            (paper_id, paper.get("title", ""), paper.get("abstract", ""),
             summary_text, keywords, topic_tags),
        )

    async def _upsert_embedding(
        self,
        conn: sqlite3.Connection,
        paper_id: str,
        paper: dict[str, Any],
        content_hash: str,
        now: str,
    ) -> None:
        """Generate and store embedding for a paper."""
        search_text = _build_search_text(paper)
        vector = await self.embedding_client.embed(search_text)

        if vector is None:
            logger.warning("Failed to generate embedding for paper {}", paper_id)
            return

        # Auto-detect dimension: if configured dimension is 0 or wrong, detect from first successful embedding
        actual_dim = len(vector)
        if self._vector_dim != actual_dim:
            logger.info(
                "Embedding dimension mismatch: got {} (configured {}), recreating paper_vectors table",
                actual_dim,
                self._vector_dim,
            )
            try:
                # Drop and recreate with correct dimension
                conn.execute("DROP TABLE IF EXISTS paper_vectors")
                conn.execute(f"""
                    CREATE VIRTUAL TABLE paper_vectors USING vec0(
                        paper_id TEXT,
                        embedding float[{actual_dim}]
                    )
                """)
                self._vector_dim = actual_dim
                conn.commit()
            except Exception as e:
                logger.warning("Failed to recreate paper_vectors table: {}", e)
                return

        # Store in paper_embeddings
        embedding_blob = vector_to_bytes(vector)
        conn.execute(
            """
            INSERT INTO paper_embeddings (paper_id, embedding_model, embedding_blob, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                embedding_model=excluded.embedding_model,
                embedding_blob=excluded.embedding_blob,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at
            """,
            (paper_id, self._config.embedding_model, embedding_blob, content_hash, now),
        )

        # Store in sqlite-vec
        if self._sqlite_vec_available:
            try:
                # Delete existing
                conn.execute("DELETE FROM paper_vectors WHERE paper_id = ?", (paper_id,))
                # Insert new
                conn.execute(
                    "INSERT INTO paper_vectors (paper_id, embedding) VALUES (?, ?)",
                    (paper_id, embedding_blob),
                )
            except Exception as e:
                logger.warning("Failed to store vector in sqlite-vec: {}", e)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        topic: str | None = None,
        tags: list[str] | None = None,
        year: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
        source: str | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Search papers using hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            topic: Filter by topic (substring match)
            tags: Filter by tags (any match)
            year: Filter by exact year
            year_from: Filter by year >= this value
            year_to: Filter by year <= this value
            categories: Filter by categories (any match)
            source: Filter by source (e.g. "arxiv")
            rerank: Whether to apply LLM reranking

        Returns:
            List of result dicts with paper_id, title, scores, and metadata
        """
        conn = self._get_conn()

        # Build filter conditions (applied in Python after FTS)
        results: list[dict[str, Any]] = []

        # 1. Keyword search (FTS5) - do full search first, filter in Python
        all_lexical_results = self._fts_search(conn, query)

        # Apply filters in Python
        for r in all_lexical_results:
            r["lexical_score"] = r.get("rank", 0)
            r["vector_score"] = 0.0
            r["matched_filters"] = self._get_matched_filters(conn, r["paper_id"])
            if self._apply_filters(r, topic, tags, year, year_from, year_to, categories, source):
                results.append(r)

        # 2. Vector search (if available and enabled)
        if self._sqlite_vec_available and self.embedding_client.is_enabled():
            vector_results = await self._vector_search(conn, query)
            for r in vector_results:
                if r["paper_id"] not in {x["paper_id"] for x in results}:
                    r["lexical_score"] = 0.0
                    r["vector_score"] = r.pop("score", 0)
                    r["matched_filters"] = self._get_matched_filters(conn, r["paper_id"])
                    if self._apply_filters(r, topic, tags, year, year_from, year_to, categories, source):
                        results.append(r)

        # 3. RRF fusion
        k = self._config.hybrid_search_rrf_k
        rrf_scores: dict[str, float] = {}
        for r in results:
            paper_id = r["paper_id"]
            lex_rank = r.get("lexical_rank", len(results) + 1)
            vec_rank = r.get("vector_rank", len(results) + 1)
            lex_score = r.get("lexical_score", 0)
            vec_score = r.get("vector_score", 0)

            # RRF fusion
            rrf = (0.0 if lex_rank == 0 else 1.0 / (k + lex_rank)) * self._config.lexical_weight + \
                  (0.0 if vec_rank == 0 else 1.0 / (k + vec_rank)) * self._config.vector_weight

            # Also add raw scores
            rrf += lex_score * self._config.lexical_weight + vec_score * self._config.vector_weight

            rrf_scores[paper_id] = rrf
            r["final_score"] = rrf

        # Sort by final score
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Limit to top_k
        results = results[:top_k]

        # 4. LLM rerank (top N)
        if rerank and self._config.enable_rerank and results:
            reranked = await self._rerank(query, results[: self._config.rerank_top_k])
            if reranked is not None:
                results = reranked

        return results

    def _fts_search(
        self,
        conn: sqlite3.Connection,
        query: str,
    ) -> list[dict[str, Any]]:
        """Execute FTS5 keyword search."""
        try:
            # Escape FTS5 special characters
            fts_query = query.replace('"', '""')

            sql = """
                SELECT
                    papers.paper_id,
                    papers.title,
                    papers.year,
                    papers.source,
                    bm25(papers_fts) as rank,
                    snippet(papers_fts, 1, '<mark>', '</mark>', '...', 30) as abstract_snippet,
                    highlight(papers_fts, 1, '<mark>', '</mark>') as title_highlight
                FROM papers_fts
                JOIN papers ON papers_fts.paper_id = papers.paper_id
                WHERE papers_fts MATCH ?
                ORDER BY rank
                LIMIT 50
            """
            cursor = conn.execute(sql, [fts_query])
            rows = cursor.fetchall()

            results = []
            for i, row in enumerate(rows):
                r = dict(row)
                r["lexical_rank"] = i + 1
                results.append(r)

            return results
        except Exception as e:
            logger.warning("FTS search failed: {}", e)
            return []

    async def _vector_search(
        self,
        conn: sqlite3.Connection,
        query: str,
    ) -> list[dict[str, Any]]:
        """Execute vector search using sqlite-vec."""
        if not self.embedding_client.is_enabled():
            return []

        vector = await self.embedding_client.embed(query)
        if vector is None:
            return []

        try:
            embedding_blob = vector_to_bytes(vector)

            sql = f"""
                SELECT
                    v.paper_id,
                    v.distance,
                    1.0 / (1.0 + v.distance) as score
                FROM paper_vectors v
                WHERE v.embedding MATCH ?
                ORDER BY distance
                LIMIT 50
            """
            cursor = conn.execute(sql, [embedding_blob])
            rows = cursor.fetchall()

            results = []
            for i, row in enumerate(rows):
                r = dict(row)
                r["vector_rank"] = i + 1
                results.append(r)

            # Batch fetch paper metadata to avoid N+1 queries
            if results:
                paper_ids = [str(r["paper_id"]) for r in results]
                placeholders = ",".join("?" * len(paper_ids))
                metadata_rows = conn.execute(
                    f"SELECT paper_id, title, year, source FROM papers WHERE paper_id IN ({placeholders})",
                    paper_ids,
                ).fetchall()
                metadata_map = {row["paper_id"]: row for row in metadata_rows}
                for r in results:
                    paper_id = r["paper_id"]
                    if paper_id in metadata_map:
                        row = metadata_map[paper_id]
                        r["title"] = row["title"]
                        r["year"] = row["year"]
                        r["source"] = row["source"]

            return results
        except Exception as e:
            logger.warning("Vector search failed: {}", e)
            return []

    async def _rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Rerank candidates using LLM.

        Returns reranked list or None on failure.
        """
        if not candidates:
            return None

        api_key = (self._config.embedding_api_key or "").strip()
        if not api_key:
            return None

        # Build rerank prompt
        papers_text = []
        for i, p in enumerate(candidates, 1):
            title = p.get("title", "Unknown")
            abstract = p.get("abstract", p.get("abstract_snippet", ""))
            year = p.get("year", "Unknown")
            source = p.get("source", "")
            papers_text.append(
                f"[{i}] {title} ({year}, {source})\n   Abstract: {abstract[:300]}..."
            )

        rerank_prompt = f"""You are a research paper relevance ranking expert. Given a query and a list of candidate papers, rank them by relevance to the query.

Query: {query}

Candidate Papers:
{chr(10).join(papers_text)}

Your task: Rank these papers from most relevant to least relevant for the given query.
Consider:
1. Title match with query intent
2. Abstract content relevance
3. Recency and importance

Output a JSON array of paper indices in ranking order, most relevant first:
[1, 3, 2, 5, ...]

Only output the JSON array, nothing else."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to use dashscope or configured provider
                api_base = self._config.embedding_api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"

                payload = {
                    "model": "qwen-plus",
                    "messages": [{"role": "user", "content": rerank_prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                # Try OpenAI-compatible endpoint first
                url = f"{api_base.rstrip('/')}/chat/completions"
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    return None

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Parse JSON ranking
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    ranking = json.loads(content[json_start:json_end])
                    # Reorder candidates
                    paper_map = {i + 1: p for i, p in enumerate(candidates)}
                    reranked = []
                    for rank, idx in enumerate(ranking, 1):
                        if idx in paper_map:
                            p = paper_map[idx]
                            p["rerank_score"] = len(ranking) - rank
                            reranked.append(p)
                    # Add any missing papers
                    seen = set(ranking)
                    for i, p in enumerate(candidates):
                        if i + 1 not in seen:
                            p["rerank_score"] = 0
                            reranked.append(p)
                    return reranked
        except Exception as e:
            logger.warning("Rerank failed: {}", e)

        return None

    def _get_matched_filters(
        self, conn: sqlite3.Connection, paper_id: str
    ) -> dict[str, Any]:
        """Get which filters matched for a paper."""
        row = conn.execute(
            "SELECT topic, tags_json, year, categories_json, source FROM papers WHERE paper_id = ?",
            (paper_id,)
        ).fetchone()

        if not row:
            return {}

        return {
            "topic": row["topic"],
            "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
            "year": row["year"],
            "categories": json.loads(row["categories_json"]) if row["categories_json"] else [],
            "source": row["source"],
        }

    def _apply_filters(
        self,
        r: dict[str, Any],
        topic: str | None,
        tags: list[str] | None,
        year: str | None,
        year_from: int | None,
        year_to: int | None,
        categories: list[str] | None,
        source: str | None,
    ) -> bool:
        """Apply filters to a search result.

        Returns True if the result passes all filters, False otherwise.
        """
        filters = r.get("matched_filters", {})

        # Topic filter (substring match, case-insensitive)
        if topic:
            paper_topic = filters.get("topic", "").lower()
            if topic.lower() not in paper_topic:
                return False

        # Tags filter (any match)
        if tags:
            paper_tags = [t.lower() for t in filters.get("tags", [])]
            if not any(tag.lower() in paper_tags for tag in tags):
                return False

        # Year filter (exact match)
        if year:
            if filters.get("year") != year:
                return False

        # Year range filter
        if year_from is not None or year_to is not None:
            try:
                paper_year = int(filters.get("year", 0))
            except (ValueError, TypeError):
                return False
            if year_from is not None and paper_year < year_from:
                return False
            if year_to is not None and paper_year > year_to:
                return False

        # Categories filter (any match)
        if categories:
            paper_cats = [c.lower() for c in filters.get("categories", [])]
            if not any(cat.lower() in paper_cats for cat in categories):
                return False

        # Source filter (exact match, case-insensitive)
        if source:
            if filters.get("source", "").lower() != source.lower():
                return False

        return True

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Get a paper by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()

        if not row:
            return None

        result = dict(row)
        # Parse JSON fields
        for field in ["authors_json", "tags_json", "categories_json", "summary_json"]:
            if field in result and result[field]:
                try:
                    result[field.replace("_json", "")] = json.loads(result[field])
                except Exception:
                    result[field.replace("_json", "")] = []
            if field in result:
                del result[field]

        return result

    def count(self) -> int:
        """Get total number of indexed papers."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM papers").fetchone()
        return row["cnt"] if row else 0

    def get_paper_graph_sync_status(self, paper_id: str) -> dict[str, Any]:
        """Return graph sync status for a paper: {status: 'ok'|'failed'|'', error: str}."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT graph_sync_status, graph_sync_error FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if not row:
            return {"status": "", "error": "paper not found"}
        return {"status": row["graph_sync_status"], "error": row["graph_sync_error"]}

    def list_graph_sync_failures(self, limit: int = 100) -> list[dict[str, Any]]:
        """List papers whose graph sync failed, ordered by most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT paper_id, graph_sync_status, graph_sync_error, updated_at
            FROM papers
            WHERE graph_sync_status = 'failed'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Shared graph-rebuild helper (used by both CLI and agent tools)
# ---------------------------------------------------------------------------

_EMPTY_KG_STATS: dict[str, int] = {
    "concepts": 0,
    "authors": 0,
    "citation_edges": 0,
    "paper_concept_edges": 0,
    "author_collaborations": 0,
    "paper_related_edges": 0,
}


async def rebuild_graph_from_workspace(
    workspace: Path,
    semantic_config: SemanticSearchConfig,
) -> dict[str, Any]:
    """Rebuild the knowledge graph from all papers in a workspace.

    Args:
        workspace: Path to the workspace directory.
        semantic_config: Semantic search configuration.

    Returns:
        A dict with keys: papers_dir_found (bool), total (int),
        count (int), stats (dict).
    """
    import json as _json

    literature_dir = workspace / "literature"
    papers_dir = literature_dir / "papers"

    result: dict[str, Any] = {
        "papers_dir_found": papers_dir.exists(),
        "total": 0,
        "count": 0,
        "stats": {},
    }

    if not result["papers_dir_found"]:
        return result

    # Load all papers
    papers: list[dict[str, Any]] = []
    for json_file in papers_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                papers.append(_json.load(f))
        except Exception:
            pass

    result["total"] = len(papers)

    if not papers:
        result["stats"] = _EMPTY_KG_STATS.copy()
        return result

    # Initialize search index and graph
    db_path = workspace / semantic_config.sqlite_db_path
    search_index = SearchIndex(db_path, semantic_config)
    await search_index.initialize()
    kg = search_index.get_graph()

    result["count"] = kg.rebuild_from_papers(papers)
    result["stats"] = kg.stats()
    search_index.close()

    return result
