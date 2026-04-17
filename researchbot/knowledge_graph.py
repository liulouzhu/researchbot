"""Knowledge Graph for ResearchBot.

Stores paper relationships as a graph in SQLite (same database as SearchIndex):
- citations: paper -> paper (referencing)
- concepts: paper -> concept nodes
- authors: author nodes with collaboration edges
- related_works: paper -> paper (related by OpenAlex)

Schema uses pure edge tables — no external graph DB required.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def _normalize_author_id(name: str) -> str:
    """Normalize an author name into a stable, unique ID.

    Format: FamilyName_FirstInitial[s]. e.g. "Vaswani, Ashish" -> "vaswani_a"
    Falls back to SHA256 of lowercased name if parsing fails.
    """
    if not name:
        return ""

    # Try "Family, Given" format
    if "," in name:
        parts = name.split(",", 1)
        family = parts[0].strip().lower()
        given = parts[1].strip() if len(parts) > 1 else ""
        initial = given[0].lower() if given else ""
        if family:
            return f"{family}_{initial}" if initial else f"{family}"

    # Try "Given Family" format (no comma)
    parts = name.strip().split()
    if len(parts) >= 2:
        family = parts[-1].lower()
        initial = parts[0][0].lower() if parts[0] else ""
        return f"{family}_{initial}" if initial else family

    # Fallback: hash of lowercased name
    return hashlib.sha256(name.lower().encode()).hexdigest()[:16]


def _safe_id(s: str) -> str:
    """Make a string safe for use as a graph node ID."""
    if not s:
        return ""
    # Replace common separators with underscores
    s = re.sub(r"[\s\-/.]", "_", s)
    # Remove anything that's not alphanumeric or underscore
    s = re.sub(r"[^\w]", "", s)
    return s.lower()[:64]


class KnowledgeGraph:
    """Knowledge graph backed by SQLite (same file as SearchIndex).

    Connection strategy:
    - **Own connection**: When constructed with a `db_path` (str/Path), KG
      manages its own connection with WAL/busy_timeout. Used for read-only
      queries and bulk rebuild operations.
    - **Borrowed connection**: When constructed with a `conn` (sqlite3.Connection),
      KG writes to that connection so the caller controls the transaction.
      Used by SearchIndex.upsert_paper to keep paper+graph writes atomic.

    Usage:
        # Option 1: own connection (for reads, rebuild)
        kg = KnowledgeGraph(db_path="/path/to/search.sqlite3")
        citing = kg.get_citing_papers("paper_id")
        kg.close()

        # Option 2: borrowed connection (for atomic upsert with SearchIndex)
        kg = KnowledgeGraph(conn=search_index._get_conn())
        kg.upsert_paper(paper, commit=False)
        # caller commits
    """

    def __init__(
        self,
        search_index_or_db: "SearchIndex | str | Path | None" = None,  # noqa: F821
        *,
        conn: sqlite3.Connection | None = None,
        db_path: str | Path | None = None,
    ):
        self._db_path: Path | None = None
        self._local_conn: sqlite3.Connection | None = None
        self._owns_connection: bool = False

        # Priority 1: explicit conn (borrowed)
        if conn is not None:
            self._local_conn = conn
            self._owns_connection = False
            return

        # Priority 2: explicit db_path
        if db_path is not None:
            self._db_path = Path(db_path)
            self._owns_connection = True
            return

        # Priority 3: legacy positional arg (SearchIndex or str/Path)
        if search_index_or_db is None:
            raise ValueError("KnowledgeGraph requires conn, db_path, or search_index_or_db")

        if isinstance(search_index_or_db, (str, Path)):
            self._db_path = Path(search_index_or_db)
            self._owns_connection = True
        else:
            # SearchIndex instance — use its own connection for atomic writes
            si = search_index_or_db
            self._db_path = si._db_path
            self._local_conn = si._get_conn()
            self._owns_connection = False

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create this KnowledgeGraph's SQLite connection."""
        if self._local_conn is not None:
            return self._local_conn
        assert self._db_path is not None, "KnowledgeGraph not initialized with a db_path or conn"
        # Import here to avoid circular import
        from researchbot.search_index import create_sqlite_connection

        self._local_conn = create_sqlite_connection(self._db_path)
        return self._local_conn

    # Backward-compat alias — external code (e.g. concept_explore.py) calls kg._conn()
    _conn = _get_conn

    def close(self) -> None:
        """Close this KnowledgeGraph's connection if it owns it.

        Safe to call multiple times. Does NOT close borrowed connections.
        """
        if self._owns_connection and self._local_conn is not None:
            try:
                self._local_conn.close()
            except Exception:
                pass
            self._local_conn = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """Create graph tables if they don't exist.

        Called automatically by SearchIndex.initialize() — you don't need to
        call this separately unless you only have a KnowledgeGraph instance.
        """
        conn = self._get_conn()

        # Concept nodes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                level INTEGER DEFAULT 0,
                wikipedia_url TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)

        # Author nodes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                author_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)

        # Paper -> Paper citation edges (citing -> cited)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS citation_edges (
                citing_paper_id TEXT NOT NULL,
                cited_paper_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (citing_paper_id, cited_paper_id)
            )
        """)

        # Paper -> Concept edges
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_concepts (
                paper_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                score REAL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'openalex',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (paper_id, concept_id)
            )
        """)

        # Author collaboration edges (co-authorship)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS author_collaborations (
                author_id_1 TEXT NOT NULL,
                author_id_2 TEXT NOT NULL,
                paper_count INTEGER DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (author_id_1, author_id_2)
            )
        """)

        # Tracks which paper contributed to a collaboration (for idempotent counting)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS author_collaboration_papers (
                author_id_1 TEXT NOT NULL,
                author_id_2 TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                PRIMARY KEY (author_id_1, author_id_2, paper_id)
            )
        """)

        # Paper <-> Paper related edges (from OpenAlex related_works)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_related (
                paper_id_1 TEXT NOT NULL,
                paper_id_2 TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'openalex',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (paper_id_1, paper_id_2)
            )
        """)

        # Indexes for fast traversal
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_citing ON citation_edges(citing_paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_cited ON citation_edges(cited_paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_concepts_paper ON paper_concepts(paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_concepts_concept ON paper_concepts(concept_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collaborations_a1 ON author_collaborations(author_id_1)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collaborations_a2 ON author_collaborations(author_id_2)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_related_p1 ON paper_related(paper_id_1)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collab_papers_a1 ON author_collaboration_papers(author_id_1)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collab_papers_a2 ON author_collaboration_papers(author_id_2)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collab_papers_pid ON author_collaboration_papers(paper_id)"
        )

        conn.commit()

    # -------------------------------------------------------------------------
    # Paper-level upsert (writes all edges for one paper at once)
    # -------------------------------------------------------------------------

    def upsert_paper(self, paper: dict[str, Any], *, commit: bool = True) -> None:
        """Write all graph edges for a paper.

        Field-by-field update semantics (per-field replace semantics):
        - key absent from paper dict  → preserve existing edges (no-op for that field)
        - key present as [] (empty)   → clear all existing edges for that field
        - key present as [x, ...]     → replace existing edges with new list

        This ensures partial paper objects (e.g. from paper_save, paper_enrich)
        do not accidentally wipe edges that were loaded from a different source.

        Storage model:
        - citations:     unidirectional (citing → cited); traversal methods
                         support both outbound and inbound queries
        - concepts:      unidirectional (paper → concept)
        - collaborations: unidirectional (author pair); paper contributions
                          are tracked in author_collaboration_papers for
                          correct, idempotent paper_count
        - related_works: unidirectional in storage (paper → related);
                         traversal is bidirectional via UNION queries

        Each edge table uses a PRIMARY KEY that enforces uniqueness, so
        duplicate inserts are impossible regardless of update semantics.
        """
        paper_id = paper.get("paper_id", "")
        if not paper_id:
            return

        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Citations — only replace if referenced_works key is explicitly present
        if "referenced_works" in paper:
            self._upsert_citations(conn, paper_id, paper["referenced_works"], now)

        # Concepts — only replace if concepts key is explicitly present
        if "concepts" in paper:
            self._upsert_concepts(conn, paper_id, paper["concepts"], now)

        # Authors + collaborations — only replace if authors key is explicitly present
        if "authors" in paper:
            self._upsert_authors(conn, paper_id, paper["authors"], now)

        # Related works — only replace if related_works key is explicitly present
        if "related_works" in paper:
            self._upsert_related_works(conn, paper_id, paper["related_works"], now)

        if commit:
            conn.commit()

    def _upsert_citations(
        self, conn: sqlite3.Connection, paper_id: str, references: list[str], now: str
    ) -> None:
        """Replace citation edges for paper_id (delete old, insert new).

        Always deletes existing edges for this paper first, then inserts.
        Empty references list is valid — it clears all old citation edges.
        Duplicate inserts are prevented by PRIMARY KEY (citing, cited).
        """
        # Remove stale citation edges for this paper
        conn.execute(
            "DELETE FROM citation_edges WHERE citing_paper_id = ?",
            (paper_id,),
        )
        for ref_id in references:
            if not ref_id or ref_id == paper_id:
                continue
            conn.execute(
                """
                INSERT INTO citation_edges (citing_paper_id, cited_paper_id, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(citing_paper_id, cited_paper_id) DO UPDATE SET
                    source=excluded.source, updated_at=excluded.updated_at
                """,
                (paper_id, str(ref_id).strip(), "crossref/openalex", now),
            )

    def _upsert_concepts(
        self, conn: sqlite3.Connection, paper_id: str, concepts: list[Any], now: str
    ) -> None:
        """Replace paper->concept edges for paper_id (delete old, insert new).

        concepts can be a list of strings or dicts (OpenAlex format).
        """
        # Remove stale paper->concept edges (concept nodes themselves are kept)
        conn.execute(
            "DELETE FROM paper_concepts WHERE paper_id = ?",
            (paper_id,),
        )
        for c in concepts:
            if isinstance(c, dict):
                concept_id = str(c.get("id", c.get("display_name", ""))).strip()
                display_name = c.get("display_name", concept_id)
                level = c.get("level", 0)
                wikipedia_url = c.get("wikipedia_url", "") or ""
            else:
                concept_id = str(c).strip()
                display_name = concept_id
                level = 0
                wikipedia_url = ""

            if not concept_id:
                continue

            safe_id = _safe_id(concept_id)

            # Upsert concept node
            conn.execute(
                """
                INSERT INTO concepts (concept_id, display_name, level, wikipedia_url, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(concept_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    level=excluded.level,
                    wikipedia_url=excluded.wikipedia_url,
                    updated_at=excluded.updated_at
                """,
                (safe_id, display_name, level, wikipedia_url, now),
            )

            # Upsert paper->concept edge
            conn.execute(
                """
                INSERT INTO paper_concepts (paper_id, concept_id, score, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, concept_id) DO UPDATE SET
                    score=excluded.score, source=excluded.source, updated_at=excluded.updated_at
                """,
                (paper_id, safe_id, 1.0, "openalex", now),
            )

    def _upsert_authors(
        self, conn: sqlite3.Connection, paper_id: str, authors: list[Any], now: str
    ) -> None:
        """Insert author nodes and collaboration edges with replace semantics."""
        # Normalize all author names first
        normalized: list[tuple[str, str]] = []  # (author_id, display_name)
        for a in authors:
            name = str(a) if not isinstance(a, dict) else a.get("display_name", a.get("name", ""))
            if not name:
                continue
            author_id = _normalize_author_id(name)
            normalized.append((author_id, name))

        # Replace semantics: remove old collaboration contributions for this paper
        # Bulk decrement paper_count for all old pairs at once (avoids N+1)
        conn.execute(
            """
            UPDATE author_collaborations
            SET paper_count = MAX(0, paper_count - 1), updated_at = ?
            WHERE (author_id_1, author_id_2) IN (
                SELECT author_id_1, author_id_2 FROM author_collaboration_papers WHERE paper_id = ?
            )
            """,
            (now, paper_id),
        )
        # Delete old collaboration paper records for this paper
        conn.execute("DELETE FROM author_collaboration_papers WHERE paper_id = ?", (paper_id,))
        # Clean up zero-count collaboration rows left behind
        conn.execute("DELETE FROM author_collaborations WHERE paper_count <= 0")

        # Insert author nodes and build new collaboration edges
        for author_id, display_name in normalized:
            conn.execute(
                """
                INSERT INTO authors (author_id, display_name, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(author_id) DO UPDATE SET
                    display_name=excluded.display_name, updated_at=excluded.updated_at
                """,
                (author_id, display_name, now),
            )

        # Pairwise collaboration edges (re-insert with current author set)
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                aid1, _ = normalized[i]
                aid2, _ = normalized[j]
                # Use canonical ordering for the edge key
                if aid1 > aid2:
                    aid1, aid2 = aid2, aid1

                # Insert collaboration paper record
                conn.execute(
                    "INSERT INTO author_collaboration_papers (author_id_1, author_id_2, paper_id) VALUES (?, ?, ?)",
                    (aid1, aid2, paper_id),
                )
                # Upsert collaboration edge (paper_count is now correct from prior decrement + insert)
                conn.execute(
                    """
                    INSERT INTO author_collaborations (author_id_1, author_id_2, paper_count, updated_at)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(author_id_1, author_id_2) DO UPDATE SET
                        paper_count=author_collaborations.paper_count + 1,
                        updated_at=excluded.updated_at
                    """,
                    (aid1, aid2, now),
                )

    def _upsert_related_works(
        self, conn: sqlite3.Connection, paper_id: str, related: list[str], now: str
    ) -> None:
        """Replace paper<->paper related edges for paper_id (delete old, insert new).

        Always deletes existing edges for this paper first, then inserts.
        Empty related_works list is valid — it clears all old related edges.
        Storage is unidirectional (paper_id_1 → paper_id_2); traversal
        in get_related_papers uses UNION to traverse bidirectionally.
        Duplicate inserts are prevented by PRIMARY KEY (paper_id_1, paper_id_2).
        """
        # Remove stale related-work edges for this paper
        conn.execute(
            "DELETE FROM paper_related WHERE paper_id_1 = ?",
            (paper_id,),
        )
        for rel_id in related:
            if not rel_id or rel_id == paper_id:
                continue
            conn.execute(
                """
                INSERT INTO paper_related (paper_id_1, paper_id_2, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id_1, paper_id_2) DO UPDATE SET
                    source=excluded.source, updated_at=excluded.updated_at
                """,
                (paper_id, str(rel_id).strip(), "openalex", now),
            )

    # -------------------------------------------------------------------------
    # Graph traversal queries
    # -------------------------------------------------------------------------

    def get_citing_papers(self, paper_id: str, depth: int = 1) -> list[str]:
        """Get paper IDs that cite the given paper (inbound citations).

        Args:
            paper_id: The paper to look up
            depth: How many hops outward (depth=1 means direct citers only)

        Returns:
            List of paper IDs that cite this paper
        """
        if depth <= 0:
            return []
        conn = self._get_conn()

        if depth == 1:
            rows = conn.execute(
                "SELECT citing_paper_id FROM citation_edges WHERE cited_paper_id = ?",
                (paper_id,),
            ).fetchall()
            return [r["citing_paper_id"] for r in rows]

        # Multi-hop: use iterative expansion
        visited: set[str] = {paper_id}
        frontier = {paper_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            placeholders = ",".join("?" * len(frontier))
            rows = conn.execute(
                f"SELECT citing_paper_id FROM citation_edges WHERE cited_paper_id IN ({placeholders}) AND citing_paper_id NOT IN ({placeholders})",
                list(frontier) + list(frontier),
            ).fetchall()
            for r in rows:
                pid = r["citing_paper_id"]
                if pid not in visited:
                    visited.add(pid)
                    next_frontier.add(pid)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(paper_id)
        return list(visited)

    def get_cited_papers(self, paper_id: str, depth: int = 1) -> list[str]:
        """Get papers cited by the given paper (outbound citations).

        Args:
            paper_id: The paper to look up
            depth: How many hops outward

        Returns:
            List of paper IDs cited by this paper
        """
        if depth <= 0:
            return []
        conn = self._get_conn()

        if depth == 1:
            rows = conn.execute(
                "SELECT cited_paper_id FROM citation_edges WHERE citing_paper_id = ?",
                (paper_id,),
            ).fetchall()
            return [r["cited_paper_id"] for r in rows]

        visited: set[str] = {paper_id}
        frontier = {paper_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            placeholders = ",".join("?" * len(frontier))
            rows = conn.execute(
                f"SELECT cited_paper_id FROM citation_edges WHERE citing_paper_id IN ({placeholders}) AND cited_paper_id NOT IN ({placeholders})",
                list(frontier) + list(frontier),
            ).fetchall()
            for r in rows:
                pid = r["cited_paper_id"]
                if pid not in visited:
                    visited.add(pid)
                    next_frontier.add(pid)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(paper_id)
        return list(visited)

    def get_papers_by_concept(self, concept_id: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Get paper IDs associated with a concept, ordered by score.

        Returns:
            List of (paper_id, score) tuples
        """
        conn = self._get_conn()
        safe_id = _safe_id(concept_id)
        rows = conn.execute(
            """
            SELECT paper_id, score FROM paper_concepts
            WHERE concept_id = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (safe_id, top_k),
        ).fetchall()
        return [(r["paper_id"], r["score"]) for r in rows]

    def get_co_authors(self, author_id: str, top_k: int = 20) -> list[tuple[str, int]]:
        """Get co-authors of an author, ordered by collaboration frequency.

        Returns:
            List of (author_id, paper_count) tuples
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                CASE WHEN author_id_1 = ? THEN author_id_2 ELSE author_id_1 END AS co_author,
                paper_count
            FROM author_collaborations
            WHERE author_id_1 = ? OR author_id_2 = ?
            ORDER BY paper_count DESC
            LIMIT ?
            """,
            (author_id, author_id, author_id, top_k),
        ).fetchall()
        return [(r["co_author"], r["paper_count"]) for r in rows]

    def get_concept_neighbors(self, concept_id: str, top_k: int = 20) -> list[tuple[str, int]]:
        """Get other concepts that co-occur with this concept in papers.

        Returns:
            List of (concept_id, co_occurrence_count) tuples
        """
        conn = self._get_conn()
        safe_id = _safe_id(concept_id)
        rows = conn.execute(
            """
            SELECT pc2.concept_id, COUNT(*) AS cnt
            FROM paper_concepts pc1
            JOIN paper_concepts pc2 ON pc1.paper_id = pc2.paper_id
            WHERE pc1.concept_id = ? AND pc2.concept_id != ?
            GROUP BY pc2.concept_id
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (safe_id, safe_id, top_k),
        ).fetchall()
        return [(r["concept_id"], r["cnt"]) for r in rows]

    def get_related_papers(self, paper_id: str, depth: int = 2) -> list[str]:
        """Get papers related to the given paper via related_works edges.

        Args:
            paper_id: The paper to look up
            depth: How many hops (depth=1 = direct neighbors only)

        Returns:
            List of related paper IDs
        """
        if depth <= 0:
            return []
        conn = self._get_conn()
        visited: set[str] = {paper_id}
        frontier: set[str] = {paper_id}

        for _ in range(depth):
            if not frontier:
                break
            # Build parameter lists: [frontier..., visited...] for each SELECT
            frontier_list = list(frontier)
            visited_list = list(visited)
            front_ph = ",".join("?" * len(frontier_list))
            visited_ph = ",".join("?" * len(visited_list))

            # SELECT 1: paper_2 where paper_1 is in frontier and paper_2 is NOT visited
            # SELECT 2: paper_1 where paper_2 is in frontier and paper_1 is NOT visited
            rows = conn.execute(
                f"""
                SELECT paper_id_2 AS related_id FROM paper_related
                WHERE paper_id_1 IN ({front_ph}) AND paper_id_2 NOT IN ({visited_ph})
                UNION
                SELECT paper_id_1 AS related_id FROM paper_related
                WHERE paper_id_2 IN ({front_ph}) AND paper_id_1 NOT IN ({visited_ph})
                """,
                frontier_list + visited_list + frontier_list + visited_list,
            ).fetchall()
            next_frontier: set[str] = set()
            for r in rows:
                pid = r["related_id"]
                if pid not in visited:
                    visited.add(pid)
                    next_frontier.add(pid)
            frontier = next_frontier

        visited.discard(paper_id)
        return list(visited)

    def find_common_citations(self, paper_ids: list[str]) -> list[tuple[str, int]]:
        """Find papers cited by all of the given papers (co-citation analysis).

        Returns:
            List of (paper_id, count) — count is always len(paper_ids) for common
        """
        if not paper_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"""
            SELECT cited_paper_id, COUNT(DISTINCT citing_paper_id) AS cnt
            FROM citation_edges
            WHERE citing_paper_id IN ({placeholders})
            GROUP BY cited_paper_id
            HAVING cnt = ?
            """,
            paper_ids + [len(paper_ids)],
        ).fetchall()
        return [(r["cited_paper_id"], r["cnt"]) for r in rows]

    def find_path(self, paper_a: str, paper_b: str, max_depth: int = 3) -> list[list[str]] | None:
        """Find citation paths between two papers (BFS).

        Returns:
            List of paths, each path is a list of paper IDs from paper_a to paper_b.
            Returns None if no path found within max_depth.
        """
        if max_depth <= 0 or paper_a == paper_b:
            return None

        conn = self._get_conn()

        # BFS from paper_a to paper_b
        if paper_a == paper_b:
            return [[paper_a]]

        visited: set[str] = {paper_a}
        frontier: list[list[str]] = [[paper_a]]

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: list[list[str]] = []
            for path in frontier:
                current = path[-1]
                # Get both outbound (cited) and inbound (citing) neighbors
                placeholders = "?"
                cited_rows = conn.execute(
                    "SELECT cited_paper_id FROM citation_edges WHERE citing_paper_id = ?",
                    (current,),
                ).fetchall()
                citing_rows = conn.execute(
                    "SELECT citing_paper_id FROM citation_edges WHERE cited_paper_id = ?",
                    (current,),
                ).fetchall()

                neighbors = [r["cited_paper_id"] for r in cited_rows] + [
                    r["citing_paper_id"] for r in citing_rows
                ]

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    new_path = path + [neighbor]
                    if neighbor == paper_b:
                        return [new_path]
                    visited.add(neighbor)
                    next_frontier.append(new_path)
            frontier = next_frontier

        return None  # No path within max_depth

    def get_paper_concepts(self, paper_id: str) -> list[str]:
        """Get all concept IDs associated with a paper."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT concept_id FROM paper_concepts WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        return [r["concept_id"] for r in rows]

    def get_paper_authors(self, paper_id: str) -> list[str]:
        """Get author IDs for all authors of a paper."""
        # Note: paper->author relationship is not explicitly stored;
        # this requires looking at the papers table's authors_json
        conn = self._get_conn()
        row = conn.execute(
            "SELECT authors_json FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row:
            return []
        import json

        try:
            authors = json.loads(row["authors_json"])
            return [_normalize_author_id(a) for a in authors]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Co-citation analysis and recommendation
    # -------------------------------------------------------------------------

    def find_cocitation_candidates(
        self,
        paper_ids: list[str],
        min_count: int = 1,
    ) -> list[tuple[str, int, int]]:
        """Find papers cited by at least min_count of the given papers.

        Unlike find_common_citations (intersection), this returns candidates
        with varying coverage levels, enabling threshold-based ranking.

        Args:
            paper_ids: List of paper IDs (the "collection")
            min_count: Minimum number of input papers that must cite a candidate
                       (1 = all cited papers, 2 = cited by at least 2, etc.)

        Returns:
            List of (paper_id, matched_count, total_citation_count) tuples,
            sorted by matched_count descending, then total_citation_count descending.
            - matched_count: how many of the input papers cite this candidate
            - total_citation_count: total citations across the entire graph
        """
        if not paper_ids or min_count < 1:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(paper_ids))

        # Get candidates with their coverage among input papers and global citation count
        rows = conn.execute(
            f"""
            SELECT
                ce.cited_paper_id AS paper_id,
                COUNT(DISTINCT ce.citing_paper_id) AS matched_count
            FROM citation_edges ce
            WHERE ce.citing_paper_id IN ({placeholders})
            GROUP BY ce.cited_paper_id
            HAVING matched_count >= ?
            ORDER BY matched_count DESC
            """,
            paper_ids + [min_count],
        ).fetchall()
        candidate_ids = [r["paper_id"] for r in rows]

        # Batch-fetch total citation counts for all candidates (avoids correlated subquery)
        if candidate_ids:
            c_placeholders = ",".join("?" * len(candidate_ids))
            total_rows = conn.execute(
                f"SELECT cited_paper_id, COUNT(*) AS total_citations FROM citation_edges WHERE cited_paper_id IN ({c_placeholders}) GROUP BY cited_paper_id",
                candidate_ids,
            ).fetchall()
            total_map = {r["cited_paper_id"]: r["total_citations"] for r in total_rows}
        else:
            total_map = {}

        result = []
        for r in rows:
            pid = r["paper_id"]
            matched = r["matched_count"]
            total = total_map.get(pid, 0)
            result.append((pid, matched, total))

        # Sort by matched_count desc, total_citations desc
        result.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return result

    def recommend_cocited_papers(
        self,
        paper_ids: list[str],
        *,
        min_count: int = 1,
        concept_id: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        top_k: int = 20,
        use_time_decay: bool = False,
        current_year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend foundational papers based on co-citation analysis.

        Given a set of papers (e.g., a user's collection), finds papers that
        are frequently cited together by these papers — suggesting foundational
        influence on the collection.

        Scoring:
            score = matched_count * coverage_boost * concept_boost * time_decay
            - matched_count: how many collection papers cite this candidate
            - coverage_boost: 1 + matched_count / len(paper_ids)  (favor high coverage)
            - concept_boost: 2.0 if concept matches, else 1.0
            - time_decay: pow(0.95, age_in_years) if enabled, else 1.0

        Args:
            paper_ids: List of paper IDs in the collection
            min_count: Minimum number of collection papers that must cite a candidate
            concept_id: If set, boost papers that have this concept
            year_from: If set, only include papers published >= year_from
            year_to: If set, only include papers published <= year_to
            top_k: Maximum number of results to return
            use_time_decay: If True, apply exponential time decay (favor recent papers)
            current_year: Year to use for time decay calculation (default: current year)

        Returns:
            List of recommendation dicts with keys:
                paper_id, title, year, citation_count, matched_count,
                matched_paper_ids, matched_concepts, score, explanation
        """
        if not paper_ids:
            return []

        if current_year is None:
            from datetime import datetime

            current_year = datetime.now().year

        # Get candidates
        candidates = self.find_cocitation_candidates(paper_ids, min_count=min_count)
        if not candidates:
            return []

        candidate_ids = [pid for pid, _, _ in candidates]

        # Load metadata for all candidates in one query
        conn = self._get_conn()
        placeholders = ",".join("?" * len(candidate_ids))

        # Get citation_count from citation_edges (total inbound citations)
        cite_count_rows = conn.execute(
            f"""
            SELECT cited_paper_id, COUNT(*) AS citation_count
            FROM citation_edges
            WHERE cited_paper_id IN ({placeholders})
            GROUP BY cited_paper_id
            """,
            candidate_ids,
        ).fetchall()
        cite_count_map = {r["cited_paper_id"]: r["citation_count"] for r in cite_count_rows}

        meta_rows = conn.execute(
            f"""
            SELECT p.paper_id, p.title, p.year,
                   GROUP_CONCAT(pc.concept_id) AS concept_ids
            FROM papers p
            LEFT JOIN paper_concepts pc ON p.paper_id = pc.paper_id
            WHERE p.paper_id IN ({placeholders})
            GROUP BY p.paper_id
            """,
            candidate_ids,
        ).fetchall()

        meta_map: dict[str, dict[str, Any]] = {}
        for r in meta_rows:
            concept_ids = []
            if r["concept_ids"]:
                concept_ids = [c.strip() for c in r["concept_ids"].split(",") if c.strip()]
            meta_map[r["paper_id"]] = {
                "title": r["title"] or r["paper_id"],
                "year": r["year"] or "",
                "citation_count": cite_count_map.get(r["paper_id"], 0),
                "concept_ids": concept_ids,
            }

        # For candidates without metadata, create minimal entries
        for pid, _, total_citations in candidates:
            if pid not in meta_map:
                meta_map[pid] = {
                    "title": pid,
                    "year": "",
                    "citation_count": total_citations,
                    "concept_ids": [],
                }

        # Get which input papers cite each candidate
        cited_placeholders = ",".join("?" * len(candidate_ids))
        citing_placeholders = ",".join("?" * len(paper_ids))
        citing_rows = conn.execute(
            f"""
            SELECT cited_paper_id, citing_paper_id
            FROM citation_edges
            WHERE cited_paper_id IN ({cited_placeholders})
              AND citing_paper_id IN ({citing_placeholders})
            """,
            candidate_ids + paper_ids,
        ).fetchall()

        # Build citing_papers map: candidate_id -> list of citing input paper_ids
        citing_map: dict[str, set[str]] = {pid: set() for pid in candidate_ids}
        for r in citing_rows:
            citing_map[r["cited_paper_id"]].add(r["citing_paper_id"])

        # Normalize concept_id for comparison
        safe_concept = _safe_id(concept_id) if concept_id else None

        # Batch-fetch all concept display names for all candidates (avoids N+1 loop queries)
        all_concept_ids: set[str] = set()
        for pid, _, _ in candidates:
            all_concept_ids.update(meta_map.get(pid, {}).get("concept_ids", []))
        if all_concept_ids:
            c_ph = ",".join("?" * len(all_concept_ids))
            concept_rows = conn.execute(
                f"SELECT concept_id, display_name FROM concepts WHERE concept_id IN ({c_ph})",
                list(all_concept_ids),
            ).fetchall()
            concept_map: dict[str, str] = {r["concept_id"]: r["display_name"] for r in concept_rows}
        else:
            concept_map = {}

        # Score and filter
        results: list[dict[str, Any]] = []
        collection_size = len(paper_ids)

        for paper_id, matched_count, total_citations in candidates:
            meta = meta_map.get(paper_id, {})

            # Year filter
            try:
                year = int(meta.get("year", 0)) if meta.get("year") else 0
            except (ValueError, TypeError):
                year = 0

            if year_from is not None and year < year_from:
                continue
            if year_to is not None and year > year_to:
                continue

            # Compute score components
            coverage_ratio = matched_count / collection_size
            coverage_boost = 1.0 + coverage_ratio

            concept_ids = meta.get("concept_ids", [])
            concept_boost = 2.0 if (safe_concept and safe_concept in concept_ids) else 1.0

            if use_time_decay and year > 0:
                age = current_year - year
                time_decay = pow(0.95, max(0, age))
            else:
                time_decay = 1.0

            score = matched_count * coverage_boost * concept_boost * time_decay

            # Build explanation
            matched_list = sorted(citing_map.get(paper_id, set()))
            if matched_count == collection_size:
                explanation = (
                    f"Foundational paper cited by all {matched_count} collection papers. "
                    f"Citations in collection: {matched_count}, total citations: {total_citations}."
                )
            else:
                explanation = (
                    f"Cited by {matched_count} of {collection_size} collection papers "
                    f"({coverage_ratio:.0%} coverage). Total citations: {total_citations}."
                )

            if safe_concept and safe_concept in concept_ids:
                explanation += f" Matches concept filter."
            if use_time_decay and year > 0:
                explanation += f" Time decay applied (age={current_year - year} years)."

            matched_concepts = [
                {"concept_id": cid, "display_name": concept_map.get(cid, cid)}
                for cid in concept_ids
            ]

            results.append(
                {
                    "paper_id": paper_id,
                    "title": meta.get("title", paper_id),
                    "year": meta.get("year", ""),
                    "citation_count": meta.get("citation_count", total_citations),
                    "matched_count": matched_count,
                    "matched_paper_ids": matched_list,
                    "matched_concepts": matched_concepts,
                    "score": round(score, 4),
                    "explanation": explanation,
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------

    def rebuild_from_papers(self, papers: list[dict[str, Any]], *, batch_size: int = 50) -> int:
        """Bulk import papers into the graph.

        Transaction strategy:
        - **Own connection** (`db_path=` or str/Path): manages its own
          `BEGIN IMMEDIATE` → batch upserts → `COMMIT` / `ROLLBACK`.
        - **Borrowed connection** (`conn=` or `KnowledgeGraph(search_index)`):
          does NOT issue BEGIN/COMMIT — the caller controls the transaction.
          This prevents "cannot start a transaction within a transaction" errors.

        Args:
            papers: List of paper dicts to import
            batch_size: Number of papers per transaction (default 50).
                        Only effective when using own connection.

        Returns:
            Number of papers successfully imported
        """
        count = 0
        conn = self._get_conn()

        if self._owns_connection:
            # Own connection — manage transactions ourselves
            for batch_start in range(0, len(papers), batch_size):
                batch = papers[batch_start : batch_start + batch_size]
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    for paper in batch:
                        try:
                            self.upsert_paper(paper, commit=False)
                            count += 1
                        except Exception as e:
                            logger.warning(
                                "Failed to upsert paper {} into graph: {}",
                                paper.get("paper_id", "?"),
                                e,
                            )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.warning("Batch transaction failed during rebuild_graph: {}", e)
        else:
            # Borrowed connection — caller controls transaction, just write
            for paper in papers:
                try:
                    self.upsert_paper(paper, commit=False)
                    count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to upsert paper {} into graph: {}",
                        paper.get("paper_id", "?"),
                        e,
                    )

        return count

    def stats(self) -> dict[str, int]:
        """Return counts of nodes and edges in the graph."""
        conn = self._get_conn()
        return {
            "concepts": conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
            "authors": conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0],
            "citation_edges": conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0],
            "paper_concept_edges": conn.execute("SELECT COUNT(*) FROM paper_concepts").fetchone()[
                0
            ],
            "author_collaborations": conn.execute(
                "SELECT COUNT(*) FROM author_collaborations"
            ).fetchone()[0],
            "paper_related_edges": conn.execute("SELECT COUNT(*) FROM paper_related").fetchone()[0],
        }
