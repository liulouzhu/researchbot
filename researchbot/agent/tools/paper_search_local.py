"""Local semantic search tool for papers.

Provides hybrid search (FTS5 + vector) over local literature storage.
Supports filtering by topic, tags, year, categories, and source.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.config.schema import SemanticSearchConfig
from researchbot.search_index import SearchIndex


class PaperSearchLocalTool(Tool):
    """Search local papers using semantic search.

    Supports hybrid search combining keyword matching (FTS5) and
    vector similarity (sqlite-vec). Falls back to FTS5-only
    when vector search is unavailable.

    Filtering:
    - topic: substring match on topic/tags
    - tags: any of the specified tags match
    - year: exact year match
    - year_from/year_to: year range
    - categories: any of the categories match
    - source: exact source match (arxiv, crossref, openalex)

    Graph expansion (via knowledge graph):
    - expand_via_citations: extend results along citation edges
    - expand_depth: how many citation hops to traverse
    """

    name = "paper_search_local"
    description = (
        "Search local papers using semantic search. "
        "Combines keyword and vector similarity for best results. "
        "Supports filtering by topic, tags, year, categories, and source. "
        "Uses hybrid search with reranking when enabled. "
        "Optionally expands results via citation graph (cited/citing papers)."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g. 'machine learning for graph analysis')",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "topic": {
                "type": "string",
                "description": "Filter by topic (substring match)",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by tags (any match)",
            },
            "year": {
                "type": "string",
                "description": "Filter by exact publication year",
            },
            "year_from": {
                "type": "integer",
                "description": "Filter by year >= this value",
            },
            "year_to": {
                "type": "integer",
                "description": "Filter by year <= this value",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by categories (any match)",
            },
            "source": {
                "type": "string",
                "description": "Filter by source (arxiv, crossref, openalex)",
            },
            "rerank": {
                "type": "boolean",
                "description": "Apply LLM reranking to top results",
                "default": True,
            },
            "expand_via_citations": {
                "type": "boolean",
                "description": "Expand results via citation graph (cited/citing papers)",
                "default": False,
            },
            "expand_depth": {
                "type": "integer",
                "description": "Citation graph expansion depth (1-3, default 1)",
                "minimum": 1,
                "maximum": 3,
                "default": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._index: SearchIndex | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_index(self) -> SearchIndex:
        """Get or create the search index."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        return self._index

    async def _ensure_initialized(self) -> None:
        """Ensure index is initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    index = self._get_index()
                    await index.initialize()
                    self._initialized = True

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        topic: str | None = None,
        tags: list[str] | None = None,
        year: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
        source: str | None = None,
        rerank: bool = True,
        expand_via_citations: bool = False,
        expand_depth: int = 1,
        **kwargs: Any,
    ) -> str:
        """Search local papers."""
        if not self._workspace:
            return "Error: workspace not configured"

        query = (query or "").strip()
        if not query:
            return "Error: query must be a non-empty string"

        try:
            await self._ensure_initialized()
            index = self._get_index()

            results = await index.search(
                query=query,
                top_k=top_k,
                topic=topic,
                tags=tags,
                year=year,
                year_from=year_from,
                year_to=year_to,
                categories=categories,
                source=source,
                rerank=rerank,
            )

            # Graph expansion via citation edges
            expansion_note = ""
            if expand_via_citations and results:
                try:
                    from researchbot.knowledge_graph import KnowledgeGraph

                    kg = index.get_graph()
                    # Track source paper for each expanded paper (preserves source mapping)
                    expanded_source: dict[str, str] = {}

                    # Collect cited and citing papers from initial results
                    for r in results:
                        pid = r.get("paper_id", "")
                        if not pid:
                            continue
                        # Get outbound citations (papers this paper cites)
                        cited = kg.get_cited_papers(pid, depth=expand_depth)
                        for cp in cited:
                            if cp not in expanded_source:
                                expanded_source[cp] = pid
                        # Get inbound citations (papers that cite this paper)
                        citing = kg.get_citing_papers(pid, depth=expand_depth)
                        for cp in citing:
                            if cp not in expanded_source:
                                expanded_source[cp] = pid

                    if expanded_source:
                        # Fetch metadata for expanded papers
                        expanded_papers = []
                        for eid, src_pid in expanded_source.items():
                            paper = index.get_paper(eid)
                            if paper:
                                paper["_expanded"] = True
                                paper["final_score"] = 0.0
                                paper["_expansion_source"] = src_pid
                                expanded_papers.append(paper)

                        if expanded_papers:
                            # Merge expanded papers into results (below initial results)
                            results.extend(expanded_papers[:top_k])
                            expansion_note = f" (+ {len(expanded_papers)} citation-expanded)"
                except Exception:
                    pass  # Don't fail if graph expansion fails

            if not results:
                return f"No papers found for query: {query}\n\nTry broadening your search or removing filters."

            # Build output
            search_mode = "hybrid (FTS5 + vector)" if index.sqlite_vec_available and index.embedding_client.available else "FTS5 keyword only"
            if expansion_note:
                search_mode += " + citation graph expansion"

            lines = [
                f"Search results for: {query}\n",
                f"Found {len(results)} result(s){expansion_note}\n",
                f"Search mode: {search_mode}\n",
                "-" * 60,
            ]

            for i, r in enumerate(results, 1):
                paper_id = r.get("paper_id", "?")
                title = r.get("title", "?")
                paper_year = r.get("year", "?")
                paper_source = r.get("source", "")
                final_score = r.get("final_score", 0)
                lex_score = r.get("lexical_score", 0)
                vec_score = r.get("vector_score", 0)
                rerank_score = r.get("rerank_score")
                is_expanded = r.get("_expanded", False)
                expansion_src = r.get("_expansion_source", "")

                prefix = "[E] " if is_expanded else f"[{i}] "
                lines.append(f"\n{prefix}{title}")
                lines.append(f"    Paper ID: {paper_id}")
                lines.append(f"    Year: {paper_year} | Source: {paper_source}")
                if is_expanded and expansion_src:
                    lines.append(f"    (via citation from {expansion_src})")
                if not is_expanded:
                    lines.append(f"    Scores: final={final_score:.3f} lexical={lex_score:.3f} vector={vec_score:.3f}")
                    if rerank_score is not None:
                        lines.append(f"    Rerank score: {rerank_score}")

                # Show matched filters if any
                filters = r.get("matched_filters", {})
                if filters.get("topic"):
                    lines.append(f"    Topic: {filters['topic']}")
                if filters.get("tags"):
                    lines.append(f"    Tags: {', '.join(filters['tags'][:5])}")

                # Show abstract snippet if available
                snippet = r.get("abstract_snippet", "")
                if snippet:
                    lines.append(f"    ...{snippet}...")

            return "\n".join(lines)

        except Exception as e:
            return f"Search error: {e}"

    def close(self) -> None:
        """Close the search index."""
        if self._index:
            self._index.close()
            self._index = None
            self._initialized = False


class PaperIndexTool(Tool):
    """Update the local paper search index.

    Indexes or re-indexes papers in local storage for semantic search.
    Normally called automatically, but can be invoked manually.
    """

    name = "paper_index"
    description = (
        "Update the local paper search index. "
        "Normally happens automatically when papers are saved, "
        "but can be invoked manually to rebuild the index."
    )

    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Index a specific paper by ID (optional)",
            },
            "rebuild": {
                "type": "boolean",
                "description": "Rebuild the entire index from all local papers",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._index: SearchIndex | None = None

    def _resolve_path(self, path: str) -> Path:
        """Resolve path within workspace."""
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_index(self) -> SearchIndex:
        """Get or create the search index."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        return self._index

    def _load_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Load a paper from local storage."""
        json_path = self._resolve_path(f"literature/papers/{paper_id}.json")
        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_all_papers(self) -> list[dict[str, Any]]:
        """Load all papers from local storage."""
        papers_dir = self._resolve_path("literature/papers")
        if not papers_dir.exists():
            return []

        papers = []
        import json as json_module
        for fp in papers_dir.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    papers.append(json_module.load(f))
            except Exception:
                continue
        return papers

    async def execute(
        self,
        paper_id: str | None = None,
        rebuild: bool = False,
        **kwargs: Any,
    ) -> str:
        """Update the search index."""
        if not self._workspace:
            return "Error: workspace not configured"

        index = self._get_index()
        await index.initialize()

        if rebuild:
            papers = self._load_all_papers()
            updated = 0
            for paper in papers:
                pid = paper.get("paper_id", "")
                if pid:
                    result = await index.upsert_paper(paper)
                    if result["content_changed"]:
                        updated += 1
            index.close()
            return f"Index rebuilt: {updated}/{len(papers)} papers updated"

        if paper_id:
            paper = self._load_paper(paper_id)
            if not paper:
                return f"Paper not found: {paper_id}"
            result = await index.upsert_paper(paper)
            index.close()
            if result["content_changed"]:
                return f"Indexed paper: {paper_id}"
            else:
                return f"Paper unchanged (skipped): {paper_id}"

        return "Error: specify paper_id or rebuild=True"


class GraphQueryTool(Tool):
    """Query the local knowledge graph for paper relationships.

    Provides citation analysis, co-authorship discovery, concept relationships,
    and path finding between papers — all from locally stored data.
    """

    name = "graph_query"
    description = (
        "Query the local knowledge graph for paper citation relationships, "
        "co-authors, concept connections, and paths between papers. "
        "Use this instead of external APIs when you have papers indexed locally. "
        "Call this tool after finding relevant papers to understand their relationships."
    )

    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Graph operation to perform",
                "enum": [
                    "get_citing_papers",
                    "get_cited_papers",
                    "get_papers_by_concept",
                    "get_co_authors",
                    "get_concept_neighbors",
                    "find_common_citations",
                    "find_path",
                    "get_related_papers",
                    "stats",
                ],
            },
            "paper_id": {
                "type": "string",
                "description": "Paper ID for citation/co-author queries",
            },
            "concept_id": {
                "type": "string",
                "description": "Concept ID for concept-related queries",
            },
            "author_id": {
                "type": "string",
                "description": "Author ID for co-author queries (e.g. 'vaswani_a')",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paper IDs for common citations or path queries",
            },
            "paper_a": {
                "type": "string",
                "description": "Source paper ID for path finding",
            },
            "paper_b": {
                "type": "string",
                "description": "Target paper ID for path finding",
            },
            "depth": {
                "type": "integer",
                "description": "Graph traversal depth (1-3, default 1)",
                "minimum": 1,
                "maximum": 3,
                "default": 1,
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
        },
        "required": ["operation"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()
        self._index: SearchIndex | None = None
        self._kg_initialized = False

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        return p

    def _get_graph(self):
        """Get or create the knowledge graph (sync, no await needed)."""
        if self._index is None:
            db_path = self._resolve_path(self._semantic_config.sqlite_db_path)
            self._index = SearchIndex(db_path, self._semantic_config)
        # Ensure graph tables exist (sync, idempotent)
        self._index._ensure_graph_initialized()
        from researchbot.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(self._index)

    def _load_paper_metadata(self, paper_ids: list[str]) -> dict[str, dict]:
        """Load title/year for a list of paper IDs."""
        if not self._index:
            return {}
        conn = self._index._get_conn()
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT paper_id, title, year FROM papers WHERE paper_id IN ({placeholders})",
            paper_ids,
        ).fetchall()
        return {r["paper_id"]: dict(r) for r in rows}

    def _paper_exists(self, paper_id: str) -> bool:
        """Check whether a paper exists in local storage/index."""
        if not self._index or not paper_id:
            return False
        conn = self._index._get_conn()
        row = conn.execute(
            "SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1",
            (paper_id,),
        ).fetchone()
        return row is not None

    def execute(
        self,
        operation: str,
        paper_id: str | None = None,
        concept_id: str | None = None,
        author_id: str | None = None,
        paper_ids: list[str] | None = None,
        paper_a: str | None = None,
        paper_b: str | None = None,
        depth: int = 1,
        top_k: int = 20,
        **kwargs: Any,
    ) -> str:
        """Query the knowledge graph."""
        if not self._workspace:
            return "Error: workspace not configured"

        try:
            kg = self._get_graph()

            if operation == "stats":
                s = kg.stats()
                return (
                    f"Knowledge Graph Stats:\n"
                    f"  Concepts: {s['concepts']}\n"
                    f"  Authors: {s['authors']}\n"
                    f"  Citation edges: {s['citation_edges']}\n"
                    f"  Paper-concept edges: {s['paper_concept_edges']}\n"
                    f"  Author collaborations: {s['author_collaborations']}\n"
                    f"  Paper related edges: {s['paper_related_edges']}"
                )

            if operation == "get_citing_papers":
                if not paper_id:
                    return "Error: paper_id is required for get_citing_papers"
                ids = kg.get_citing_papers(paper_id, depth=depth)
                if not ids:
                    if not self._paper_exists(paper_id):
                        return (
                            f"Paper '{paper_id}' is not indexed in local papers/graph. "
                            "Index or save the paper first (e.g., paper_get + paper_save/paper_enrich), "
                            "then run graph_query again."
                        )
                    return f"No citing papers found for {paper_id}"
                meta = self._load_paper_metadata(ids)
                lines = [f"Papers that cite '{paper_id}' (depth={depth}):\n"]
                for i, pid in enumerate(ids[:top_k], 1):
                    m = meta.get(pid, {})
                    title = m.get("title", pid)
                    year = m.get("year", "?")
                    lines.append(f"  {i}. {title} ({year}) [{pid}]")
                return "\n".join(lines)

            if operation == "get_cited_papers":
                if not paper_id:
                    return "Error: paper_id is required for get_cited_papers"
                ids = kg.get_cited_papers(paper_id, depth=depth)
                if not ids:
                    if not self._paper_exists(paper_id):
                        return (
                            f"Paper '{paper_id}' is not indexed in local papers/graph. "
                            "Index or save the paper first (e.g., paper_get + paper_save/paper_enrich), "
                            "then run graph_query again."
                        )
                    return f"No cited papers found for {paper_id}"
                meta = self._load_paper_metadata(ids)
                lines = [f"Papers cited by '{paper_id}' (depth={depth}):\n"]
                for i, pid in enumerate(ids[:top_k], 1):
                    m = meta.get(pid, {})
                    title = m.get("title", pid)
                    year = m.get("year", "?")
                    lines.append(f"  {i}. {title} ({year}) [{pid}]")
                return "\n".join(lines)

            if operation == "get_related_papers":
                if not paper_id:
                    return "Error: paper_id is required for get_related_papers"
                ids = kg.get_related_papers(paper_id, depth=depth)
                if not ids:
                    return f"No related papers found for {paper_id}"
                meta = self._load_paper_metadata(ids)
                lines = [f"Papers related to '{paper_id}' (depth={depth}):\n"]
                for i, pid in enumerate(ids[:top_k], 1):
                    m = meta.get(pid, {})
                    title = m.get("title", pid)
                    year = m.get("year", "?")
                    lines.append(f"  {i}. {title} ({year}) [{pid}]")
                return "\n".join(lines)

            if operation == "get_papers_by_concept":
                if not concept_id:
                    return "Error: concept_id is required for get_papers_by_concept"
                results = kg.get_papers_by_concept(concept_id, top_k=top_k)
                if not results:
                    return f"No papers found for concept: {concept_id}"
                paper_ids_list = [pid for pid, _ in results]
                meta = self._load_paper_metadata(paper_ids_list)
                lines = [f"Papers tagged with concept '{concept_id}':\n"]
                for i, (pid, score) in enumerate(results[:top_k], 1):
                    m = meta.get(pid, {})
                    title = m.get("title", pid)
                    lines.append(f"  {i}. {title} (score={score:.2f}) [{pid}]")
                return "\n".join(lines)

            if operation == "get_co_authors":
                if not author_id:
                    return "Error: author_id is required for get_co_authors (e.g. 'vaswani_a')"
                results = kg.get_co_authors(author_id, top_k=top_k)
                if not results:
                    return f"No co-authors found for: {author_id}"
                # Get author display names
                conn = self._index._get_conn()
                aids = [aid for aid, _ in results]
                placeholders = ",".join("?" * len(aids))
                rows = conn.execute(
                    f"SELECT author_id, display_name FROM authors WHERE author_id IN ({placeholders})",
                    aids,
                ).fetchall()
                name_map = {r["author_id"]: r["display_name"] for r in rows}
                lines = [f"Co-authors of '{name_map.get(author_id, author_id)}' (author_id={author_id}):\n"]
                for i, (aid, count) in enumerate(results[:top_k], 1):
                    name = name_map.get(aid, aid)
                    lines.append(f"  {i}. {name} (collaborated on {count} paper(s)) [{aid}]")
                return "\n".join(lines)

            if operation == "get_concept_neighbors":
                if not concept_id:
                    return "Error: concept_id is required for get_concept_neighbors"
                results = kg.get_concept_neighbors(concept_id, top_k=top_k)
                if not results:
                    return f"No neighboring concepts found for: {concept_id}"
                conn = self._index._get_conn()
                cids = [cid for cid, _ in results]
                placeholders = ",".join("?" * len(cids))
                rows = conn.execute(
                    f"SELECT concept_id, display_name FROM concepts WHERE concept_id IN ({placeholders})",
                    cids,
                ).fetchall()
                name_map = {r["concept_id"]: r["display_name"] for r in rows}
                lines = [f"Concepts co-occurring with '{name_map.get(concept_id, concept_id)}':\n"]
                for i, (cid, cnt) in enumerate(results[:top_k], 1):
                    name = name_map.get(cid, cid)
                    lines.append(f"  {i}. {name} (co-occurred in {cnt} papers) [{cid}]")
                return "\n".join(lines)

            if operation == "find_common_citations":
                if not paper_ids or len(paper_ids) < 2:
                    return "Error: paper_ids (list of 2+ paper IDs) is required for find_common_citations"
                results = kg.find_common_citations(paper_ids)
                if not results:
                    return "No common citations found among the given papers."
                paper_ids_list = [pid for pid, _ in results]
                meta = self._load_paper_metadata(paper_ids_list)
                lines = [f"Papers commonly cited by all {len(paper_ids)} papers:\n"]
                for i, (pid, _) in enumerate(results[:top_k], 1):
                    m = meta.get(pid, {})
                    title = m.get("title", pid)
                    lines.append(f"  {i}. {title} [{pid}]")
                return "\n".join(lines)

            if operation == "find_path":
                if not paper_a or not paper_b:
                    return "Error: paper_a and paper_b are required for find_path"
                path = kg.find_path(paper_a, paper_b, max_depth=depth)
                if path is None:
                    return f"No citation path found between '{paper_a}' and '{paper_b}' within depth={depth}"
                # Load metadata for all papers in path
                all_ids = []
                for p in path:
                    all_ids.extend(p)
                all_ids = list(dict.fromkeys(all_ids))  # dedupe preserving order
                meta = self._load_paper_metadata(all_ids)
                lines = [f"Citation path from '{meta.get(paper_a, {}).get('title', paper_a)}' to '{meta.get(paper_b, {}).get('title', paper_b)}':\n"]
                for pi, pids in enumerate(path, 1):
                    step_lines = []
                    for i, pid in enumerate(pids):
                        m = meta.get(pid, {})
                        title = m.get("title", pid)
                        year = m.get("year", "?")
                        step_lines.append(f"    {'→' if i > 0 else ''}{title} ({year}) [{pid}]")
                    lines.append(f"  Path {pi}:")
                    lines.extend(step_lines)
                return "\n".join(lines)

            return f"Unknown operation: {operation}"

        except Exception as e:
            return f"Graph query error: {e}"


class KnowledgeGraphRebuildTool(Tool):
    """Rebuild the knowledge graph from all local papers.

    Scans literature/papers/, extracts graph relationships
    (citations, concepts, authors, related_works), and rebuilds the graph.
    Use this after importing many papers or when graph statistics seem stale.
    """

    name = "knowledge_graph_rebuild"
    description = (
        "Rebuild (重建) the entire knowledge graph from all papers in local storage. "
        "This reconstructs citation edges, concept nodes, author nodes, and related-work edges "
        "from the papers stored in literature/papers/. "
        "If the user says '重建图谱', '重建知识图谱', 'rebuild the graph', or 'rebuild knowledge graph', "
        "call this tool directly and do not use exec or shell commands first. "
        "Use this after bulk-importing papers or when graph queries return stale/missing data."
    )

    parameters = {
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace directory path (optional). Uses configured workspace if not provided.",
            },
            "config": {
                "type": "string",
                "description": "Path to config file (optional). Uses default config if not provided.",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
    ):
        self._workspace = Path(workspace) if workspace else None
        self._semantic_config = semantic_config or SemanticSearchConfig()

    async def execute(
        self,
        workspace: str | None = None,
        config: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Rebuild the knowledge graph."""
        from researchbot.config.loader import load_config, set_config_path
        from researchbot.search_index import rebuild_graph_from_workspace

        # Resolve workspace
        ws: Path | None
        if workspace:
            ws = Path(workspace) if Path(workspace).is_absolute() else (self._workspace / workspace if self._workspace else Path(workspace))
        elif self._workspace:
            ws = self._workspace
        else:
            return "Error: workspace not configured and no workspace provided"

        # Load config if provided
        semantic_config = self._semantic_config
        if config:
            config_path = Path(config).expanduser().resolve()
            if not config_path.exists():
                return f"Error: config file not found: {config_path}"
            set_config_path(config_path)
            loaded_config = load_config(config_path)
            semantic_config = getattr(loaded_config.literature, "semantic_search", None) or semantic_config

        papers_dir = ws / "literature" / "papers"
        if not papers_dir.exists():
            return (
                f"Papers directory not found: {papers_dir}\n"
                "Nothing to rebuild. Make sure papers are saved first "
                "(e.g., use paper_save or paper_enrich)."
            )

        result = await rebuild_graph_from_workspace(ws, semantic_config)

        lines = [
            f"Knowledge graph rebuild complete\n",
            f"Papers directory: {papers_dir} [{'found' if result['papers_dir_found'] else 'NOT FOUND'}]\n",
            f"Total papers found: {result['total']}\n",
            f"Papers processed: {result['count']}\n",
            "-" * 50,
        ]

        stats = result.get("stats") or {}
        if stats:
            lines.extend([
                "Graph statistics:",
                f"  Concepts:             {stats.get('concepts', 0):>6}",
                f"  Authors:              {stats.get('authors', 0):>6}",
                f"  Citation edges:       {stats.get('citation_edges', 0):>6}",
                f"  Paper-concept edges:  {stats.get('paper_concept_edges', 0):>6}",
                f"  Author collaborations:{stats.get('author_collaborations', 0):>6}",
                f"  Paper related edges:  {stats.get('paper_related_edges', 0):>6}",
            ])
        else:
            lines.append("(no graph stats available)")

        return "\n".join(lines)
