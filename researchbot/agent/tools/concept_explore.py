"""Concept exploration tool using the knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchbot.agent.tools.base import Tool
from researchbot.agent.tools.openalex_client import search_openalex_concepts
from researchbot.knowledge_graph import KnowledgeGraph, _safe_id
from researchbot.search_index import SearchIndex
from researchbot.config.schema import SemanticSearchConfig


TRACKS_FILE = Path.home() / ".researchbot" / "concept_tracks.json"
EXPLORATION_FILE = Path.home() / ".researchbot" / "exploration_context.json"


def _format_paper_list(papers: list[dict[str, Any]], start: int = 1) -> list[str]:
    """Format a list of papers into output lines."""
    lines = []
    for i, p in enumerate(papers, start):
        title = p.get("title", "?")
        year = p.get("year", "?")
        cited = p.get("citation_count") or p.get("cited_by_count", 0)
        lines.append(f"  [{i}] {title} ({year}) — cited by {cited}")
    return lines


def _get_kg_and_index(workspace: str | None, semantic_config: SemanticSearchConfig | None):
    """Helper to get KnowledgeGraph and SearchIndex instances."""
    if not workspace or not semantic_config:
        return None, None
    db_path = Path(workspace) / semantic_config.sqlite_db_path
    if not db_path.parent.exists():
        return None, None
    index = SearchIndex(db_path, semantic_config)
    index._ensure_graph_initialized()
    kg = index.get_graph()
    return kg, index


def _get_paper_metadata(
    paper_id: str,
    index: SearchIndex | None,
    kg: KnowledgeGraph | None,
    fields: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch paper metadata: try index.get_paper() first, fall back to raw SQL."""
    if index:
        p = index.get_paper(paper_id)
        if p:
            return p
    if kg:
        conn = kg._conn()
        cols = ", ".join(fields) if fields else "paper_id, title, year, citation_count, cited_by_count"
        row = conn.execute(f"SELECT {cols} FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row:
            result = dict(row)
            # Parse authors_json if present
            if "authors_json" in result and result["authors_json"]:
                try:
                    result["authors"] = json.loads(result["authors_json"])
                except Exception:
                    result["authors"] = []
            return result
    return None


async def _map_topic_to_concepts(
    topic: str,
    kg: KnowledgeGraph | None,
    openalex_api_key: str | None,
    proxy: str | None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Map a topic string to concept entries (local + OpenAlex)."""
    candidates: dict[str, dict[str, Any]] = {}

    # 1. Search local concepts table
    if kg:
        conn = kg._conn()
        pattern = f"%{topic}%"
        rows = conn.execute(
            "SELECT concept_id, display_name, level, wikipedia_url FROM concepts WHERE display_name LIKE ? LIMIT ?",
            (pattern, top_k * 2),
        ).fetchall()
        for r in rows:
            cid = r["concept_id"]
            candidates[cid] = {
                "id": cid,
                "display_name": r["display_name"],
                "level": r["level"],
                "wikipedia_url": r["wikipedia_url"] or "",
                "source": "local",
                "paper_count": 0,
            }
        # Get paper counts in one batch
        if candidates:
            placeholders = ",".join("?" * len(candidates))
            count_rows = conn.execute(
                f"SELECT concept_id, COUNT(*) as cnt FROM paper_concepts WHERE concept_id IN ({placeholders}) GROUP BY concept_id",
                list(candidates.keys()),
            ).fetchall()
            for row in count_rows:
                candidates[row["concept_id"]]["paper_count"] = row["cnt"]

    # 2. Supplement from OpenAlex if local results < 3
    if len(candidates) < 3:
        try:
            oa_results = await search_openalex_concepts(
                query=topic,
                api_key=openalex_api_key,
                proxy=proxy,
                top_k=top_k,
            )
            for oa in oa_results:
                cid = _safe_id(oa["display_name"])
                if cid not in candidates:
                    candidates[cid] = {
                        "id": cid,
                        "display_name": oa["display_name"],
                        "level": oa["level"],
                        "wikipedia_url": oa["wikipedia_url"] or "",
                        "source": "openalex",
                        "paper_count": oa["paper_count"],
                    }
        except Exception:
            pass

    # Sort by paper_count desc, take top_k
    sorted_concepts = sorted(candidates.values(), key=lambda x: x["paper_count"], reverse=True)
    return sorted_concepts[:top_k]


class ConceptTrackStore:
    """Persistent store for concept track subscriptions."""

    def __init__(self) -> None:
        TRACKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not TRACKS_FILE.exists():
            self._save({"tracks": []})

    def _load(self) -> dict[str, Any]:
        try:
            with open(TRACKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"tracks": []}

    def _save(self, data: dict[str, Any]) -> None:
        with open(TRACKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_all(self) -> list[dict[str, Any]]:
        return self._load().get("tracks", [])

    def add_or_update(self, concept_id: str, display_name: str) -> None:
        data = self._load()
        tracks = data["tracks"]
        existing = next((t for t in tracks if t["id"] == concept_id), None)
        now = datetime.now(timezone.utc).isoformat()
        if existing:
            existing["last_checked_at"] = now
            existing["name"] = display_name
        else:
            tracks.append({
                "id": concept_id,
                "name": display_name,
                "added_at": now,
                "last_checked_at": now,
                "last_new_count": 0,
            })
        self._save(data)

    def remove(self, concept_id: str) -> bool:
        data = self._load()
        original_len = len(data["tracks"])
        data["tracks"] = [t for t in data["tracks"] if t["id"] != concept_id]
        self._save(data)
        return len(data["tracks"]) < original_len

    def clear(self) -> None:
        self._save({"tracks": []})


class ExplorationContext:
    """In-memory exploration state for interactive concept drilling.

    Saved to ~/.researchbot/exploration_context.json so the agent can
    continue the conversation across turns.
    """

    def __init__(self) -> None:
        self.stack: list[dict[str, Any]] = []

    def push(self, concept_id: str, concept_name: str, neighbors: list[dict[str, Any]], papers: list[dict[str, Any]]) -> None:
        self.stack.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "neighbors": neighbors,
            "papers": papers,
        })
        self._save()

    def pop(self) -> dict[str, Any] | None:
        if not self.stack:
            return None
        self.stack.pop()
        self._save()
        return self.stack[-1] if self.stack else None

    def clear(self) -> None:
        self.stack = []
        self._save()

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def _save(self) -> None:
        EXPLORATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPLORATION_FILE, "w", encoding="utf-8") as f:
            json.dump({"stack": self.stack}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "ExplorationContext":
        """Load existing context from file, or return empty context."""
        ctx = cls()
        if not EXPLORATION_FILE.exists():
            return ctx
        try:
            with open(EXPLORATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ctx.stack = data.get("stack", [])
            return ctx
        except (json.JSONDecodeError, IOError):
            return ctx


class ConceptExploreTool(Tool):
    """Explore concepts in the knowledge graph.

    Supports three modes:
    - explore_topic: input a topic, map to concepts, explore concept network
    - explore_paper: explore concept network starting from a paper
    - track_concepts: subscribe to concept updates
    """

    name = "concept_explore"
    description = (
        "Explore concepts in the knowledge graph. Use explore_topic to discover "
        "concepts from a keyword, explore_paper to drill into a paper's concepts, "
        "or track_concepts to subscribe to updates for specific concepts."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["explore_topic", "explore_paper", "track_concepts", "list_tracks", "remove_track"],
                "description": "The exploration action to perform",
            },
            "topic": {
                "type": "string",
                "description": "Topic keyword for explore_topic",
            },
            "paper_id": {
                "type": "string",
                "description": "Paper ID for explore_paper",
            },
            "concept_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concept IDs to track (for track_concepts), or concept to explore directly (for explore_topic)",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of papers/concepts to return per step (default 10)",
                "default": 10,
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        workspace: str | None = None,
        semantic_config: SemanticSearchConfig | None = None,
        openalex_api_key: str | None = None,
        proxy: str | None = None,
    ):
        self._workspace = workspace
        self._semantic_config = semantic_config
        self._openalex_api_key = openalex_api_key
        self._proxy = proxy
        self._track_store = ConceptTrackStore()
        self._ctx = ExplorationContext.load()

    async def execute(
        self,
        action: str,
        topic: str | None = None,
        paper_id: str | None = None,
        concept_ids: list[str] | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> str:
        if action == "explore_topic":
            if concept_ids:
                return await self._resume_exploration(concept_ids[0])
            return await self._explore_topic(topic or "", top_k)
        elif action == "explore_paper":
            return await self._explore_paper(paper_id or "", top_k)
        elif action == "track_concepts":
            return await self._track_concepts(concept_ids or [])
        elif action == "list_tracks":
            return await self._list_tracks()
        elif action == "remove_track":
            return await self._remove_track(paper_id or "")
        else:
            return f"Unknown action: {action}"

    async def _resume_exploration(self, concept_id: str) -> str:
        """Resume exploration at a specific concept (drill-down continuation)."""
        kg, index = _get_kg_and_index(self._workspace, self._semantic_config)
        if kg is None:
            return "Error: Knowledge graph not available."
        conn = kg._conn()
        row = conn.execute("SELECT display_name FROM concepts WHERE concept_id = ?", (concept_id,)).fetchone()
        if not row:
            return f"Error: concept '{concept_id}' not found in local graph."
        concept_name = row["display_name"]
        return await self._start_concept_exploration(concept_id, concept_name, top_k=top_k)

    async def _explore_topic(self, topic: str, top_k: int) -> str:
        """Map topic to concepts and start interactive exploration."""
        kg, index = _get_kg_and_index(self._workspace, self._semantic_config)
        if kg is None:
            return "Error: Knowledge graph not available. Please configure semantic_search in config."

        # Map topic to concepts
        concepts = await _map_topic_to_concepts(
            topic, kg, self._openalex_api_key, self._proxy, top_k=5
        )

        if not concepts:
            return f"No concepts found for topic: {topic}\n\nTry a different keyword or enrich some papers first."

        # Show concept selection
        local = [c for c in concepts if c["source"] == "local"]
        oa = [c for c in concepts if c["source"] == "openalex"]

        lines = [f"探索主题: \"{topic}\"\n"]
        if local:
            lines.append(f"在本地图谱中找到 {len(local)} 个匹配的概念：")
            for i, c in enumerate(local, 1):
                lines.append(f"  [{i}] {c['display_name']} ({c['id']}) — {c['paper_count']} 篇论文")
        if oa:
            lines.append(f"\n补充自 OpenAlex：")
            for i, c in enumerate(oa, len(local) + 1):
                lines.append(f"  [{i}] {c['display_name']} — {c['paper_count']} 篇论文")

        lines.append("\n请选择概念编号，或直接输入 concept_id：")
        lines.append("(输入 'quit' 退出探索，输入 'back' 返回)")

        # Auto-select if only one concept
        if len(concepts) == 1:
            c = concepts[0]
            return await self._start_concept_exploration(c["id"], c["display_name"], top_k)

        return "\n".join(lines)

    async def _start_concept_exploration(self, concept_id: str, concept_name: str, top_k: int) -> str:
        """Begin exploration at a specific concept."""
        kg, index = _get_kg_and_index(self._workspace, self._semantic_config)
        if kg is None:
            return "Error: Knowledge graph not available."

        # Get neighbors
        neighbors = kg.get_concept_neighbors(concept_id, top_k=top_k)
        # Batch-fetch neighbor names to avoid N+1
        neighbor_ids = [nid for nid, _ in neighbors]
        name_map = {}
        if neighbor_ids:
            placeholders = ",".join("?" * len(neighbor_ids))
            rows = kg._conn().execute(
                f"SELECT concept_id, display_name FROM concepts WHERE concept_id IN ({placeholders})",
                neighbor_ids,
            ).fetchall()
            name_map = {r["concept_id"]: r["display_name"] for r in rows}
        neighbor_details = [
            {"id": nid, "name": name_map.get(nid, nid), "co_count": cnt}
            for nid, cnt in neighbors
        ]

        # Get representative papers
        paper_scores = kg.get_papers_by_concept(concept_id, top_k=top_k)
        papers = []
        for pid, score in paper_scores:
            p = _get_paper_metadata(pid, index, kg)
            if p:
                papers.append(p)

        # Save context
        self._ctx.push(concept_id, concept_name, neighbor_details, papers)

        lines = [
            f"概念: {concept_name}",
            f"═══════════════════════════════════",
            f"邻居概念（共 {len(neighbor_details)} 个）：",
        ]
        for i, n in enumerate(neighbor_details, 1):
            lines.append(f"  [{i}] {n['name']} ({n['id']}) — {n['co_count']}次共现")

        lines.append(f"\n代表性论文（共 {len(papers)} 篇）：")
        lines.extend(_format_paper_list(papers[:top_k]))

        lines.append("\n选择操作：")
        lines.append("  [1-{}] 沿邻居概念继续深入".format(len(neighbor_details) if neighbor_details else 0))
        lines.append("  [p] 查看代表性论文详情")
        lines.append("  [back] 返回上一层")
        lines.append("  [quit] 退出探索")

        return "\n".join(lines)

    async def _explore_paper(self, paper_id: str, top_k: int) -> str:
        """Explore concept network starting from a paper."""
        kg, index = _get_kg_and_index(self._workspace, self._semantic_config)
        if kg is None:
            return "Error: Knowledge graph not available."

        if not paper_id:
            return "Error: paper_id is required for explore_paper."

        # Get paper metadata
        paper = _get_paper_metadata(paper_id, index, kg, fields=["title", "year", "authors_json", "citation_count", "cited_by_count"])
        if not paper:
            return f"Error: Paper '{paper_id}' not found. Check the paper ID and try again."

        # Get concepts for this paper
        concept_ids = kg.get_paper_concepts(paper_id)
        if not concept_ids:
            return f"Paper '{paper_id}' has no concept tags in the graph. Enrich it with paper_enrich first."

        # Batch-fetch all concept display names
        concept_placeholders = ",".join("?" * len(concept_ids))
        concept_name_rows = kg._conn().execute(
            f"SELECT concept_id, display_name FROM concepts WHERE concept_id IN ({concept_placeholders})",
            concept_ids,
        ).fetchall()
        concept_name_map = {r["concept_id"]: r["display_name"] for r in concept_name_rows}

        # Batch-fetch paper counts for all concepts
        count_rows = kg._conn().execute(
            f"SELECT concept_id, COUNT(*) as cnt FROM paper_concepts WHERE concept_id IN ({concept_placeholders}) GROUP BY concept_id",
            concept_ids,
        ).fetchall()
        count_map = {r["concept_id"]: r["cnt"] for r in count_rows}

        # Collect all neighbor IDs across all concepts for batch name lookup
        all_neighbor_ids: set[str] = set()
        concept_details = []
        for cid in concept_ids:
            name = concept_name_map.get(cid, cid)
            paper_count = count_map.get(cid, 0)
            neighbors = kg.get_concept_neighbors(cid, top_k=5)
            for nid, _ in neighbors:
                all_neighbor_ids.add(nid)
            concept_details.append({
                "id": cid,
                "name": name,
                "paper_count": paper_count,
                "neighbors": neighbors,
            })

        # Batch-fetch all neighbor names
        neighbor_name_map = {}
        if all_neighbor_ids:
            n_placeholders = ",".join("?" * len(all_neighbor_ids))
            n_rows = kg._conn().execute(
                f"SELECT concept_id, display_name FROM concepts WHERE concept_id IN ({n_placeholders})",
                list(all_neighbor_ids),
            ).fetchall()
            neighbor_name_map = {r["concept_id"]: r["display_name"] for r in n_rows}

        # Fill in neighbor names
        for cd in concept_details:
            cd["neighbors"] = [
                {"id": nid, "name": neighbor_name_map.get(nid, nid), "co_count": cnt}
                for nid, cnt in cd["neighbors"]
            ]

        # Find common citations across all concepts
        if len(concept_ids) > 1:
            common = kg.find_common_citations(concept_ids)
            common_count = len(common)
        else:
            common_count = 0

        # Build output
        title = paper.get("title", "?")
        year = paper.get("year", "?")
        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown"

        lines = [
            f"探索论文: {paper_id}",
            f"═══════════════════════════════════",
            f"标题: {title}",
            f"年份: {year} | 作者: {author_str}",
            f"\n概念标签（共 {len(concept_details)} 个）：",
        ]
        for cd in concept_details:
            lines.append(f"  • {cd['name']} ({cd['id']}) — {cd['paper_count']} 篇论文")
            if cd["neighbors"]:
                neighbor_str = ", ".join(f"{n['name']}({n['co_count']})" for n in cd["neighbors"][:3])
                lines.append(f"    邻居: {neighbor_str}")

        if common_count > 0:
            lines.append(f"\n共同论文（{len(concept_ids)} 个概念交集）: {common_count} 篇")

        lines.append("\n选择操作：")
        for i, cd in enumerate(concept_details, 1):
            lines.append(f"  [{i}] 沿 {cd['name']} → {cd['paper_count']} 篇深入")
        lines.append("  [c] 查看共同论文")
        lines.append("  [quit] 退出探索")

        # Save initial state for paper exploration
        self._ctx.push(concept_details[0]["id"], concept_details[0]["name"], concept_details[0]["neighbors"], [])

        return "\n".join(lines)

    async def _track_concepts(self, concept_ids: list[str]) -> str:
        """Add or update concept track subscriptions."""
        if not concept_ids:
            return "Error: concept_ids required for track_concepts."

        kg, _ = _get_kg_and_index(self._workspace, self._semantic_config)
        if kg is None:
            return "Error: Knowledge graph not available."

        added = []
        for cid in concept_ids:
            conn = kg._conn()
            row = conn.execute("SELECT display_name FROM concepts WHERE concept_id = ?", (cid,)).fetchone()
            name = row["display_name"] if row else cid
            self._track_store.add_or_update(cid, name)
            added.append(f"  • {name} ({cid})")

        return "已添加/更新跟踪的概念：\n" + "\n".join(added)

    async def _list_tracks(self) -> str:
        """List all tracked concepts."""
        tracks = self._track_store.get_all()
        if not tracks:
            return "暂无跟踪的概念。使用 track_concepts 添加。"

        lines = [f"跟踪的概念（共 {len(tracks)} 个）：\n"]
        for i, t in enumerate(tracks, 1):
            name = t.get("name", t["id"])
            added = t.get("added_at", "")[:10] if t.get("added_at") else "?"
            last_checked = t.get("last_checked_at", "")[:10] if t.get("last_checked_at") else "?"
            new_count = t.get("last_new_count", 0)
            lines.append(f"  [{i}] {name}")
            lines.append(f"      ID: {t['id']}")
            lines.append(f"      添加于: {added} | 上次检查: {last_checked} | 新增: {new_count} 篇")

        lines.append("\n使用 remove_track <concept_id> 移除跟踪。")
        return "\n".join(lines)

    async def _remove_track(self, concept_id: str) -> str:
        """Remove a concept from tracking."""
        if not concept_id:
            return "Error: concept_id required."
        removed = self._track_store.remove(concept_id)
        if removed:
            return f"已移除跟踪: {concept_id}"
        return f"未找到跟踪: {concept_id}"