"""Concept exploration tool using the knowledge graph."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


TRACKS_FILE = Path.home() / ".researchbot" / "concept_tracks.json"
EXPLORATION_FILE = Path.home() / ".researchbot" / "exploration_context.json"


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

    def update_last_checked(self, concept_id: str, new_count: int) -> None:
        data = self._load()
        for t in data["tracks"]:
            if t["id"] == concept_id:
                t["last_checked_at"] = datetime.now(timezone.utc).isoformat()
                t["last_new_count"] = new_count
                break
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
        self.current_concept_id: str = ""
        self.current_concept_name: str = ""

    def push(self, concept_id: str, concept_name: str, neighbors: list[dict[str, Any]], papers: list[dict[str, Any]]) -> None:
        self.stack.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "neighbors": neighbors,  # [{id, name, co_count}]
            "papers": papers,         # [{paper_id, title, year, cited_by_count}]
        })
        self.current_concept_id = concept_id
        self.current_concept_name = concept_name
        self._save()

    def pop(self) -> dict[str, Any] | None:
        if not self.stack:
            return None
        self.stack.pop()
        self._save()
        if self.stack:
            top = self.stack[-1]
            self.current_concept_id = top["concept_id"]
            self.current_concept_name = top["concept_name"]
        else:
            self.current_concept_id = ""
            self.current_concept_name = ""
        return top if self.stack else None

    def clear(self) -> None:
        self.stack = []
        self.current_concept_id = ""
        self.current_concept_name = ""
        self._save()

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def _save(self) -> None:
        EXPLORATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stack": self.stack,
            "current_concept_id": self.current_concept_id,
            "current_concept_name": self.current_concept_name,
        }
        with open(EXPLORATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
            ctx.current_concept_id = data.get("current_concept_id", "")
            ctx.current_concept_name = data.get("current_concept_name", "")
            return ctx
        except (json.JSONDecodeError, IOError):
            return ctx