"""SQLite persistence helpers for FastAPI demo state.

This module provides a small persistence layer for server_fastapi.py so
generated charts, evaluations, and system logs survive process restarts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


_LOCK = Lock()


class StudioStore:
    """Simple SQLite-backed store for studio entities."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS charts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        artist TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        bpm INTEGER NOT NULL,
                        genre TEXT NOT NULL,
                        rating REAL NOT NULL DEFAULT 0,
                        plays INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chart_id INTEGER NOT NULL,
                        fun INTEGER NOT NULL,
                        musicality INTEGER NOT NULL,
                        playability INTEGER NOT NULL,
                        coherence INTEGER NOT NULL,
                        comments TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(chart_id) REFERENCES charts(id)
                    );

                    CREATE TABLE IF NOT EXISTS key_value (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );
                    """
                )

    def append_log(self, entry: Dict[str, Any]) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO system_logs(timestamp, level, message) VALUES(?, ?, ?)",
                    (entry["timestamp"], entry["level"], entry["message"]),
                )

    def list_logs(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, level, message FROM system_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_chart(self, chart: Dict[str, Any]) -> Dict[str, Any]:
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO charts(title, artist, difficulty, bpm, genre, rating, plays, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chart["title"],
                        chart["artist"],
                        chart["difficulty"],
                        int(chart["bpm"]),
                        chart["genre"],
                        float(chart.get("rating", 0)),
                        int(chart.get("plays", 0)),
                        chart["created_at"],
                    ),
                )
                chart_id = int(cur.lastrowid)
        stored = dict(chart)
        stored["id"] = chart_id
        return stored

    def list_charts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, artist, difficulty, bpm, genre, rating, plays, created_at FROM charts ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unrated_chart(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, artist, difficulty, bpm, genre, rating, plays, created_at
                FROM charts
                WHERE rating <= 0
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def submit_evaluation(
        self,
        *,
        chart_id: int,
        fun: int,
        musicality: int,
        playability: int,
        coherence: int,
        comments: str,
        created_at: str,
    ) -> bool:
        rating = (fun + musicality + playability + coherence) / 4
        with _LOCK:
            with self._connect() as conn:
                existing = conn.execute("SELECT id FROM charts WHERE id = ?", (chart_id,)).fetchone()
                if not existing:
                    return False
                conn.execute(
                    """
                    INSERT INTO evaluations(chart_id, fun, musicality, playability, coherence, comments, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chart_id, fun, musicality, playability, coherence, comments, created_at),
                )
                conn.execute("UPDATE charts SET rating = ? WHERE id = ?", (rating, chart_id))
        return True

    def list_evaluations(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chart_id, fun, musicality, playability, coherence, comments, created_at
                FROM evaluations ORDER BY id DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def set_json(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO key_value(key, value_json) VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    (key, payload),
                )

    def get_json(self, key: str, default: Any) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM key_value WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

