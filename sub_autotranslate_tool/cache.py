"""Persistent translation memory.

Anime releases repeat themselves heavily: opening/ending lyrics, location
signs and nameplates are identical in every episode of a season. Caching
translated lines on disk means a season costs roughly one episode's worth of
tokens for that material, and — just as usefully — the repeated lines come
out worded identically everywhere.

The cache key covers everything that can change the expected output: model,
language pair, line kind, speaker, and a fingerprint of the prompt/glossary/
context. Editing the prompt or the glossary therefore invalidates entries
instead of silently serving output from the old instructions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "sub-autotranslate-tool"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "translations.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    key         TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    translation TEXT NOT NULL,
    model       TEXT,
    kind        TEXT,
    created     REAL DEFAULT (julianday('now'))
);
"""


def make_fingerprint(*parts: str) -> str:
    """Short stable hash of the instructions a translation depended on."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


class TranslationCache:
    """SQLite-backed store of previously translated lines."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CACHE_PATH
        self._conn: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            # A cache is an optimisation; never fail the run over it.
            logger.warning("translation cache disabled (%s): %s", self.path, exc)
            self._conn = None

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    @staticmethod
    def key(
        *,
        payload: str,
        model: str,
        source_language: str,
        target_language: str,
        kind: str,
        speaker: str,
        fingerprint: str,
    ) -> str:
        return make_fingerprint(
            payload, model, source_language, target_language, kind, speaker, fingerprint
        )

    def get(self, key: str) -> str | None:
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT translation FROM translations WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("cache read failed: %s", exc)
            return None
        return row[0] if row else None

    def put_many(self, items: list[tuple[str, str, str, str, str]]) -> None:
        """Store (key, source, translation, model, kind) rows."""
        if self._conn is None or not items:
            return
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO translations "
                "(key, source, translation, model, kind) VALUES (?, ?, ?, ?, ?)",
                items,
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("cache write failed: %s", exc)

    def count(self) -> int:
        if self._conn is None:
            return 0
        try:
            return self._conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
        except sqlite3.Error:
            return 0

    def clear(self) -> int:
        removed = self.count()
        if self._conn is not None:
            try:
                self._conn.execute("DELETE FROM translations")
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.warning("cache clear failed: %s", exc)
                return 0
        return removed

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class NullCache(TranslationCache):
    """Drop-in cache that stores nothing (--no-cache)."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the DB setup
        self.path = Path(os.devnull)
        self._conn = None
