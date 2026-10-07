"""SQLite conversations, isolated by random session IDs."""

import sqlite3
from contextlib import closing
from pathlib import Path


def _connect(data_dir: Path) -> sqlite3.Connection:
    directory = data_dir / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "sessions.sqlite", timeout=15)
    db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id)")
    return db


def load_history(session_id: str, data_dir: Path, limit: int = 10) -> list[dict[str, str]]:
    with closing(_connect(data_dir)) as db:
        rows = db.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                          (session_id, limit)).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def save_turn(session_id: str, question: str, answer: str, data_dir: Path) -> None:
    with closing(_connect(data_dir)) as db:
        with db:
            db.executemany("INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                           [(session_id, "user", question), (session_id, "assistant", answer)])
