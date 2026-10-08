"""SQLite draft review and local support-ticket response adapter."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewConflict(ValueError):
    pass


class DraftStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    ticket_text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    generator_version TEXT NOT NULL,
                    suggested_response TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                    reviewer TEXT,
                    final_response TEXT,
                    review_reason TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ticket_responses (
                    id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    draft_id TEXT NOT NULL UNIQUE,
                    response TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(draft_id) REFERENCES drafts(id)
                );
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def create(self, ticket_id: str, ticket_text: str, label: str,
               classifier_version: str, generator_version: str,
               suggested_response: str) -> dict:
        draft_id = str(uuid4())
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                INSERT INTO drafts (id, ticket_id, ticket_text, label, classifier_version,
                    generator_version, suggested_response, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (draft_id, ticket_id, ticket_text, label, classifier_version,
                  generator_version, suggested_response, utc_now()))
        return self.get(draft_id)

    def get(self, draft_id: str) -> dict | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        return dict(row) if row else None

    def decide(self, draft_id: str, reviewer: str, approve: bool,
               edited_response: str | None = None, reason: str | None = None) -> dict:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Reviewer is required")
        if approve:
            if edited_response is not None and not edited_response.strip():
                raise ValueError("Approved response cannot be blank")
        elif not reason or not reason.strip():
            raise ValueError("Rejection reason is required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if row is None:
                raise KeyError(draft_id)
            if row["status"] != "pending":
                raise ReviewConflict("Draft has already been reviewed")
            status = "approved" if approve else "rejected"
            final = (edited_response.strip() if edited_response is not None else
                     row["suggested_response"]) if approve else None
            now = utc_now()
            connection.execute("""
                UPDATE drafts SET status=?, reviewer=?, final_response=?, review_reason=?, reviewed_at=?
                WHERE id=?
            """, (status, reviewer, final, reason.strip() if reason else None, now, draft_id))
            if approve:
                connection.execute("""
                    INSERT INTO ticket_responses (id, ticket_id, draft_id, response, reviewer, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (str(uuid4()), row["ticket_id"], draft_id, final, reviewer, now))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get(draft_id)

    def ticket_responses(self, ticket_id: str) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT * FROM ticket_responses WHERE ticket_id=? ORDER BY created_at",
                                      (ticket_id,)).fetchall()
        return [dict(row) for row in rows]
