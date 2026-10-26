"""SQLite persistence for minimal tickets and their append-only event stream."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"completed"}
KNOWN_STATUSES = {"pending", "running", "completed", "failed"}
CATEGORIES = {"message", "trace", "outputs", "error", "middleware"}


class TicketEventStoreError(RuntimeError):
    """Base error for persisted ticket operations."""


class TicketHistoryConflict(TicketEventStoreError):
    """Persisted ticket state conflicts with the requested operation."""


class TicketPersistenceError(TicketEventStoreError):
    """A durable write failed; processing must stop."""


class TicketAlreadyTerminal(TicketEventStoreError):
    """A terminal ticket cannot be executed or delivered again."""


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    user_id: str
    status: str
    tokens: int
    input_tokens: int
    output_tokens: int
    llm_call_count: int
    created_at: str
    updated_at: str
    created: bool


class TicketEventStore:
    """Persist ticket state, model usage and user-scoped append-only events."""

    def __init__(self, database: str | Path, *, timeout: float = 10.0) -> None:
        self.database = str(database)
        self.timeout = timeout
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_schema(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                    tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0),
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    llm_call_count INTEGER NOT NULL DEFAULT 0 CHECK (llm_call_count >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_tickets_user
                    ON tickets (user_id, ticket_id);

                CREATE TABLE IF NOT EXISTS ticket_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (category IN
                        ('message', 'trace', 'outputs', 'error', 'middleware')),
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE (user_id, seq),
                    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
                );

                CREATE INDEX IF NOT EXISTS ix_ticket_events_ticket
                    ON ticket_events (user_id, ticket_id, seq);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tickets)")
            }
            for name in ("input_tokens", "output_tokens", "llm_call_count"):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE tickets ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
        except sqlite3.Error as error:
            raise TicketPersistenceError(f"could not initialize agent ticket database: {error}") from error
        finally:
            connection.close()

    def create_ticket(
        self, *, ticket_id: str, user_id: str, status: str = "pending",
        tokens: int = 0, created_at: str | None = None, updated_at: str | None = None,
        initial_message: dict[str, Any] | None = None,
        initial_metadata: dict[str, Any] | None = None,
        creation_metadata: dict[str, Any] | None = None,
    ) -> TicketRecord:
        if not ticket_id or not user_id:
            raise ValueError("ticket_id and user_id must be non-empty")
        if status not in KNOWN_STATUSES or type(tokens) is not int or tokens < 0:
            raise ValueError("ticket status or tokens is invalid")
        now = _utc_now()
        created_at = created_at or now
        updated_at = updated_at or created_at
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,),
            ).fetchone()
            if row is not None:
                if row["user_id"] != user_id:
                    raise TicketHistoryConflict(f"ticket {ticket_id} already belongs to another user")
                connection.commit()
                return _ticket_record(row, created=False)
            connection.execute(
                """INSERT INTO tickets
                   (ticket_id, user_id, status, tokens, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticket_id, user_id, status, tokens, created_at, updated_at),
            )
            self._append_event_in_transaction(
                connection, user_id=user_id, ticket_id=ticket_id,
                event_type="ticket.created", category="trace",
                content={"ticket_id": ticket_id}, metadata=creation_metadata or {}, created_at=now,
            )
            if initial_message is not None:
                _append_initial_message(
                    connection, user_id, ticket_id, initial_message,
                    metadata=initial_metadata or {},
                )
            connection.commit()
            return TicketRecord(
                ticket_id, user_id, status, tokens, 0, 0, 0, created_at, updated_at, True,
            )
        except TicketEventStoreError:
            connection.rollback()
            raise
        except (sqlite3.Error, ValueError) as error:
            connection.rollback()
            if isinstance(error, ValueError):
                raise
            raise TicketPersistenceError(f"ticket creation failed: {error}") from error
        finally:
            connection.close()

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if row is None:
                raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
            if row["status"] not in KNOWN_STATUSES or row["tokens"] < 0:
                raise TicketHistoryConflict(f"ticket {ticket_id} has invalid stored fields")
            return _ticket_record(row, created=False)
        finally:
            connection.close()

    def latest_uninitialized_ticket(self, user_id: str) -> TicketRecord | None:
        """Find a created, nonterminal ticket without its checkpoint event."""
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT t.* FROM tickets t JOIN ticket_events created
                     ON created.ticket_id = t.ticket_id AND created.event_type = 'ticket.created'
                   WHERE t.user_id = ? AND t.status = 'pending'
                     AND NOT EXISTS (
                         SELECT 1 FROM ticket_events initialized
                         WHERE initialized.ticket_id = t.ticket_id
                           AND initialized.event_type = 'checkpoint.initialized'
                     )
                   ORDER BY created.event_id DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
            return _ticket_record(row, created=False) if row is not None else None
        finally:
            connection.close()

    def checkpoint_initialized(self, ticket_id: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT 1 FROM ticket_events WHERE ticket_id = ?
                   AND event_type = 'checkpoint.initialized' LIMIT 1""",
                (ticket_id,),
            ).fetchone()
            if row is None:
                exists = connection.execute(
                    "SELECT 1 FROM tickets WHERE ticket_id = ?", (ticket_id,),
                ).fetchone()
                if exists is None:
                    raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
            return row is not None
        finally:
            connection.close()

    def latest_checkpoint_ticket(self, user_id: str) -> TicketRecord | None:
        """Return the ticket associated with the user's most recent initialized checkpoint."""
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT t.* FROM tickets t JOIN ticket_events initialized
                     ON initialized.ticket_id = t.ticket_id
                    AND initialized.event_type = 'checkpoint.initialized'
                   WHERE t.user_id = ?
                   ORDER BY initialized.event_id DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
            return _ticket_record(row, created=False) if row is not None else None
        finally:
            connection.close()

    def mark_checkpoint_initialized(self, ticket_id: str) -> None:
        user_id = self._ticket_user(ticket_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                """SELECT 1 FROM ticket_events WHERE ticket_id = ?
                   AND event_type = 'checkpoint.initialized' LIMIT 1""",
                (ticket_id,),
            ).fetchone()
            if exists is None:
                self._append_event_in_transaction(
                    connection, user_id=user_id, ticket_id=ticket_id,
                    event_type="checkpoint.initialized", category="trace",
                    content={"ticket_id": ticket_id}, metadata={}, created_at=_utc_now(),
                )
                connection.execute(
                    "UPDATE tickets SET updated_at = ? WHERE ticket_id = ?",
                    (_utc_now(), ticket_id),
                )
            connection.commit()
        except TicketEventStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"checkpoint marker persistence failed: {error}") from error
        finally:
            connection.close()

    def set_status(self, ticket_id: str, status: str) -> None:
        if status not in KNOWN_STATUSES:
            raise ValueError(f"invalid ticket status: {status!r}")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if row is None:
                raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
            if row["status"] in TERMINAL_STATUSES and status != row["status"]:
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is {row['status']}")
            connection.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
                (status, _utc_now(), ticket_id),
            )
            connection.commit()
        except TicketEventStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket status persistence failed: {error}") from error
        finally:
            connection.close()

    def status(self, ticket_id: str) -> str:
        return self.get_ticket(ticket_id).status

    def save_token_usage(self, ticket_id: str, *, input_tokens: int, output_tokens: int,
                         total_tokens: int, llm_call_count: int) -> None:
        values = (input_tokens, output_tokens, total_tokens, llm_call_count)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("token usage values must be non-negative integers")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE tickets SET input_tokens = ?, output_tokens = ?, tokens = ?,
                   llm_call_count = ?, updated_at = ? WHERE ticket_id = ?""",
                (*values[:3], values[3], _utc_now(), ticket_id),
            )
            if cursor.rowcount != 1:
                raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
            connection.commit()
        except TicketEventStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket token usage persistence failed: {error}") from error
        finally:
            connection.close()

    def commit_handoff(self, ticket_id: str, *, state: dict[str, Any], final_message: str) -> bool:
        if state.get("ticket", {}).get("ticket_id") != ticket_id:
            raise TicketHistoryConflict("handoff state ticket_id does not match the target ticket")
        notice_id = f"handoff:{ticket_id}:notice"
        state_notice = next((
            message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            for message in state.get("messages", [])
            if (message.get("id") if isinstance(message, dict) else getattr(message, "id", None)) == notice_id
        ), None)
        if not isinstance(final_message, str) or not final_message.strip() or state_notice != final_message:
            raise TicketHistoryConflict("handoff state does not match its customer notice")
        # Handoff is a WorkflowState.ticket.status; the ticket processing itself is complete.
        return self._commit_terminal(ticket_id, "completed", final_message=final_message)

    def commit_completed(self, ticket_id: str, final_message: Any = None) -> bool:
        if not isinstance(final_message, str) or not final_message.strip():
            raise TicketHistoryConflict("completed ticket is missing its customer-facing final message")
        return self._commit_terminal(ticket_id, "completed", final_message=final_message)

    def _commit_terminal(self, ticket_id: str, status: str, *, final_message: Any = None) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if row is None:
                raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
            if row["status"] in TERMINAL_STATUSES:
                if row["status"] == status:
                    connection.commit()
                    return False
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is already {row['status']}")
            connection.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
                (status, _utc_now(), ticket_id),
            )
            self._append_event_in_transaction(
                connection, user_id=self._ticket_user_in_transaction(connection, ticket_id),
                ticket_id=ticket_id, event_type="ticket.completed", category="outputs",
                content={"message": final_message} if final_message is not None else {},
                metadata={}, created_at=_utc_now(),
            )
            connection.commit()
            return True
        except TicketEventStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"terminal status persistence failed: {error}") from error
        finally:
            connection.close()

    def append_events(self, events: list[dict[str, Any]]) -> None:
        """Append a batch with a strictly increasing sequence per user."""
        normalized = []
        required = {"user_id", "ticket_id", "event_type", "category", "content", "metadata", "created_at"}
        for event in events:
            if not isinstance(event, dict) or set(event) != required:
                raise ValueError("event must use the declared ticket event envelope")
            if any(not isinstance(event[key], str) or not event[key] for key in (
                "user_id", "ticket_id", "event_type", "category", "created_at",
            )):
                raise ValueError("event identifiers, type, category, and timestamp must be non-empty strings")
            if event["category"] not in CATEGORIES:
                raise ValueError(f"unknown event category: {event['category']!r}")
            if not isinstance(event["metadata"], dict):
                raise ValueError("event metadata must be an object")
            normalized.append(event)
        if not normalized:
            return

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            next_seq: dict[str, int] = {}
            for event in normalized:
                ticket_user = self._ticket_user_in_transaction(connection, event["ticket_id"])
                if ticket_user != event["user_id"]:
                    raise TicketHistoryConflict(
                        f"event ticket {event['ticket_id']} does not belong to user {event['user_id']}"
                    )
                if event["user_id"] not in next_seq:
                    row = connection.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM ticket_events WHERE user_id = ?",
                        (event["user_id"],),
                    ).fetchone()
                    next_seq[event["user_id"]] = int(row["next_seq"])
                self._append_event_in_transaction(
                    connection, **event, seq=next_seq[event["user_id"]],
                )
                next_seq[event["user_id"]] += 1
            touched = {(event["ticket_id"]) for event in normalized}
            for ticket_id in touched:
                connection.execute(
                    "UPDATE tickets SET updated_at = ? WHERE ticket_id = ?",
                    (_utc_now(), ticket_id),
                )
            connection.commit()
        except TicketEventStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket event batch persistence failed: {error}") from error
        finally:
            connection.close()

    def events(self, *, user_id: str, ticket_id: str | None = None) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            query = "SELECT * FROM ticket_events WHERE user_id = ?"
            args: tuple[Any, ...] = (user_id,)
            if ticket_id is not None:
                query += " AND ticket_id = ?"
                args += (ticket_id,)
            query += " ORDER BY seq"
            return [
                {**dict(row), "content": json.loads(row["content"]),
                 "metadata": json.loads(row["metadata"])}
                for row in connection.execute(query, args)
            ]
        finally:
            connection.close()

    def _ticket_user(self, ticket_id: str) -> str:
        connection = self._connect()
        try:
            return self._ticket_user_in_transaction(connection, ticket_id)
        finally:
            connection.close()

    @staticmethod
    def _ticket_user_in_transaction(connection: sqlite3.Connection, ticket_id: str) -> str:
        row = connection.execute("SELECT user_id FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise TicketHistoryConflict(f"ticket {ticket_id} does not exist")
        return row["user_id"]

    @staticmethod
    def _append_event_in_transaction(
        connection: sqlite3.Connection, *, user_id: str, ticket_id: str,
        event_type: str, category: str, content: Any, metadata: dict[str, Any],
        created_at: str, seq: int | None = None,
    ) -> None:
        if seq is None:
            row = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM ticket_events WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            seq = int(row["next_seq"])
        connection.execute(
            """INSERT INTO ticket_events
               (user_id, ticket_id, seq, event_type, category, content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, ticket_id, seq, event_type, category, _json_text(content),
             _json_text(metadata), created_at),
        )

    @staticmethod
    def thread_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        return user_id


def _ticket_record(row: sqlite3.Row, *, created: bool) -> TicketRecord:
    return TicketRecord(
        row["ticket_id"], row["user_id"], row["status"], row["tokens"],
        row["input_tokens"], row["output_tokens"], row["llm_call_count"],
        row["created_at"], row["updated_at"], created,
    )


def _append_initial_message(
    connection: sqlite3.Connection, user_id: str, ticket_id: str,
    message: dict[str, Any], *, metadata: dict[str, Any],
) -> None:
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("initial_message must be a serialized message with string content")
    TicketEventStore._append_event_in_transaction(
        connection, user_id=user_id, ticket_id=ticket_id,
        event_type="ticket.message.received", category="message",
        content=message, metadata={"source": "ticket_entry", **metadata},
        created_at=_utc_now(),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
