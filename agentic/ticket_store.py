"""SQLite-backed ticket ownership, messages, and atomic terminal commits."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = ("open", "processing", "waiting_reply")
TERMINAL_STATUSES = {"completed", "handed_off"}
KNOWN_STATUSES = set(ACTIVE_STATUSES) | TERMINAL_STATUSES
PROCESSING_STATUSES = {"pending", "running", "completed", "failed"}


class TicketStoreError(RuntimeError):
    """Base error for persisted ticket operations."""


class TicketHistoryConflict(TicketStoreError):
    """Existing rows are ambiguous or inconsistent and need repair."""


class TicketPersistenceError(TicketStoreError):
    """A durable write failed; the caller must stop processing."""


class TicketAlreadyTerminal(TicketStoreError):
    """A terminal ticket cannot be executed or delivered again."""


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    user_id: str
    external_user_id: str
    account_id: str
    owner_name: str
    channel: str | None
    status: str
    created: bool


class TicketStore:
    """Use the existing UDA Hub tables and enforce active-ticket uniqueness in SQLite."""

    def __init__(self, database: str | Path, *, timeout: float = 10.0) -> None:
        self.database = str(database)
        self.timeout = timeout

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_constraints(self) -> None:
        """Install constraints and add overall-processing metadata to old databases."""
        connection = self._connect()
        try:
            self._require_core_schema(connection)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(ticket_metadata)")
            }
            if "processing_status" not in columns:
                # Existing rows may already have advanced execution. Do not replay them
                # from their original message during the schema migration.
                connection.execute(
                    "ALTER TABLE ticket_metadata ADD COLUMN processing_status "
                    "TEXT NOT NULL DEFAULT 'running'"
                )
            if "checkpoint_initialized" not in columns:
                connection.execute(
                    "ALTER TABLE ticket_metadata ADD COLUMN checkpoint_initialized "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            connection.execute(
                "UPDATE ticket_metadata SET processing_status='completed' "
                "WHERE status IN ('completed', 'handed_off') "
                "AND processing_status <> 'completed'"
            )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS uq_one_active_ticket_insert
                BEFORE INSERT ON ticket_metadata
                WHEN NEW.status IN ('open', 'processing', 'waiting_reply')
                BEGIN
                    SELECT CASE WHEN EXISTS (
                        SELECT 1
                        FROM tickets AS new_ticket
                        JOIN tickets AS existing_ticket
                          ON existing_ticket.user_id = new_ticket.user_id
                        JOIN ticket_metadata AS existing_metadata
                          ON existing_metadata.ticket_id = existing_ticket.ticket_id
                        WHERE new_ticket.ticket_id = NEW.ticket_id
                          AND existing_metadata.status IN ('open', 'processing', 'waiting_reply')
                    ) THEN RAISE(ABORT, 'active_ticket_exists') END;
                END;

                CREATE TRIGGER IF NOT EXISTS uq_one_active_ticket_update
                BEFORE UPDATE OF status ON ticket_metadata
                WHEN NEW.status IN ('open', 'processing', 'waiting_reply')
                BEGIN
                    SELECT CASE WHEN EXISTS (
                        SELECT 1
                        FROM tickets AS target_ticket
                        JOIN tickets AS existing_ticket
                          ON existing_ticket.user_id = target_ticket.user_id
                        JOIN ticket_metadata AS existing_metadata
                          ON existing_metadata.ticket_id = existing_ticket.ticket_id
                        WHERE target_ticket.ticket_id = NEW.ticket_id
                          AND existing_metadata.ticket_id <> NEW.ticket_id
                          AND existing_metadata.status IN ('open', 'processing', 'waiting_reply')
                    ) THEN RAISE(ABORT, 'active_ticket_exists') END;
                END;

                CREATE TABLE IF NOT EXISTS ticket_message_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_ticket_message_events_ticket
                    ON ticket_message_events (ticket_id, event_id);
                """
            )
        except sqlite3.Error as error:
            raise TicketPersistenceError(f"could not install active-ticket constraints: {error}") from error
        finally:
            connection.close()

    def acquire_or_create(
        self,
        external_user_id: str,
        *,
        account_id: str | None = None,
        channel: str | None = None,
        ticket_id: str | None = None,
        initial_message: Any | None = None,
    ) -> TicketRecord:
        if not isinstance(external_user_id, str) or not external_user_id:
            raise ValueError("external_user_id must be a non-empty string")
        if account_id is not None and (not isinstance(account_id, str) or not account_id):
            raise ValueError("account_id must be a non-empty string or null")
        initial_parts = _message_parts(initial_message, role="user") if initial_message is not None else None
        self.ensure_constraints()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            user = self._resolve_user(connection, external_user_id, account_id)
            rows = connection.execute(
                """
                SELECT t.ticket_id, t.user_id, t.account_id, t.channel,
                       m.ticket_id AS metadata_ticket_id, m.status, t.created_at
                FROM tickets AS t
                LEFT JOIN ticket_metadata AS m ON m.ticket_id = t.ticket_id
                WHERE t.user_id = ?
                ORDER BY t.created_at, t.ticket_id
                """,
                (user["user_id"],),
            ).fetchall()
            active = []
            for row in rows:
                if row["metadata_ticket_id"] is None:
                    raise TicketHistoryConflict(
                        f"ticket {row['ticket_id']} has no metadata; history was left unchanged"
                    )
                status = row["status"]
                if status not in KNOWN_STATUSES:
                    raise TicketHistoryConflict(
                        f"ticket {row['ticket_id']} has unknown status {status!r}"
                    )
                if status in ACTIVE_STATUSES:
                    active.append(row)
            if len(active) > 1:
                ids = ", ".join(row["ticket_id"] for row in active)
                raise TicketHistoryConflict(
                    f"user {external_user_id} has multiple active tickets: {ids}"
                )
            if active:
                row = active[0]
                connection.commit()
                return TicketRecord(
                    row["ticket_id"], row["user_id"], external_user_id,
                    row["account_id"], user["user_name"], row["channel"], row["status"], False,
                )

            new_ticket_id = ticket_id or str(uuid.uuid4())
            connection.execute(
                "INSERT INTO tickets (ticket_id, account_id, user_id, channel) VALUES (?, ?, ?, ?)",
                (new_ticket_id, user["account_id"], user["user_id"], channel),
            )
            try:
                connection.execute(
                    "INSERT INTO ticket_metadata "
                    "(ticket_id, status, processing_status, checkpoint_initialized) "
                    "VALUES (?, 'processing', 'pending', 0)",
                    (new_ticket_id,),
                )
            except sqlite3.IntegrityError as error:
                if "active_ticket_exists" in str(error):
                    # SQLite's write lock serializes competing entrants. Re-read
                    # after rollback in a fresh transaction to return the winner.
                    connection.rollback()
                    return self.acquire_or_create(
                        external_user_id, account_id=account_id, channel=channel,
                        initial_message=initial_message,
                    )
                raise
            if initial_parts is not None:
                self._insert_message(connection, new_ticket_id, *initial_parts)
            connection.commit()
            return TicketRecord(
                new_ticket_id, user["user_id"], external_user_id,
                user["account_id"], user["user_name"], channel, "processing", True,
            )
        except (TicketHistoryConflict, ValueError, TicketStoreError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket acquisition failed: {error}") from error
        finally:
            connection.close()

    def append_message(self, ticket_id: str, message: Any, *, role: str | None = None) -> str:
        message_id, db_role, content = _message_parts(message, role=role)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            status = self._ticket_status(connection, ticket_id)
            if status in TERMINAL_STATUSES:
                existing = self._message_exists(connection, message_id, ticket_id, db_role, content)
                if existing:
                    connection.commit()
                    return message_id
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is {status}")
            self._insert_message(connection, ticket_id, message_id, db_role, content)
            connection.commit()
            return message_id
        except TicketStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"message persistence failed: {error}") from error
        finally:
            connection.close()

    def append_customer_message(self, ticket_id: str, message: Any) -> str:
        """Persist a reply and mark the resumed ticket processing atomically."""
        message_id, db_role, content = _message_parts(message, role="user")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            status = self._ticket_status(connection, ticket_id)
            if status in TERMINAL_STATUSES:
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is {status}")
            self._insert_message(connection, ticket_id, message_id, db_role, content)
            updated = connection.execute(
                "UPDATE ticket_metadata SET status='processing', updated_at=CURRENT_TIMESTAMP WHERE ticket_id=?",
                (ticket_id,),
            )
            if updated.rowcount != 1:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata")
            connection.commit()
            return message_id
        except TicketStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"customer reply persistence failed: {error}") from error
        finally:
            connection.close()

    def set_workflow_status(self, ticket_id: str, status: str) -> None:
        if status not in {"processing", "waiting_reply"}:
            raise ValueError("workflow status must be processing or waiting_reply")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = self._ticket_status(connection, ticket_id)
            if current in TERMINAL_STATUSES:
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is {current}")
            updated = connection.execute(
                "UPDATE ticket_metadata SET status=?, updated_at=CURRENT_TIMESTAMP WHERE ticket_id=?",
                (status, ticket_id),
            )
            if updated.rowcount != 1:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata")
            connection.commit()
        except TicketStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket status persistence failed: {error}") from error
        finally:
            connection.close()

    def processing_status(self, ticket_id: str) -> str:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT processing_status FROM ticket_metadata WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata record")
            status = row["processing_status"]
            if status not in PROCESSING_STATUSES:
                raise TicketHistoryConflict(
                    f"ticket {ticket_id} has unknown processing status {status!r}"
                )
            return status
        finally:
            connection.close()

    def checkpoint_initialized(self, ticket_id: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT checkpoint_initialized FROM ticket_metadata WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata record")
            return bool(row["checkpoint_initialized"])
        finally:
            connection.close()

    def mark_checkpoint_initialized(self, ticket_id: str) -> None:
        self._set_processing_status(ticket_id, "running", checkpoint_initialized=True)

    def set_processing_status(self, ticket_id: str, status: str) -> None:
        if status not in PROCESSING_STATUSES:
            raise ValueError(f"invalid overall processing status: {status!r}")
        self._set_processing_status(ticket_id, status)

    def mark_processing_failed(self, ticket_id: str) -> None:
        """Record an error without overwriting an already completed ticket."""
        self._set_processing_status(ticket_id, "failed", only_if_nonterminal=True)

    def initial_customer_message(self, ticket_id: str):
        """Load the first customer message used to reconstruct a pending checkpoint."""
        from langchain_core.messages import HumanMessage

        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT message_id, content FROM ticket_messages "
                "WHERE ticket_id = ? AND role = 'user' ORDER BY rowid LIMIT 1",
                (ticket_id,),
            ).fetchone()
            if row is None:
                raise TicketHistoryConflict(
                    f"ticket {ticket_id} has no saved initial customer message"
                )
            return HumanMessage(id=row["message_id"], content=row["content"])
        finally:
            connection.close()

    def _set_processing_status(
        self, ticket_id: str, status: str, *, checkpoint_initialized: bool | None = None,
        only_if_nonterminal: bool = False,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = self._ticket_status(connection, ticket_id)
            if current in TERMINAL_STATUSES and status != "completed":
                connection.commit()
                return
            assignments = ["processing_status = ?", "updated_at = CURRENT_TIMESTAMP"]
            values: list[Any] = [status]
            if checkpoint_initialized is not None:
                assignments.append("checkpoint_initialized = ?")
                values.append(int(checkpoint_initialized))
            query = f"UPDATE ticket_metadata SET {', '.join(assignments)} WHERE ticket_id = ?"
            if only_if_nonterminal:
                query += " AND status NOT IN ('completed', 'handed_off')"
            values.append(ticket_id)
            updated = connection.execute(query, values)
            if updated.rowcount != 1:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata")
            connection.commit()
        except TicketStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"processing status persistence failed: {error}") from error
        finally:
            connection.close()

    def commit_completed(self, ticket_id: str, final_message: Any) -> bool:
        return self._commit_terminal(ticket_id, "completed", [(final_message, "agent")])

    def append_message_events(self, events: list[dict[str, Any]]) -> None:
        """Append one ticket-message event batch independently of checkpoints."""
        if not events:
            return
        normalized = []
        for event in events:
            if not isinstance(event, dict) or set(event) != {
                "ticket_id", "event_type", "content", "created_at",
            }:
                raise ValueError("ticket message event must use the declared event envelope")
            if (not isinstance(event["ticket_id"], str) or not event["ticket_id"]
                    or not isinstance(event["event_type"], str) or not event["event_type"]
                    or not isinstance(event["content"], dict)
                    or not isinstance(event["created_at"], str)):
                raise ValueError("run event fields are invalid")
            normalized.append((
                event["ticket_id"], event["event_type"],
                json.dumps(event["content"], ensure_ascii=False, sort_keys=True, default=str),
                event["created_at"],
            ))

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """INSERT INTO ticket_message_events
                   (ticket_id, event_type, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                normalized,
            )
            connection.commit()
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"ticket message event batch persistence failed: {error}") from error
        finally:
            connection.close()

    def message_events(self, ticket_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            query = """SELECT event_id, ticket_id, event_type, content, created_at
                       FROM ticket_message_events WHERE ticket_id = ? ORDER BY event_id"""
            return [
                {**dict(row), "content": json.loads(row["content"])}
                for row in connection.execute(query, (ticket_id,))
            ]
        finally:
            connection.close()

    def commit_handoff(self, ticket_id: str, *, state: dict[str, Any]) -> bool:
        """Atomically save internal handoff material, public notice, and terminal status."""
        from agentic.handoff import build_handoff_messages

        handoff_material, notification = build_handoff_messages(state)
        if state.get("ticket", {}).get("ticket_id") != ticket_id:
            raise TicketHistoryConflict("handoff state ticket_id does not match the target ticket")
        return self._commit_terminal(
            ticket_id, "handed_off",
            [(handoff_material, "system"), (notification, "system")],
        )

    def _commit_terminal(self, ticket_id: str, status: str, messages: list[tuple[Any, str]]) -> bool:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        normalized = [_message_parts(message, role=role) for message, role in messages]
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = self._ticket_status(connection, ticket_id)
            if current in TERMINAL_STATUSES:
                all_present = current == status and all(
                    self._message_exists(connection, message_id, ticket_id, role, content)
                    for message_id, role, content in normalized
                )
                if all_present:
                    connection.commit()
                    return False
                raise TicketAlreadyTerminal(f"ticket {ticket_id} is already {current}")
            for message_id, role, content in normalized:
                self._insert_message(connection, ticket_id, message_id, role, content)
            updated = connection.execute(
                "UPDATE ticket_metadata SET status = ?, processing_status='completed', "
                "updated_at = CURRENT_TIMESTAMP WHERE ticket_id = ?",
                (status, ticket_id),
            )
            if updated.rowcount != 1:
                raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata")
            connection.commit()
            return True
        except TicketStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise TicketPersistenceError(f"terminal commit failed: {error}") from error
        finally:
            connection.close()

    def status(self, ticket_id: str) -> str:
        connection = self._connect()
        try:
            return self._ticket_status(connection, ticket_id)
        finally:
            connection.close()

    @staticmethod
    def thread_id(ticket_id: str) -> str:
        if not isinstance(ticket_id, str) or not ticket_id:
            raise ValueError("ticket_id must be a non-empty string")
        return ticket_id

    @staticmethod
    def _require_core_schema(connection: sqlite3.Connection) -> None:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not {"users", "tickets", "ticket_metadata", "ticket_messages"}.issubset(tables):
            raise TicketPersistenceError("core ticket schema is missing required tables")

    @staticmethod
    def _resolve_user(
        connection: sqlite3.Connection, external_user_id: str, account_id: str | None,
    ) -> sqlite3.Row:
        query = "SELECT user_id, account_id, user_name FROM users WHERE external_user_id = ?"
        args: tuple[Any, ...] = (external_user_id,)
        if account_id is not None:
            query += " AND account_id = ?"
            args += (account_id,)
        rows = connection.execute(query, args).fetchall()
        if not rows:
            raise TicketHistoryConflict(f"no core user maps to external_user_id {external_user_id!r}")
        if len(rows) != 1:
            raise TicketHistoryConflict(
                f"external_user_id {external_user_id!r} maps to multiple core users; specify account_id"
            )
        return rows[0]

    @staticmethod
    def _ticket_status(connection: sqlite3.Connection, ticket_id: str) -> str:
        row = connection.execute(
            "SELECT status FROM ticket_metadata WHERE ticket_id = ?", (ticket_id,),
        ).fetchone()
        if row is None:
            raise TicketHistoryConflict(f"ticket {ticket_id} has no metadata record")
        if row["status"] not in KNOWN_STATUSES:
            raise TicketHistoryConflict(f"ticket {ticket_id} has unknown status {row['status']!r}")
        return row["status"]

    @staticmethod
    def _message_exists(
        connection: sqlite3.Connection, message_id: str, ticket_id: str,
        role: str, content: str,
    ) -> bool:
        row = connection.execute(
            "SELECT ticket_id, role, content FROM ticket_messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return False
        if (row["ticket_id"], row["role"], row["content"]) != (ticket_id, role, content):
            raise TicketHistoryConflict(f"native message id {message_id!r} collides with different content")
        return True

    def _insert_message(
        self, connection: sqlite3.Connection, ticket_id: str,
        message_id: str, role: str, content: str,
    ) -> None:
        if self._message_exists(connection, message_id, ticket_id, role, content):
            return
        connection.execute(
            "INSERT INTO ticket_messages (message_id, ticket_id, role, content) VALUES (?, ?, ?, ?)",
            (message_id, ticket_id, role, content),
        )


def _message_parts(message: Any, *, role: str | None) -> tuple[str, str, str]:
    if isinstance(message, dict):
        message_id = message.get("id")
        message_role = role or message.get("type") or message.get("role")
        content = message.get("content", "")
    else:
        message_id = getattr(message, "id", None)
        message_role = role or getattr(message, "type", None)
        content = getattr(message, "content", "")
    if not isinstance(message_id, str) or not message_id.strip():
        raise ValueError("persisted native messages must have a stable non-empty id")
    role_map = {
        "human": "user", "user": "user", "ai": "agent", "assistant": "agent",
        "agent": "agent", "tool": "agent", "system": "system",
    }
    db_role = role_map.get(str(message_role).lower())
    if db_role is None:
        raise ValueError(f"unsupported message role: {message_role!r}")
    if isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return message_id, db_role, text
