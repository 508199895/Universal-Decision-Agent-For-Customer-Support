"""One-time, stop-write migration from the legacy core DB to the agent DB."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from agentic.agents.agent import AgentSet
from agentic.data.tickets.ticket_store import TicketEventStore
from agentic.orchestration.workflow_state import initial_state
from agentic.orchestration.workflow import build_workflow


LEGACY_ACTIVE_STATUSES = {"open", "processing", "waiting_reply"}
LEGACY_STATUSES = LEGACY_ACTIVE_STATUSES | {"completed", "handed_off", "failed"}


@dataclass(frozen=True)
class LegacyTicket:
    ticket_id: str
    user_id: str
    account_id: str
    external_user_id: str
    user_name: str
    created_at: str
    updated_at: str
    status: str


@dataclass(frozen=True)
class LegacyMessage:
    ticket_id: str
    user_id: str
    message_id: str
    role: str
    content: str
    created_at: str
    rowid: int


def _readonly_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _backup_source(source: Path, target: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = target.with_name(f"{target.stem}.source-backup-{stamp}{target.suffix or '.db'}")
    source_connection = _readonly_connection(source)
    backup_connection = sqlite3.connect(backup)
    try:
        source_connection.backup(backup_connection)
    finally:
        backup_connection.close()
        source_connection.close()
    return backup


def _read_legacy(source: Path) -> tuple[list[LegacyTicket], list[LegacyMessage]]:
    connection = _readonly_connection(source)
    try:
        required = {"accounts", "users", "tickets", "ticket_metadata", "ticket_messages"}
        existing = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )}
        missing = required - existing
        if missing:
            raise RuntimeError(f"legacy database is missing tables: {sorted(missing)}")
        tickets: list[LegacyTicket] = []
        for row in connection.execute(
            """SELECT t.ticket_id, t.user_id, t.account_id, t.created_at,
                      u.external_user_id, u.user_name, u.account_id AS user_account_id,
                      a.account_id AS existing_account_id, m.status, m.updated_at
               FROM tickets AS t
               LEFT JOIN users AS u ON u.user_id = t.user_id
               LEFT JOIN accounts AS a ON a.account_id = t.account_id
               LEFT JOIN ticket_metadata AS m ON m.ticket_id = t.ticket_id
               ORDER BY t.created_at, t.ticket_id"""
        ):
            if (not row["external_user_id"] or row["existing_account_id"] is None
                    or row["user_account_id"] != row["account_id"]):
                raise RuntimeError(f"ticket {row['ticket_id']} has broken user/account associations")
            if row["status"] not in LEGACY_STATUSES:
                raise RuntimeError(f"ticket {row['ticket_id']} has unknown or missing status")
            tickets.append(LegacyTicket(
                row["ticket_id"], row["user_id"], row["account_id"],
                row["external_user_id"], row["user_name"], row["created_at"],
                row["updated_at"] or row["created_at"], row["status"],
            ))

        ticket_by_id = {ticket.ticket_id: ticket for ticket in tickets}
        messages: list[LegacyMessage] = []
        for row in connection.execute(
            """SELECT rowid AS source_rowid, message_id, ticket_id, role, content, created_at
               FROM ticket_messages ORDER BY created_at, rowid"""
        ):
            ticket = ticket_by_id.get(row["ticket_id"])
            if ticket is None:
                raise RuntimeError(f"message {row['message_id']} has no ticket")
            messages.append(LegacyMessage(
                ticket.ticket_id, ticket.user_id, row["message_id"], row["role"],
                row["content"] or "", row["created_at"], int(row["source_rowid"]),
            ))
        return tickets, messages
    finally:
        connection.close()


def _resolve_duplicate_active_tickets(tickets: list[LegacyTicket]) -> tuple[list[LegacyTicket], list[str]]:
    by_user: dict[str, list[LegacyTicket]] = {}
    for ticket in tickets:
        if ticket.status in LEGACY_ACTIVE_STATUSES:
            by_user.setdefault(ticket.user_id, []).append(ticket)
    completed_ids: set[str] = set()
    for user_tickets in by_user.values():
        if len(user_tickets) < 2:
            continue
        ordered = sorted(user_tickets, key=lambda item: (item.created_at, item.ticket_id))
        completed_ids.update(ticket.ticket_id for ticket in ordered[:-1])
    resolved = [
        LegacyTicket(
            ticket.ticket_id, ticket.user_id, ticket.account_id,
            ticket.external_user_id, ticket.user_name, ticket.created_at, ticket.updated_at,
            "completed" if ticket.ticket_id in completed_ids else ticket.status,
        )
        for ticket in tickets
    ]
    return resolved, sorted(completed_ids)


def _langchain_message(message: LegacyMessage):
    if message.role == "user":
        return HumanMessage(id=message.message_id, content=message.content)
    if message.role in {"agent", "ai"}:
        return AIMessage(id=message.message_id, content=message.content)
    if message.role == "system":
        return SystemMessage(id=message.message_id, content=message.content)
    raise RuntimeError(f"unsupported legacy message role {message.role!r}")


class _MigrationAgent:
    async def ainvoke(self, _payload: Any, **_kwargs):
        raise RuntimeError("the migration graph must never invoke an agent")


async def _seed_latest_active_checkpoints(
    tickets: list[LegacyTicket], messages: list[LegacyMessage],
    store: TicketEventStore, checkpointer: BaseCheckpointSaver,
) -> list[str]:
    graph = build_workflow(
        checkpointer=checkpointer, ticket_event_store=store,
        agents=AgentSet(escalation=_MigrationAgent()),
    )
    by_user: dict[str, list[LegacyTicket]] = {}
    messages_by_ticket: dict[str, list[LegacyMessage]] = {}
    for ticket in tickets:
        by_user.setdefault(ticket.user_id, []).append(ticket)
    for message in messages:
        messages_by_ticket.setdefault(message.ticket_id, []).append(message)

    seeded: list[str] = []
    for user_id, user_tickets in by_user.items():
        active_tickets = [
            ticket for ticket in user_tickets
            if ticket.status in LEGACY_ACTIVE_STATUSES
        ]
        if not active_tickets:
            continue
        latest = max(active_tickets, key=lambda item: (item.created_at, item.ticket_id))
        history = sorted(
            messages_by_ticket.get(latest.ticket_id, []),
            key=lambda item: (item.created_at, item.rowid),
        )
        if not history:
            raise RuntimeError(f"active ticket {latest.ticket_id} has no messages to seed")
        state_messages = [_langchain_message(item) for item in history]
        first_customer = next((item for item in state_messages if isinstance(item, HumanMessage)), None)
        last_customer = next((item for item in reversed(state_messages) if isinstance(item, HumanMessage)), None)
        if first_customer is None or last_customer is None:
            raise RuntimeError(f"active ticket {latest.ticket_id} has no customer message")
        state = initial_state(
            ticket_id=latest.ticket_id, owner_id=latest.external_user_id,
            owner_name=latest.user_name, message=first_customer,
        )
        state["messages"] = state_messages
        state["ticket"]["input"] = last_customer.content
        state["ticket"]["status"] = "processing"
        config = {"configurable": {"thread_id": user_id}}
        await graph.aupdate_state(config, state, as_node="__start__")
        store.mark_checkpoint_initialized(latest.ticket_id)
        store.set_status(latest.ticket_id, _target_status(latest.status))
        seeded.append(latest.ticket_id)
    return seeded


async def migrate_legacy_database(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source and target database paths must differ")
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"target database already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.stem}.migration{target.suffix or '.db'}")
    if staging.exists():
        raise FileExistsError(f"migration staging database already exists: {staging}")
    backup = _backup_source(source, target)
    try:
        tickets, messages = _read_legacy(backup)
        original_statuses = {ticket.ticket_id: ticket.status for ticket in tickets}
        tickets, duplicate_active_completed = _resolve_duplicate_active_tickets(tickets)
        store = TicketEventStore(staging)

        for ticket in tickets:
            store.create_ticket(
                ticket_id=ticket.ticket_id, user_id=ticket.user_id,
                status=_target_status(ticket.status), tokens=0,
                created_at=ticket.created_at, updated_at=ticket.updated_at,
                creation_metadata={
                    "source": "legacy_ticket",
                    "legacy_status": original_statuses[ticket.ticket_id],
                    "account_id": ticket.account_id,
                    "external_user_id": ticket.external_user_id,
                    "user_name": ticket.user_name,
                },
            )

        ticket_dates = {ticket.ticket_id: ticket.created_at for ticket in tickets}
        ordered_messages = sorted(
            messages,
            key=lambda item: (
                item.user_id, item.created_at, ticket_dates[item.ticket_id],
                item.ticket_id, item.rowid,
            ),
        )
        event_rows = []
        for item in ordered_messages:
            role = "human" if item.role == "user" else "ai" if item.role in {"agent", "ai"} else item.role
            event_rows.append({
                "user_id": item.user_id,
                "ticket_id": item.ticket_id,
                "event_type": "ticket.history.message",
                "category": "message",
                "content": {"type": role, "id": item.message_id, "content": item.content},
                "metadata": {"source": "legacy_ticket_messages", "legacy_role": item.role},
                "created_at": item.created_at,
            })
        store.append_events(event_rows)

        async with AsyncSqliteSaver.from_conn_string(str(staging)) as checkpointer:
            seeded = await _seed_latest_active_checkpoints(tickets, messages, store, checkpointer)

        connection = sqlite3.connect(str(staging))
        try:
            ticket_count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            event_count = connection.execute("SELECT COUNT(*) FROM ticket_events").fetchone()[0]
            history_message_count = connection.execute(
                "SELECT COUNT(*) FROM ticket_events WHERE event_type = 'ticket.history.message'",
            ).fetchone()[0]
            orphan_count = connection.execute(
                """SELECT COUNT(*) FROM ticket_events e LEFT JOIN tickets t
                   ON t.ticket_id = e.ticket_id AND t.user_id = e.user_id
                   WHERE t.ticket_id IS NULL""",
            ).fetchone()[0]
            for ticket in tickets:
                connection.execute(
                    "UPDATE tickets SET created_at = ?, updated_at = ? WHERE ticket_id = ?",
                    (ticket.created_at, ticket.updated_at, ticket.ticket_id),
                )
            connection.commit()
        finally:
            connection.close()
        if (ticket_count != len(tickets) or history_message_count != len(messages)
                or orphan_count):
            raise RuntimeError(
                f"migration verification failed: tickets={ticket_count}/{len(tickets)}, "
                f"history_messages={history_message_count}/{len(messages)}, orphans={orphan_count}"
            )
        staging.replace(target)
        return {
            "backup": str(backup),
            "target": str(target),
            "tickets_migrated": ticket_count,
            "messages_migrated": history_message_count,
            "events_migrated": event_count,
            "older_duplicate_active_tickets_completed": duplicate_active_completed,
            "latest_active_checkpoints_seeded": seeded,
            "verification": "passed",
        }
    except Exception:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{staging}{suffix}")
            if candidate.exists():
                candidate.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate the legacy ticket history to the new agent database. "
                    "Stop all writes to the source database before running.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    arguments = parser.parse_args()
    print(asyncio.run(migrate_legacy_database(arguments.source, arguments.target)))


def _target_status(legacy_status: str) -> str:
    if legacy_status in {"open", "processing", "waiting_reply"}:
        return "running"
    if legacy_status in {"handed_off", "completed"}:
        return "completed"
    return legacy_status


if __name__ == "__main__":
    main()
