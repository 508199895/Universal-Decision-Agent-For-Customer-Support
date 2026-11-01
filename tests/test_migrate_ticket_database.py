import asyncio
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from scripts.migrate_ticket_database import migrate_legacy_database
from agentic.data.tickets.TicketStore import TicketStore
from agentic.orchestration.workflow.graph import build_workflow


def _legacy_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE accounts (account_id TEXT PRIMARY KEY, account_name TEXT NOT NULL);
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY, account_id TEXT NOT NULL,
            external_user_id TEXT NOT NULL, user_name TEXT NOT NULL
        );
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, user_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE ticket_metadata (
            ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE ticket_messages (
            message_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        INSERT INTO accounts VALUES ('acct-a', 'Account A'), ('acct-b', 'Account B');
        INSERT INTO users VALUES
            ('user-a', 'acct-a', 'external-a', 'Alice'),
            ('user-b', 'acct-b', 'external-b', 'Bob');

        INSERT INTO tickets VALUES
            ('a-active', 'acct-a', 'user-a', '2026-01-01T00:00:00Z'),
            ('a-newer-completed', 'acct-a', 'user-a', '2026-01-02T00:00:00Z'),
            ('b-old-active', 'acct-b', 'user-b', '2026-01-01T00:00:00Z'),
            ('b-latest-active', 'acct-b', 'user-b', '2026-01-03T00:00:00Z');
        INSERT INTO ticket_metadata VALUES
            ('a-active', 'open', '2026-01-01T00:10:00Z'),
            ('a-newer-completed', 'completed', '2026-01-02T00:10:00Z'),
            ('b-old-active', 'open', '2026-01-01T00:10:00Z'),
            ('b-latest-active', 'waiting_reply', '2026-01-03T00:10:00Z');

        INSERT INTO ticket_messages VALUES
            ('a-m1', 'a-active', 'user', 'A first', '2026-01-01T00:01:00Z'),
            ('a-m2', 'a-active', 'agent', 'A reply', '2026-01-01T00:02:00Z'),
            ('a-m3', 'a-newer-completed', 'user', 'A later ticket', '2026-01-02T00:01:00Z'),
            ('b-m1', 'b-old-active', 'user', 'B old', '2026-01-01T00:01:00Z'),
            ('b-m2', 'b-latest-active', 'user', 'B latest', '2026-01-03T00:01:00Z');
        """
    )
    connection.commit()
    connection.close()


def test_migration_preserves_history_and_seeds_latest_active_user_checkpoints(tmp_path):
    source = tmp_path / "legacy.sqlite"
    target = tmp_path / "agent.sqlite"
    _legacy_database(source)

    report = asyncio.run(migrate_legacy_database(source, target))

    assert report["verification"] == "passed"
    assert report["tickets_migrated"] == 4
    assert report["messages_migrated"] == 5
    assert set(report["latest_active_checkpoints_seeded"]) == {
        "a-active", "b-latest-active",
    }
    assert len(report["older_duplicate_active_tickets_completed"]) == 1
    assert target.exists()
    assert Path(report["backup"]).is_file()
    source_connection = sqlite3.connect(source)
    assert source_connection.execute(
        "SELECT status FROM ticket_metadata WHERE ticket_id='b-old-active'",
    ).fetchone()[0] == "open"
    source_connection.close()

    connection = sqlite3.connect(target)
    ticket_columns = [row[1] for row in connection.execute("PRAGMA table_info(tickets)")]
    assert ticket_columns == [
        "ticket_id", "user_id", "status", "tokens", "input_tokens", "output_tokens",
        "llm_call_count", "created_at", "updated_at",
    ]
    assert connection.execute("SELECT COUNT(*) FROM ticket_events").fetchone()[0] == 11
    connection.close()

    store = TicketStore(target)
    assert store.status("a-active") == "running"
    assert store.status("a-newer-completed") == "completed"
    assert store.status("b-old-active") == "completed"
    assert store.status("b-latest-active") == "running"
    old_ticket_created = next(
        event for event in store.event_store.events(user_id="user-b", ticket_id="b-old-active")
        if event["event_type"] == "ticket.created"
    )
    assert old_ticket_created["metadata"]["legacy_status"] == "open"
    assert old_ticket_created["metadata"]["account_id"] == "acct-b"
    assert [
        event["content"]["id"]
        for event in store.event_store.events(user_id="user-a", ticket_id="a-active")
        if event["event_type"] == "ticket.history.message"
    ] == ["a-m1", "a-m2"]

    async def read_checkpoints():
        async with AsyncSqliteSaver.from_conn_string(str(target)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer,
                escalation_agent=object(),
            )
            states = {}
            for user_id in ("user-a", "user-b"):
                snapshot = await graph.aget_state({"configurable": {"thread_id": user_id}})
                states[user_id] = snapshot.values
            return states

    states = asyncio.run(read_checkpoints())
    assert states["user-a"]["ticket"]["ticket_id"] == "a-active"
    assert [message.id for message in states["user-a"]["messages"]] == ["a-m1", "a-m2"]
    assert states["user-b"]["ticket"]["ticket_id"] == "b-latest-active"
    assert [message.id for message in states["user-b"]["messages"]] == ["b-m2"]


def test_migration_refuses_existing_target_without_overwriting_it(tmp_path):
    source = tmp_path / "legacy.sqlite"
    target = tmp_path / "agent.sqlite"
    _legacy_database(source)
    target.write_text("keep", encoding="utf-8")

    try:
        asyncio.run(migrate_legacy_database(source, target))
    except FileExistsError:
        pass
    else:
        raise AssertionError("migration must not overwrite an existing target")

    assert target.read_text(encoding="utf-8") == "keep"
