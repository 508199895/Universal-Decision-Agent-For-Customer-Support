import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agentic.ticket_store import (
    TicketAlreadyTerminal,
    TicketHistoryConflict,
    TicketStore,
)
from agentic.v5_state import StateContractError, initial_state, merge_patch


def make_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE accounts (account_id TEXT PRIMARY KEY, account_name TEXT NOT NULL);
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY, account_id TEXT NOT NULL,
            external_user_id TEXT NOT NULL, user_name TEXT NOT NULL,
            UNIQUE (account_id, external_user_id),
            FOREIGN KEY(account_id) REFERENCES accounts(account_id)
        );
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, user_id TEXT NOT NULL,
            channel TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(account_id) REFERENCES accounts(account_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        CREATE TABLE ticket_metadata (
            ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL, main_issue_type TEXT,
            tags TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
        );
        CREATE TABLE ticket_messages (
            message_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
        );
        INSERT INTO accounts VALUES ('acct', 'Test');
        INSERT INTO users VALUES ('user-1', 'acct', 'external-1', 'One');
        INSERT INTO users VALUES ('user-2', 'acct', 'external-2', 'Two');
        """
    )
    connection.commit()
    connection.close()


def test_concurrent_entry_reuses_one_database_ticket(tmp_path):
    database = tmp_path / "core.sqlite"
    make_database(database)
    store = TicketStore(database)
    store.ensure_constraints()
    barrier = Barrier(2)

    def enter(index):
        barrier.wait()
        return TicketStore(database).acquire_or_create(
            "external-1", account_id="acct", channel=f"channel-{index}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(enter, range(2)))

    assert records[0].ticket_id == records[1].ticket_id
    assert sum(record.created for record in records) == 1
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM tickets WHERE user_id='user-1'"
    ).fetchone()[0] == 1
    connection.close()


def test_duplicate_active_history_is_reported_without_mutation(tmp_path):
    database = tmp_path / "core.sqlite"
    make_database(database)
    connection = sqlite3.connect(database)
    for ticket_id in ("old-1", "old-2"):
        connection.execute(
            "INSERT INTO tickets (ticket_id, account_id, user_id, channel) VALUES (?, 'acct', 'user-1', 'chat')",
            (ticket_id,),
        )
        connection.execute(
            "INSERT INTO ticket_metadata (ticket_id, status) VALUES (?, 'open')", (ticket_id,),
        )
    connection.commit()
    connection.close()

    store = TicketStore(database)
    with pytest.raises(TicketHistoryConflict, match="multiple active tickets"):
        store.acquire_or_create("external-1", account_id="acct")

    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT ticket_id, status FROM ticket_metadata ORDER BY ticket_id"
    ).fetchall() == [("old-1", "open"), ("old-2", "open")]
    connection.close()


def test_missing_metadata_and_unknown_status_are_rejected(tmp_path):
    database = tmp_path / "core.sqlite"
    make_database(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO tickets (ticket_id, account_id, user_id, channel) VALUES ('missing', 'acct', 'user-1', 'chat')"
    )
    connection.commit()
    connection.close()
    with pytest.raises(TicketHistoryConflict, match="no metadata"):
        TicketStore(database).acquire_or_create("external-1", account_id="acct")


def test_handoff_material_notification_and_terminal_status_are_atomic(tmp_path):
    from langchain_core.messages import HumanMessage

    database = tmp_path / "core.sqlite"
    make_database(database)
    store = TicketStore(database)
    ticket = store.acquire_or_create("external-1", account_id="acct")
    state = initial_state(
        ticket_id=ticket.ticket_id, owner_id="external-1", owner_name="One",
        message=HumanMessage(content="Need help with my booking", id="request-1"),
    )
    state["account"].update(status="unresolved", description="Cancellation result is unknown")
    state["budget"].update(outer_nodes_used=3, business_calls_used=1)
    state["ticket"]["status"] = "handed_off"
    state["escalation"].update(
        summary="Customer requested cancellation; the result remains unconfirmed.",
        description=(
            "Request: customer requested cancellation. Source: reservation status could not be confirmed. "
            "Attempt: account check. Result: cancellation remains unknown. Budget: one business call."
        ),
    )
    state["query"]["evidence"] = [{
        "source_type": "user_state", "source_ref": "reservation:r-1",
        "content": "Reservation status could not be confirmed",
    }]

    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TRIGGER fail_notice BEFORE INSERT ON ticket_messages
        WHEN NEW.content LIKE '%尚未确认人工已接手%'
        BEGIN SELECT RAISE(ABORT, 'injected message failure'); END
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(Exception, match="terminal commit failed"):
        store.commit_handoff(ticket.ticket_id, state=state)
    assert store.status(ticket.ticket_id) == "processing"
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM ticket_messages WHERE ticket_id=?", (ticket.ticket_id,),
    ).fetchone()[0] == 0
    connection.close()

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER fail_notice")
    connection.commit()
    connection.close()
    assert store.commit_handoff(ticket.ticket_id, state=state)
    assert not store.commit_handoff(ticket.ticket_id, state=state)
    assert store.status(ticket.ticket_id) == "handed_off"
    connection = sqlite3.connect(database)
    saved = connection.execute(
        "SELECT message_id, role, content FROM ticket_messages WHERE ticket_id=? ORDER BY message_id",
        (ticket.ticket_id,),
    ).fetchall()
    assert len(saved) == 2
    assert {row[1] for row in saved} == {"system"}
    assert any("交接摘要：Customer requested cancellation" in row[2] for row in saved)
    assert any("具体描述：Request: customer requested cancellation" in row[2] for row in saved)
    assert any("尚未确认人工已接手" in row[2] for row in saved)
    connection.close()
    with pytest.raises(TicketAlreadyTerminal):
        store.commit_completed(
            ticket.ticket_id,
            {"id": "different-final", "type": "ai", "content": "different delivery"},
        )


def test_message_ids_are_stable_and_duplicate_submission_is_idempotent(tmp_path):
    database = tmp_path / "core.sqlite"
    make_database(database)
    store = TicketStore(database)
    ticket = store.acquire_or_create("external-1", account_id="acct")
    message = {"id": "message-1", "type": "human", "content": "hello"}
    assert store.append_message(ticket.ticket_id, message) == "message-1"
    assert store.append_message(ticket.ticket_id, message) == "message-1"
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM ticket_messages WHERE message_id='message-1'"
    ).fetchone()[0] == 1
    connection.close()


def test_v5_state_contract_rejects_unknown_fields_and_unauthorized_writes():
    from langchain_core.messages import HumanMessage

    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )
    assert set(state) == {
        "ticket", "supervisor", "query", "clarify", "confirm", "account",
        "validate", "escalation", "budget",
    }
    with pytest.raises(StateContractError, match="unknown fields"):
        merge_patch(state, {"ticket": {"unexpected": "value"}}, writer="boundary")
    with pytest.raises(StateContractError, match="cannot update group ticket"):
        merge_patch(state, {"ticket": {"owner_id": "attacker"}}, writer="supervisor")
    updated = merge_patch(state, {"query": {"evidence": []}}, writer="query")
    assert updated["query"]["evidence"] == []
    assert updated["budget"] == state["budget"]


def test_v5_state_contract_rejects_implicit_types_and_invalid_enums():
    from langchain_core.messages import HumanMessage

    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )
    with pytest.raises(StateContractError, match="invalid supervisor.intent"):
        merge_patch(state, {"supervisor": {"intent": "DONE"}}, writer="supervisor")
    with pytest.raises(StateContractError, match="non-negative integer"):
        merge_patch(state, {"budget": {"outer_nodes_used": True}}, writer="boundary")


def test_state_patch_retains_untouched_fields_and_explicitly_clears_values():
    from langchain_core.messages import HumanMessage

    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )
    state["supervisor"]["query_instruction"] = "prior instruction"
    patched = merge_patch(
        state,
        {"supervisor": {"query_instruction": None}, "query": {"evidence": []}},
        writer="boundary",
    )
    assert patched["ticket"]["input"] == "hello"
    assert patched["supervisor"]["query_instruction"] is None
    assert patched["query"]["evidence"] == []


def test_deterministic_handoff_includes_required_context_and_fixed_notice():
    from langchain_core.messages import HumanMessage
    from agentic.handoff import HANDOFF_NOTICE, build_handoff_messages

    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="Cancel my reservation", id="message-1"),
    )
    state["account"].update(
        status="unresolved", pending_action={
            "tool_name": "cancel_reservation", "tool_args": {"reservation_id": "r-1"},
        }, description="Database connection failed after submit",
    )
    state["ticket"]["status"] = "handed_off"
    state["budget"].update(business_calls_used=2)
    state["escalation"].update(
        summary="Cancellation request remains unresolved.",
        description=(
            "Request: Cancel my reservation. Source: reservation lookup. Attempt: two business calls. "
            "Result: unresolved. Budget: business_calls_used=2."
        ),
    )
    first_material, first_notice = build_handoff_messages(state)
    second_material, second_notice = build_handoff_messages(state)
    assert first_material.content == second_material.content
    assert first_material.id == second_material.id
    assert first_notice.content == second_notice.content == HANDOFF_NOTICE
    assert "交接摘要：Cancellation request remains unresolved." in first_material.content
    assert "具体描述：Request: Cancel my reservation." in first_material.content
    assert "Cancel my reservation" in first_material.content
    assert "business_calls_used=2" in first_material.content


def test_handoff_rejects_incomplete_handoff_state(tmp_path):
    from langchain_core.messages import HumanMessage

    database = tmp_path / "core.sqlite"
    make_database(database)
    store = TicketStore(database)
    ticket = store.acquire_or_create("external-1", account_id="acct")
    state = initial_state(
        ticket_id=ticket.ticket_id, owner_id="external-1", owner_name="One",
        message=HumanMessage(content="Need help", id="request-1"),
    )
    with pytest.raises(ValueError, match="handed_off"):
        store.commit_handoff(ticket.ticket_id, state=state)
    assert store.status(ticket.ticket_id) == "processing"


def test_terminal_core_record_prevents_workflow_reexecution(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from agentic.workflow import aresume_ticket

    database = tmp_path / "core.sqlite"
    make_database(database)
    store = TicketStore(database)
    ticket = store.acquire_or_create("external-1", account_id="acct")
    store.commit_completed(
        ticket.ticket_id,
        AIMessage(content="Finished", id="final-1"),
    )

    class NeverInvoked:
        async def ainvoke(self, initial_state, config):
            raise AssertionError("terminal ticket reached the workflow")

    async def exercise():
        with pytest.raises(ValueError, match="ticket is terminal: completed"):
            await aresume_ticket(
                ticket.ticket_id, workflow=NeverInvoked(), ticket_store=store,
            )

    asyncio.run(exercise())

