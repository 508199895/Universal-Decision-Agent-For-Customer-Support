import asyncio
import sqlite3

import pytest
from langchain_core.messages import HumanMessage

from agentic.ticket_store import TicketPersistenceError, TicketStore
from agentic.v5_state import initial_state
from agentic.workflow import aenter_ticket, persistent_workflow


def create_core_ticket_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE accounts (account_id TEXT PRIMARY KEY, account_name TEXT NOT NULL);
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY, account_id TEXT NOT NULL,
            external_user_id TEXT NOT NULL, user_name TEXT NOT NULL,
            UNIQUE (account_id, external_user_id)
        );
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, user_id TEXT NOT NULL,
            channel TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE ticket_metadata (
            ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL, main_issue_type TEXT,
            tags TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE ticket_messages (
            message_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO accounts VALUES ('acct', 'Test');
        INSERT INTO users VALUES ('user-1', 'acct', 'external-1', 'One');
        """
    )
    connection.commit()
    connection.close()


class HandoffAgent:
    def __init__(self):
        self.payload = None

    async def ainvoke(self, payload, config=None):
        self.payload = payload
        return {
            "summary": "客户请求取消预订，处理结果尚未确认。",
            "description": "对话记录了取消请求；当前无法确认取消是否提交成功。",
        }


def test_v5_handoff_graph_summarizes_dialogue_and_commits_in_same_database(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_ticket_db(database)
    store = TicketStore(database)
    agent = HandoffAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    async def exercise():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            result = await aenter_ticket(
                "external-1", message, account_id="acct", channel="chat",
                workflow=workflow, ticket_store=store, ticket_id="ticket-1",
            )
            snapshot = await workflow.aget_state(
                {"configurable": {"thread_id": "ticket-1"}}
            )
            return result, snapshot

    result, snapshot = asyncio.run(exercise())

    assert set(result["escalation"]) == {"summary", "description"}
    assert result["ticket"]["status"] == "handed_off"
    assert "[human] 请取消我的预订。" in agent.payload["conversation"]
    assert set(snapshot.values) == {
        "ticket", "supervisor", "query", "clarify", "confirm",
        "account", "validate", "escalation", "budget",
    }
    checkpoint_messages = snapshot.values["ticket"]["messages"]
    assert [(item.id, item.type) for item in checkpoint_messages] == [
        ("customer-1", "human"),
        ("handoff:ticket-1:material", "system"),
        ("handoff:ticket-1:notice", "system"),
    ]
    assert store.status("ticket-1") == "handed_off"

    connection = sqlite3.connect(database)
    saved = connection.execute(
        "SELECT message_id, role, content FROM ticket_messages WHERE ticket_id=? ORDER BY rowid",
        ("ticket-1",),
    ).fetchall()
    connection.close()
    assert len(saved) == 3  # customer input, internal material, fixed customer notice
    assert saved == [
        (item.id, {"human": "user", "system": "system"}[item.type], item.content)
        for item in checkpoint_messages
    ]
    assert "交接摘要：客户请求取消预订" in saved[1][2]
    assert "具体描述：对话记录了取消请求" in saved[1][2]
    assert "尚未确认人工已接手" in saved[2][2]


def test_handoff_save_failure_preserves_nonterminal_ticket(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_ticket_db(database)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TRIGGER fail_handoff_notice BEFORE INSERT ON ticket_messages
        WHEN NEW.content LIKE '%尚未确认人工已接手%'
        BEGIN SELECT RAISE(ABORT, 'injected notice failure'); END
        """
    )
    connection.commit()
    connection.close()

    store = TicketStore(database)
    agent = HandoffAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    async def exercise():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            with pytest.raises(TicketPersistenceError, match="terminal commit failed"):
                await aenter_ticket(
                    "external-1", message, account_id="acct", channel="chat",
                    workflow=workflow, ticket_store=store, ticket_id="ticket-1",
                )

    asyncio.run(exercise())

    assert store.status("ticket-1") == "processing"
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT role, content FROM ticket_messages WHERE ticket_id=?", ("ticket-1",),
    ).fetchall()
    processing_status = connection.execute(
        "SELECT processing_status FROM ticket_metadata WHERE ticket_id='ticket-1'"
    ).fetchone()[0]
    connection.close()
    assert rows == [("user", "请取消我的预订。")]
    assert processing_status == "failed"


def test_failed_core_commit_retries_from_handoff_checkpoint_without_agent_call(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_ticket_db(database)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TRIGGER fail_handoff_notice BEFORE INSERT ON ticket_messages
        WHEN NEW.content LIKE '%尚未确认人工已接手%'
        BEGIN SELECT RAISE(ABORT, 'injected notice failure'); END
        """
    )
    connection.commit()
    connection.close()

    store = TicketStore(database)
    class CountingAgent(HandoffAgent):
        calls = 0

        async def ainvoke(self, payload, config=None):
            self.calls += 1
            return await super().ainvoke(payload)

    agent = CountingAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    async def enter():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            return await aenter_ticket(
                "external-1", message, account_id="acct", channel="chat",
                workflow=workflow, ticket_store=store, ticket_id="ticket-1",
            )

    with pytest.raises(TicketPersistenceError, match="terminal commit failed"):
        asyncio.run(enter())

    assert store.status("ticket-1") == "processing"
    async def inspect_checkpoint():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            snapshot = await workflow.aget_state(
                {"configurable": {"thread_id": "ticket-1"}}
            )
            return snapshot.values

    checkpoint = asyncio.run(inspect_checkpoint())
    assert checkpoint["ticket"]["status"] == "handed_off"
    assert set(checkpoint["escalation"]) == {"summary", "description"}
    assert [item.id for item in checkpoint["ticket"]["messages"]] == [
        "customer-1", "handoff:ticket-1:material", "handoff:ticket-1:notice",
    ]

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER fail_handoff_notice")
    connection.commit()
    connection.close()
    result = asyncio.run(enter())

    assert result["ticket"]["status"] == "handed_off"
    assert store.status("ticket-1") == "handed_off"
    assert agent.calls == 1
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM tickets WHERE ticket_id='ticket-1'"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM ticket_messages WHERE ticket_id='ticket-1'"
    ).fetchone()[0] == 3
    connection.close()


def test_failed_handoff_resumes_from_saved_node_without_duplicate_ticket(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_ticket_db(database)
    store = TicketStore(database)

    class FailOnceAgent(HandoffAgent):
        attempts = 0

        async def ainvoke(self, payload, config=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary model failure")
            return await super().ainvoke(payload)

    agent = FailOnceAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    async def enter():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            return await aenter_ticket(
                "external-1", message, account_id="acct", channel="chat",
                workflow=workflow, ticket_store=store, ticket_id="ticket-1",
            )

    with pytest.raises(RuntimeError, match="temporary model failure"):
        asyncio.run(enter())
    result = asyncio.run(enter())

    assert agent.attempts == 2
    assert result["ticket"]["status"] == "handed_off"
    assert store.status("ticket-1") == "handed_off"
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM tickets WHERE ticket_id='ticket-1'"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM ticket_messages WHERE ticket_id='ticket-1' AND message_id='customer-1'"
    ).fetchone()[0] == 1
    connection.close()


def test_initial_checkpoint_failure_reuses_ticket_and_repairs_checkpoint(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_ticket_db(database)
    store = TicketStore(database)
    agent = HandoffAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    async def fail_first_entry():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            class FailingCheckpointWrite:
                async def aget_state(self, config):
                    return await workflow.aget_state(config)

                async def aupdate_state(self, *args, **kwargs):
                    raise OSError("checkpoint write failed")

            with pytest.raises(OSError, match="checkpoint write failed"):
                await aenter_ticket(
                    "external-1", message, account_id="acct", channel="chat",
                    workflow=FailingCheckpointWrite(), ticket_store=store, ticket_id="ticket-1",
                )

    asyncio.run(fail_first_entry())
    assert store.status("ticket-1") == "processing"
    assert store.processing_status("ticket-1") == "failed"

    async def retry_entry():
        async with persistent_workflow(database, ticket_store=store, agent=agent) as workflow:
            return await aenter_ticket(
                "external-1", message, account_id="acct", channel="chat",
                workflow=workflow, ticket_store=store, ticket_id="ticket-1",
            )

    result = asyncio.run(retry_entry())
    assert result["ticket"]["status"] == "handed_off"
    assert store.status("ticket-1") == "handed_off"
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM tickets WHERE ticket_id='ticket-1'"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM ticket_messages WHERE ticket_id='ticket-1' AND message_id='customer-1'"
    ).fetchone()[0] == 1
    connection.close()
