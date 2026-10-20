import asyncio
import sqlite3
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agentic.message_journal import TicketMessageJournal
from agentic.ticket_store import TicketStore
from agentic.workflow import _invoke_with_journal, aresume_ticket


def create_core_db(path):
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
        INSERT INTO tickets VALUES ('ticket-1', 'acct', 'user-1', 'chat', CURRENT_TIMESTAMP);
        INSERT INTO ticket_metadata (ticket_id, status) VALUES ('ticket-1', 'processing');
        """
    )
    connection.commit()
    connection.close()


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
def test_journal_flushes_remaining_messages_when_graph_exits(tmp_path, exit_kind):
    database = tmp_path / "uda.sqlite"
    create_core_db(database)
    store = TicketStore(database)
    store.ensure_constraints()
    journal = TicketMessageJournal(store, ticket_id="ticket-1")

    class FakeGraph:
        async def ainvoke(self, payload, *, config, durability):
            assert durability == "sync"
            handler = config["callbacks"][0]
            await handler.on_chat_model_start(
                {}, [[HumanMessage(content="客户输入", id="human-1")]], run_id=uuid4(),
            )
            await handler.on_llm_end(
                LLMResult(generations=[[
                    ChatGeneration(message=AIMessage(content="模型输出", id="ai-1")),
                ]]),
                run_id=uuid4(),
            )
            await handler.on_tool_end(
                ToolMessage(content="工具结果", tool_call_id="tool-call-1", id="tool-1"),
                run_id=uuid4(),
            )
            if exit_kind == "error":
                raise RuntimeError("graph failed")
            if exit_kind == "cancel":
                raise asyncio.CancelledError()
            return {"ok": True}

    async def exercise():
        if exit_kind == "normal":
            assert await _invoke_with_journal(FakeGraph(), "ticket-1", journal) == {"ok": True}
        else:
            error = RuntimeError if exit_kind == "error" else asyncio.CancelledError
            with pytest.raises(error):
                await _invoke_with_journal(FakeGraph(), "ticket-1", journal)

    asyncio.run(exercise())
    events = store.message_events("ticket-1")
    assert [event["event_id"] for event in events] == [1, 2, 3]
    assert [event["event_type"] for event in events] == [
        "llm.human.input", "llm.ai.response", "llm.tool.result",
    ]
    assert [event["content"]["content"] for event in events] == [
        "客户输入", "模型输出", "工具结果",
    ]
    assert all(event["ticket_id"] == "ticket-1" for event in events)


def test_journal_flushes_when_buffer_reaches_batch_size(tmp_path):
    database = tmp_path / "uda.sqlite"
    create_core_db(database)
    store = TicketStore(database)
    store.ensure_constraints()
    journal = TicketMessageJournal(store, ticket_id="ticket-1", flush_size=1)

    asyncio.run(journal.on_chat_model_start(
        {}, [[HumanMessage(content="客户输入", id="human-1")]], run_id=uuid4(),
    ))

    events = store.message_events("ticket-1")
    assert len(events) == 1
    assert events[0]["content"]["type"] == "human"


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
def test_resume_ticket_journal_flushes_on_every_graph_exit(tmp_path, exit_kind):
    database = tmp_path / "uda.sqlite"
    create_core_db(database)
    store = TicketStore(database)
    store.ensure_constraints()

    class FakeGraph:
        async def aget_state(self, config):
            assert config["configurable"]["thread_id"] == "ticket-1"
            return type("Snapshot", (), {
                "values": {"ticket": {"ticket_id": "ticket-1", "status": "waiting_reply"}},
                "next": ("wait_for_reply",),
            })()

        async def ainvoke(self, payload, *, config, durability):
            assert durability == "sync"
            assert payload.resume == "客户补充" if payload is not None else False
            handler = config["callbacks"][0]
            await handler.on_chat_model_start(
                {}, [[HumanMessage(content="客户补充", id="human-resume-1")]], run_id=uuid4(),
            )
            await handler.on_llm_end(
                LLMResult(generations=[[
                    ChatGeneration(message=AIMessage(content="恢复后的回复", id="ai-resume-1")),
                ]]),
                run_id=uuid4(),
            )
            if exit_kind == "error":
                raise RuntimeError("resumed graph failed")
            if exit_kind == "cancel":
                raise asyncio.CancelledError()
            return {"ok": True}

    async def exercise():
        if exit_kind == "normal":
            assert await aresume_ticket(
                "ticket-1", workflow=FakeGraph(), reply="客户补充", ticket_store=store,
            ) == {"ok": True}
        else:
            error = RuntimeError if exit_kind == "error" else asyncio.CancelledError
            with pytest.raises(error):
                await aresume_ticket(
                    "ticket-1", workflow=FakeGraph(), reply="客户补充", ticket_store=store,
                )

    asyncio.run(exercise())
    events = store.message_events("ticket-1")
    assert [event["event_type"] for event in events] == [
        "llm.human.input", "llm.ai.response",
    ]
    assert [event["content"]["content"] for event in events] == [
        "客户补充", "恢复后的回复",
    ]
