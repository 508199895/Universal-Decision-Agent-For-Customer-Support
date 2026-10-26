import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agentic.data.tickets.message_journal import TicketJournal
from agentic.data.tickets.ticket_store import TicketEventStore
from agentic.orchestration.workflow import _invoke_with_journal


def _store_with_ticket(path):
    store = TicketEventStore(path)
    store.create_ticket(ticket_id="ticket-1", user_id="user-1")
    return store


def test_journal_records_messages_usage_and_tool_metadata(tmp_path):
    store = _store_with_ticket(tmp_path / "agent.sqlite")
    journal = TicketJournal(store, user_id="user-1", ticket_id="ticket-1")
    human = HumanMessage(content="question", id="message-1")
    assistant = AIMessage(
        content="answer", id="message-2",
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )
    tool = ToolMessage(content="lookup result", tool_call_id="call-1", id="message-3")

    async def record():
        await journal.on_chat_model_start({}, [[human]], name="query")
        await journal.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=assistant)]]),
            name="query", run_id="run-1",
        )
        await journal.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=assistant)]]),
            name="query", run_id="run-1",
        )
        await journal.on_tool_end({"messages": [tool]}, name="lookup")
        await journal.persist_usage_snapshot()
        await journal.flush()

    asyncio.run(record())
    events = store.events(user_id="user-1", ticket_id="ticket-1")
    assert [event["event_type"] for event in events] == [
        "ticket.created", "llm.human.input", "llm.ai.response", "llm.ai.response",
        "llm.tool.result",
    ]
    assert [event["seq"] for event in events] == [1, 2, 3, 4, 5]
    assert events[1]["content"]["id"] == "message-1"
    assert events[2]["metadata"] == {
        "caller": "query", "usage": assistant.usage_metadata,
    }
    assert events[4]["metadata"] == {"tool_name": "lookup"}
    assert store.get_ticket("ticket-1").tokens == 5
    record = store.get_ticket("ticket-1")
    assert (record.input_tokens, record.output_tokens, record.llm_call_count) == (3, 2, 1)


def test_journal_excludes_agent_summary_as_customer_input(tmp_path):
    store = _store_with_ticket(tmp_path / "agent.sqlite")
    journal = TicketJournal(store, user_id="user-1", ticket_id="ticket-1")

    async def record():
        await journal.on_chat_model_start(
            {}, [[HumanMessage(content="internal", id="summary-1", name="summary")]],
        )
        await journal.on_chat_model_start(
            {}, [[HumanMessage(content="customer", id="message-1")]],
        )
        await journal.flush()

    asyncio.run(record())
    events = store.events(user_id="user-1", ticket_id="ticket-1")
    assert [event["event_type"] for event in events] == ["ticket.created", "llm.human.input"]
    assert events[-1]["content"]["id"] == "message-1"


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
def test_invoke_journal_flushes_tail_on_every_exit(tmp_path, exit_kind):
    store = _store_with_ticket(tmp_path / "agent.sqlite")
    journal = TicketJournal(store, user_id="user-1", ticket_id="ticket-1")

    class FakeWorkflow:
        async def ainvoke(self, _payload, config, durability):
            assert durability == "async"
            active_journal = config["callbacks"][0]
            await active_journal.record_lifecycle(
                "test.event", "trace", {"value": "buffered"},
            )
            await active_journal.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=AIMessage(
                    content="answer", id="model-answer",
                    usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                ))]]),
                run_id="run-finally",
            )
            if exit_kind == "error":
                raise RuntimeError("graph failed")
            if exit_kind == "cancel":
                raise asyncio.CancelledError()
            return {"ok": True}

    async def invoke():
        if exit_kind == "error":
            with pytest.raises(RuntimeError, match="graph failed"):
                await _invoke_with_journal(FakeWorkflow(), "user-1", journal)
        elif exit_kind == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await _invoke_with_journal(FakeWorkflow(), "user-1", journal)
        else:
            assert await _invoke_with_journal(FakeWorkflow(), "user-1", journal) == {"ok": True}

    asyncio.run(invoke())
    events = store.events(user_id="user-1", ticket_id="ticket-1")
    record = store.get_ticket("ticket-1")
    assert (record.tokens, record.input_tokens, record.output_tokens, record.llm_call_count) == (3, 1, 2, 1)
    event_types = [event["event_type"] for event in events]
    assert "test.event" in event_types
    if exit_kind == "normal":
        assert event_types[-1] == "run.end"
    elif exit_kind == "error":
        assert event_types[-1] == "run.error"
    else:
        assert event_types[-1] == "run.cancel"


def test_flush_threshold_writes_events_immediately(tmp_path):
    store = _store_with_ticket(tmp_path / "agent.sqlite")
    journal = TicketJournal(store, user_id="user-1", ticket_id="ticket-1", flush_size=1)

    asyncio.run(journal.record_lifecycle("run.start", "trace", {"ticket_id": "ticket-1"}))

    events = store.events(user_id="user-1", ticket_id="ticket-1")
    assert [event["event_type"] for event in events] == ["ticket.created", "run.start"]
