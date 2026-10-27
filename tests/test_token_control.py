import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentic.capabilities.middleware.token_control import (
    CONTEXT_SUMMARIZATION_THRESHOLD,
    TicketRuntimeContext,
    TicketTokenBudgetMiddleware,
    TicketTokenLimitExceeded,
    build_ticket_middleware,
)
from agentic.data.tickets.message_journal import TicketJournal
from agentic.data.tickets.ticket_store import TicketEventStore
from agentic.orchestration.workflow import _normalize_checkpoint_state


class _MiddlewareModel:
    _llm_type = "fake-chat"

    def with_retry(self):
        return self


def test_ticket_token_middleware_blocks_before_call_at_the_limit(tmp_path):
    store = TicketEventStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1")
    journal = TicketJournal(
        store, user_id="user-1", ticket_id="ticket-1", token_limit=200_000,
    )
    journal.total_tokens = 200_000
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=TicketRuntimeContext(journal=journal)),
    )
    called = False

    def handler(_request):
        nonlocal called
        called = True
        return "model response"

    with pytest.raises(TicketTokenLimitExceeded, match="200000"):
        TicketTokenBudgetMiddleware().wrap_model_call(request, handler)
    assert called is False

    async def check_summarizer():
        runtime = SimpleNamespace(context=request.runtime.context)
        await build_ticket_middleware(_MiddlewareModel())[1].abefore_model({}, runtime)

    with pytest.raises(TicketTokenLimitExceeded, match="200000"):
        asyncio.run(check_summarizer())


def test_agent_middleware_blocks_real_agent_model_call_at_ticket_cap(tmp_path):
    store = TicketEventStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1")
    journal = TicketJournal(store, user_id="user-1", ticket_id="ticket-1")
    journal.total_tokens = journal.token_limit
    model = FakeListChatModel(responses=["should not be returned"])
    agent = create_agent(
        model=model,
        middleware=build_ticket_middleware(model),
        context_schema=TicketRuntimeContext,
    )

    async def invoke():
        with pytest.raises(TicketTokenLimitExceeded):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="question")]},
                context=TicketRuntimeContext(journal=journal),
            )

    asyncio.run(invoke())
    assert model.i == 0


def test_first_call_usage_reaches_limit_and_next_agent_call_is_blocked(tmp_path):
    class UsageModel(FakeListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.i += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="first response",
                usage_metadata={
                    "input_tokens": 150_000,
                    "output_tokens": 50_000,
                    "total_tokens": 200_000,
                },
            ))])

    store = TicketEventStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1")
    journal = TicketJournal(
        store, user_id="user-1", ticket_id="ticket-1", token_limit=200_000,
    )
    model = UsageModel(responses=[])
    agent = create_agent(
        model=model,
        middleware=build_ticket_middleware(model),
        context_schema=TicketRuntimeContext,
    )
    context = TicketRuntimeContext(journal=journal)

    async def invoke_twice():
        await agent.ainvoke(
            {"messages": [HumanMessage(content="first question")]},
            config={"callbacks": [journal]}, context=context,
        )
        assert journal.total_tokens == 200_000
        with pytest.raises(TicketTokenLimitExceeded):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="second question")]},
                config={"callbacks": [journal]}, context=context,
            )

    asyncio.run(invoke_twice())
    assert model.i == 1


def test_middleware_factory_uses_shared_context_threshold():
    middleware = build_ticket_middleware(model=_MiddlewareModel())

    assert len(middleware) == 2
    assert middleware[1].trigger == ("tokens", CONTEXT_SUMMARIZATION_THRESHOLD)
    counter = middleware[1].token_counter
    assert getattr(counter, "func", counter).__name__ == "count_tokens_approximately"


def test_existing_total_token_column_is_preserved_when_usage_columns_are_added(tmp_path):
    database = tmp_path / "legacy-agent.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, status TEXT NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO tickets VALUES ('ticket-1', 'user-1', 'running', 1234, 'created', 'updated');""",
    )
    connection.close()

    store = TicketEventStore(database)
    record = store.get_ticket("ticket-1")

    assert (record.tokens, record.input_tokens, record.output_tokens, record.llm_call_count) == (1234, 0, 0, 0)


def test_old_checkpoint_budget_group_is_removed_during_normalization():
    state = {
        "messages": [],
        "ticket": {"messages": []},
        "budget": {"outer_nodes_used": 4},
    }

    normalized, changed = _normalize_checkpoint_state(state)

    assert "budget" not in normalized
    assert "messages" not in normalized["ticket"]
    assert changed is True
