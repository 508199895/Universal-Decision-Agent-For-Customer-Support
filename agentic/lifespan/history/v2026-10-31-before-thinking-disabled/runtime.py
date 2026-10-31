"""Runtime resource lifecycle for the compiled ticket workflow."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agentic.capabilities.middleware.token_control import DEFAULT_MAX_OUTPUT_TOKENS
from agentic.capabilities.tools.support_toolset import acollect_support_tools
from agentic.data.engine import close_engine, get_session_factory, init_engine
from agentic.data.tickets.TicketEventStore import TicketEventStore
from agentic.data.tickets.TicketStore import TicketStore
from agentic.orchestration.ticket.manager import TicketManager
from agentic.orchestration.workflow.factory import build_workflow
from agentic.orchestration.workflow.factory import WorkflowManager


def _default_agent_database() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "core" / "agent_tickets.db"


def _default_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is missing.")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
    )


@asynccontextmanager
async def persistent_workflow(
    database: str | os.PathLike,
    *,
    escalation_agent,
    supervisor_agent=None,
):
    """Open a workflow and keep its saver alive for the context lifetime."""
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        yield build_workflow(
            checkpointer=checkpointer,
            escalation_agent=escalation_agent,
            supervisor_agent=supervisor_agent,
        )


@asynccontextmanager
async def default_workflow_runtime():
    """Create runtime resources, compile the graph, and close the model on exit."""
    database = _default_agent_database()
    model = _default_model()
    try:
        await init_engine(database)
        session_factory = get_session_factory()
        ticket_store = TicketStore(session_factory)
        event_store = TicketEventStore(session_factory)
        ticket_manager = TicketManager(ticket_store)
        support_tools = await acollect_support_tools()
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            workflow = WorkflowManager().initialize(
                model=model,
                tools=list(support_tools),
                checkpointer=checkpointer,
            )
            yield workflow, ticket_manager, ticket_store, event_store
    finally:
        try:
            await _close_model_clients(model)
        finally:
            await close_engine()


async def _close_model_clients(model: BaseChatModel) -> None:
    for client, asynchronous in (
        (getattr(model, "root_async_client", None), True),
        (getattr(model, "root_client", None), False),
    ):
        if client is None:
            continue
        result = client.close()
        if asynchronous:
            await result
