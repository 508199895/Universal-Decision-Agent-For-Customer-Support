"""Runtime resource lifecycle for the compiled ticket workflow."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agentic.capabilities.middleware.token_control import DEFAULT_MAX_OUTPUT_TOKENS
from agentic.capabilities.mcp.client import mcp_client_lifespan
from agentic.capabilities.tools.support_toolset import acollect_support_tools
from agentic.capabilities.tools.feishu_handoff import FeishuTextSender
from agentic.data.engine import close_engine, get_session_factory, init_engine
from agentic.data.tickets.TicketEventStore import TicketEventStore
from agentic.data.tickets.TicketStore import TicketStore
from agentic.orchestration.ticket.manager import TicketManager
from agentic.config import load_app_config
from agentic.orchestration.workflow.graph import WorkflowManager


def _default_agent_database() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "core" / "agent_tickets.db"


def _default_model(model_config: Mapping[str, Any] | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    load_dotenv()
    if model_config is None:
        app_config = load_app_config()
        model_config = app_config["models"]["enabled"][app_config["models"]["default"]]
    provider = model_config.get("provider")
    if provider not in {"deepseek", "openai-compatible", "openai"}:
        raise ValueError(f"Unsupported chat model provider: {model_config.get('provider')}")
    api_mode = model_config.get("api_mode", "chat_completions")
    if api_mode not in {"chat_completions", "responses"}:
        raise ValueError(f"Unsupported model API mode: {api_mode}")
    default_api_key_env = (
        "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    )
    api_key_name = model_config.get("api_key_env", default_api_key_env)
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise ValueError(f"{api_key_name} is missing.")
    model_env = model_config.get("model_env")
    base_url_env = model_config.get("base_url_env")
    model_name = os.getenv(model_env, model_config["model"]) if model_env else model_config["model"]
    base_url = (
        os.getenv(base_url_env, model_config.get("base_url", ""))
        if base_url_env else model_config.get("base_url")
    )
    max_tokens_env = model_config.get("max_tokens_env", "")
    max_tokens = int(os.getenv(max_tokens_env, model_config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)))
    temperature_env = model_config.get("temperature_env")
    temperature = float(os.getenv(temperature_env, model_config.get("temperature", 0))) if temperature_env else float(model_config.get("temperature", 0))
    model_kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": model_config.get("extra_body", {}),
        "use_responses_api": api_mode == "responses",
    }
    if model_config.get("reasoning_effort") is not None:
        model_kwargs["reasoning_effort"] = model_config["reasoning_effort"]
    if base_url:
        model_kwargs["base_url"] = base_url
    return ChatOpenAI(
        **model_kwargs,
    )


@asynccontextmanager
async def default_workflow_runtime(
    *,
    handoff_sender: FeishuTextSender | None = None,
    app_config: Mapping[str, Any] | None = None,
):
    """Create runtime resources from application config and compile the graph."""
    load_dotenv()
    app_config = dict(app_config or load_app_config())
    database = _default_agent_database()
    models: dict[str, BaseChatModel] = {}
    try:
        for name, model_config in app_config["models"]["enabled"].items():
            models[name] = _default_model(model_config)
        await init_engine(database)
        session_factory = get_session_factory()
        ticket_store = TicketStore(session_factory)
        event_store = TicketEventStore(session_factory)
        ticket_manager = TicketManager(ticket_store)
        enabled_mcp_tools = [
            name for name in app_config["tools"]["enabled"]
            if name.startswith(("account_", "qdrant_"))
        ]
        enabled_mcp_servers = sorted({
            "account" if name.startswith("account_") else "qdrant"
            for name in enabled_mcp_tools
        })
        async with AsyncExitStack() as mcp_stack:
            if enabled_mcp_servers:
                await mcp_stack.enter_async_context(mcp_client_lifespan(*enabled_mcp_servers))
            support_tools = await acollect_support_tools(enabled_mcp_tools)
            async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
                workflow = WorkflowManager().initialize(
                    app_config=app_config,
                    models=models,
                    tools=list(support_tools),
                    checkpointer=checkpointer,
                    handoff_sender=handoff_sender,
                )
                yield workflow, ticket_manager, ticket_store, event_store
    finally:
        try:
            for model in models.values():
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
