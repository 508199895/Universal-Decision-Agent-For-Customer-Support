"""Build the account lookup expert graph."""

from __future__ import annotations

from typing import Any

from langchain.agents.structured_output import ProviderStrategy
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from agentic.agents.agent import build_agent_from_config
from agentic.agents.account.state import AccountResult, AccountState
from agentic.capabilities.middleware.token_control import TicketRuntimeContext


TOOL_NAME = "lookup_current_user_account"
TOOL_DESCRIPTION = "Read the current ticket user's account and current status for the given task."


def build_graph(*, app_config, models, available_tools):
    account_agent = build_agent_from_config(
        agent_name="account",
        app_config=app_config,
        models=models,
        available_tools=available_tools,
        response_format=ProviderStrategy(AccountResult),
    )

    async def run_account(
        state: AccountState,
        config: RunnableConfig,
        runtime: Runtime[TicketRuntimeContext],
    ) -> dict[str, Any]:
        result = await account_agent.ainvoke(
            {"messages": list(state["messages"])},
            config=config,
            context=runtime.context,
        )
        structured = result.get("structured_response") if isinstance(result, dict) else result
        account_result = (
            structured if isinstance(structured, AccountResult)
            else AccountResult.model_validate(structured)
        )
        return {
            "messages": result.get("messages", []) if isinstance(result, dict) else [],
            "result": account_result,
        }

    builder = StateGraph(AccountState, context_schema=TicketRuntimeContext)
    builder.add_node("account", run_account)
    builder.add_edge(START, "account")
    builder.add_edge("account", END)
    return builder.compile()
