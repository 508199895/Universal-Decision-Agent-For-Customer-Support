"""Single customer-support agent assembled with its fixed startup toolset."""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agentic.capabilities.support_toolset import validate_support_tools


def build_customer_support_agent(
    *, model: BaseChatModel, tools: Sequence[BaseTool], middleware=(), context_schema=None,
):
    """Create one tool-calling agent with a fixed set of customer-support tools."""
    if model is None:
        raise ValueError("customer-support agent requires a model")
    selected_tools = validate_support_tools(tools)
    return create_agent(
        model=model, tools=list(selected_tools), middleware=list(middleware),
        context_schema=context_schema,
    )
