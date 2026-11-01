"""Build the main support Agent selected by a workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agentic.agents.agent import build_agent_from_config
from agentic.agents.supervisor.contract import SupervisorResponse


def build_graph(
    *,
    app_config: Mapping,
    models: Mapping[str, BaseChatModel],
    available_tools: Mapping[str, BaseTool],
    extra_tools: Sequence[BaseTool] = (),
):
    """Construct the supervisor Agent with workflow-provided expert tools."""
    return build_agent_from_config(
        agent_name="supervisor",
        app_config=app_config,
        models=models,
        available_tools=available_tools,
        response_format=ProviderStrategy(SupervisorResponse),
        extra_tools=extra_tools,
    )
