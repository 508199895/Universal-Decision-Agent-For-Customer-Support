"""Assemble the configured main Agent and specialist Agent tools."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool
import yaml

from agentic.agents.agent import wrap_agent_graph
from agentic.config import AGENTIC_ROOT
from agentic.capabilities.tools.account_tools import bind_current_user_account_tools
from agentic.capabilities.tools.clarification import ask_clarification
from agentic.capabilities.tools.feishu_handoff import (
    FeishuTextSender,
    build_feishu_handoff_tool,
)

class WorkflowManager:
    """Load the selected workflow, bind configured Agents, and compile the graph."""

    def __init__(self) -> None:
        self.graph = None

    def initialize(
        self,
        *,
        app_config: Mapping[str, Any],
        models: Mapping[str, Any],
        tools: Sequence[BaseTool],
        checkpointer: BaseCheckpointSaver,
        handoff_sender: FeishuTextSender | None = None,
    ):
        """Build and cache the configured direct Agent graph."""
        if self.graph is not None:
            return self.graph
        if checkpointer is None or not models:
            raise ValueError("workflow initialization requires models and a durable checkpointer")

        workflow_selection = app_config.get("workflow")
        if not isinstance(workflow_selection, Mapping):
            raise ValueError("app config must select a workflow")
        workflow_dir = (AGENTIC_ROOT / workflow_selection["directory"]).resolve()
        workflow_path = workflow_dir / f"{workflow_selection['name']}.yaml"
        try:
            workflow_dir.relative_to(AGENTIC_ROOT.resolve())
        except ValueError as error:
            raise ValueError("workflow config directory must stay within agentic") from error
        try:
            workflow_config = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"could not read workflow config {workflow_path}: {error}") from error
        if not isinstance(workflow_config, dict) or set(workflow_config) != {
            "main_agent", "expert_agents",
        }:
            raise ValueError(
                f"workflow config {workflow_path} must contain main_agent and expert_agents"
            )
        main_agent_name = workflow_config["main_agent"]
        expert_names = workflow_config["expert_agents"]
        if not isinstance(main_agent_name, str) or not isinstance(expert_names, list) or any(
            not isinstance(name, str) for name in expert_names
        ):
            raise ValueError("workflow Agent names must be strings")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", main_agent_name) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in expert_names
        ):
            raise ValueError("workflow Agent names must be simple identifiers")
        if len(expert_names) != len(set(expert_names)) or main_agent_name in expert_names:
            raise ValueError("workflow Agent list contains duplicates")

        globally_enabled = set(app_config["tools"]["enabled"])
        tool_catalog: dict[str, BaseTool] = {}

        def register(candidate: BaseTool) -> None:
            if candidate.name in tool_catalog:
                raise ValueError(f"duplicate tool name: {candidate.name}")
            tool_catalog[candidate.name] = candidate

        for candidate in bind_current_user_account_tools(tools):
            register(candidate)
        register(ask_clarification)
        register(build_feishu_handoff_tool(handoff_sender))
        missing_enabled = sorted(globally_enabled - tool_catalog.keys())
        if missing_enabled:
            raise ValueError(
                "globally enabled tools are unavailable: " + ", ".join(missing_enabled)
            )
        available_tools = {
            name: tool_catalog[name] for name in sorted(globally_enabled)
        }

        expert_tools: list[BaseTool] = []
        for expert_name in expert_names:
            if not expert_name.replace("_", "").isalnum():
                raise ValueError(f"invalid expert Agent name: {expert_name!r}")
            module = importlib.import_module(f"agentic.agents.{expert_name}.graph")
            expert_graph = module.build_graph(
                app_config=app_config,
                models=models,
                available_tools=available_tools,
            )
            expert_tools.append(wrap_agent_graph(
                graph=expert_graph,
                name=module.TOOL_NAME,
                description=module.TOOL_DESCRIPTION,
            ))
        expert_tool_names = [tool.name for tool in expert_tools]
        if len(expert_tool_names) != len(set(expert_tool_names)) or any(
            name in available_tools for name in expert_tool_names
        ):
            raise ValueError("workflow expert tool names must be unique")
        main_module = importlib.import_module(f"agentic.agents.{main_agent_name}.graph")
        self.graph = main_module.build_graph(
            app_config=app_config,
            models=models,
            available_tools=available_tools,
            extra_tools=expert_tools,
            checkpointer=checkpointer,
        )
        return self.graph
