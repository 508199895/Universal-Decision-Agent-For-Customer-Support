"""Shared configuration loading, Agent construction, and expert graph tools."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import RunnableConfig

from agentic.agents.prompt import apply_prompt_template
from agentic.config import AGENTIC_ROOT
from agentic.capabilities.middleware.token_control import (
    TicketRuntimeContext,
    build_ticket_middleware,
)
from agentic.capabilities.middleware.tool_restriction import (
    ToolRestrictionMiddleware,
)


AGENT_NAMES = ("supervisor", "query", "validate", "account")
_INTERNAL_AGENT_CONFIGS = {
    "query": ("public_knowledge", "query.yaml"),
    "validate": ("public_knowledge", "validate.yaml"),
}
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} {path} must contain a mapping")
    return loaded


def load_agent_config(agent_name: str, filename: str | None = None) -> dict[str, Any]:
    """Read one Agent YAML, including internal query/validate configs."""
    if not isinstance(agent_name, str) or not _NAME_PATTERN.fullmatch(agent_name):
        raise ValueError("agent_name must be a simple role name")
    if agent_name in _INTERNAL_AGENT_CONFIGS:
        directory, default_filename = _INTERNAL_AGENT_CONFIGS[agent_name]
        filename = filename or default_filename
    else:
        directory, filename = agent_name, filename or "config.yaml"
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Agent config filename must be a simple filename")
    path = AGENTIC_ROOT / "agents" / directory / filename
    config = _read_yaml(path, label="Agent YAML config")

    required_fields = {"name", "prompt"}
    optional_fields = {"model", "tools", "middleware", "context_schema"}
    missing = sorted(required_fields - config.keys())
    unknown = sorted(config.keys() - required_fields - optional_fields)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unsupported fields: {', '.join(unknown)}")
        raise ValueError(f"invalid Agent config {path}: {'; '.join(details)}")

    expected_name = agent_name
    if config["name"] != expected_name:
        raise ValueError(f"Agent YAML config {path} must have name: {expected_name}")
    if "model" in config and (
        not isinstance(config["model"], str) or not config["model"].strip()
    ):
        raise ValueError(f"Agent {agent_name} model must be a model name")
    tool_names = config.get("tools", [])
    if not isinstance(tool_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in tool_names
    ):
        raise ValueError(f"Agent {agent_name} tools must be a list of tool names")
    if len(tool_names) != len(set(tool_names)):
        raise ValueError(f"Agent {agent_name} tools must not contain duplicate names")

    prompt = config["prompt"]
    if not isinstance(prompt, dict):
        raise ValueError(f"Agent config {path} prompt must be a mapping")
    required_prompt_fields = {"role", "objective", "instructions", "boundaries"}
    if set(prompt) != required_prompt_fields:
        raise ValueError(f"Agent config {path} prompt fields must be {sorted(required_prompt_fields)}")
    return config


def get_agent_soul(agent_name: str) -> dict[str, Any]:
    """Compatibility name for the shared Agent YAML loader."""
    return load_agent_config(agent_name)


def build_agent_from_config(
    *,
    agent_name: str,
    app_config: Mapping[str, Any],
    models: Mapping[str, BaseChatModel],
    available_tools: Mapping[str, BaseTool],
    response_format: Any | None = None,
    state_schema: Any | None = None,
    checkpointer: Any | None = None,
    config_filename: str | None = None,
    extra_tools: Sequence[BaseTool] = (),
):
    """Construct one Agent using its model override and effective tool intersection."""
    config = load_agent_config(agent_name, config_filename)
    model_name = config.get("model", app_config["models"]["default"])
    try:
        model = models[model_name]
    except KeyError as error:
        raise ValueError(f"Agent {agent_name} references a model that is not enabled: {model_name}") from error

    enabled_names = set(app_config["tools"]["enabled"])
    agent_names = set(config.get("tools", []))
    selected_names = sorted(enabled_names & agent_names & available_tools.keys())
    selected_tools = [available_tools[name] for name in selected_names]
    selected_tools.extend(extra_tools)
    middleware_names = config.get("middleware", ["ticket_runtime"])
    if not isinstance(middleware_names, list) or any(
        not isinstance(name, str) for name in middleware_names
    ):
        raise ValueError(f"Agent {agent_name} middleware must be a list of identifiers")
    middleware = []
    for name in middleware_names:
        if name == "ticket_runtime":
            middleware.extend(build_ticket_middleware(model))
        elif name == "tool_restriction":
            middleware.append(ToolRestrictionMiddleware())
        else:
            raise ValueError(f"Agent {agent_name} references unknown middleware: {name}")

    context_schema = config.get("context_schema", "ticket_runtime")
    if context_schema is not None and context_schema != "ticket_runtime":
        raise ValueError(f"Agent {agent_name} references unknown context schema: {context_schema}")
    kwargs = {
        "model": model,
        "tools": selected_tools,
        "middleware": middleware,
        "system_prompt": apply_prompt_template(config["prompt"]),
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if state_schema is not None:
        kwargs["state_schema"] = state_schema
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if context_schema == "ticket_runtime":
        kwargs["context_schema"] = TicketRuntimeContext
    return create_agent(**kwargs)


def wrap_agent_graph(*, graph: Any, name: str, description: str) -> BaseTool:
    """Expose an isolated Agent graph as a task-in/result-out supervisor tool."""
    if graph is None:
        raise ValueError("expert graph is required")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ValueError("tool name must be a simple identifier")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("tool description must be non-empty")

    async def invoke_expert(
        task: str,
        runtime: ToolRuntime[TicketRuntimeContext],
    ) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("expert task must be a non-empty string")
        result_state = await graph.ainvoke(
            {"task": task, "messages": [HumanMessage(content=task)]},
            config=runtime.config,
            context=runtime.context,
        )
        if not isinstance(result_state, Mapping) or "result" not in result_state:
            raise ValueError(f"expert graph {name} did not return a result field")
        result = result_state["result"]
        if result is None:
            raise ValueError(f"expert graph {name} returned an empty result")
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return json.dumps(result, ensure_ascii=False, default=str)

    return tool(name, description=description)(invoke_expert)


__all__ = [
    "AGENT_NAMES",
    "AGENTIC_ROOT",
    "build_agent_from_config",
    "get_agent_soul",
    "load_agent_config",
    "wrap_agent_graph",
]
