"""Load YAML role configurations and assemble application-lifetime agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agentic.agents.escalation.contract import EscalationMaterial
from agentic.agents.prompt import apply_prompt_template
from agentic.agents.supervisor.contract import SupervisorIntentResult
from agentic.agents.validate.contract import ValidateResult
from agentic.capabilities.middleware.token_control import (
    TicketRuntimeContext,
    build_ticket_middleware,
)
from agentic.capabilities.support_toolset import validate_support_tools


AGENT_NAMES = ("supervisor", "query", "validate", "escalation")


@lru_cache(maxsize=len(AGENT_NAMES))
def get_agent_soul(agent_name: str) -> dict[str, Any]:
    """Read and cache the complete YAML configuration for one agent role."""
    if (
        not isinstance(agent_name, str)
        or not agent_name
        or Path(agent_name).name != agent_name
        or any(separator in agent_name for separator in ("/", "\\"))
    ):
        raise ValueError("agent_name must be a simple role name")
    path = Path(__file__).parent / agent_name / "config.yaml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read agent YAML config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"agent YAML config {path} must contain a mapping")

    required_fields = {"name", "prompt"}
    optional_fields = {
        "model", "tools", "middleware", "context_schema", "response_format",
    }
    allowed_fields = required_fields | optional_fields
    missing_fields = sorted(required_fields - config.keys())
    unknown_fields = sorted(config.keys() - allowed_fields)
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {', '.join(missing_fields)}")
        if unknown_fields:
            details.append(f"unsupported fields: {', '.join(unknown_fields)}")
        raise ValueError(f"invalid agent YAML config {path}: {'; '.join(details)}")
    if config["name"] != agent_name:
        raise ValueError(f"agent YAML config {path} must have name: {agent_name}")

    prompt = config["prompt"]
    if not isinstance(prompt, dict):
        raise ValueError(f"agent YAML config {path} prompt must be a mapping")
    required_prompt_fields = {"role", "objective", "instructions", "boundaries"}
    missing_prompt_fields = sorted(required_prompt_fields - prompt.keys())
    unknown_prompt_fields = sorted(prompt.keys() - required_prompt_fields)
    if missing_prompt_fields or unknown_prompt_fields:
        details = []
        if missing_prompt_fields:
            details.append(f"missing fields: {', '.join(missing_prompt_fields)}")
        if unknown_prompt_fields:
            details.append(f"unsupported fields: {', '.join(unknown_prompt_fields)}")
        raise ValueError(f"invalid prompt config in {path}: {'; '.join(details)}")
    return config

RESPONSE_FORMATS: dict[str, type[Any]] = {
    "supervisor_intent": SupervisorIntentResult,
    "validate_result": ValidateResult,
    "escalation_material": EscalationMaterial,
}


@dataclass(frozen=True)
class AgentSet:
    """Prebuilt agent runnables shared by workflows in one runtime."""

    query: Any | None = None
    supervisor: Any | None = None
    validate: Any | None = None
    escalation: Any | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Resolved options passed to the shared LangChain agent factory."""

    model: BaseChatModel
    tools: Sequence[BaseTool] = ()
    middleware: Sequence[Any] = ()
    context_schema: type[Any] | None = None
    system_prompt: str | None = None
    response_format: type[Any] | None = None


def _resolve_agent_config(
    name: str,
    raw_config: Mapping[str, Any],
    *,
    model: BaseChatModel,
    available_tools: Sequence[BaseTool],
) -> AgentConfig:
    if raw_config.get("name") != name:
        raise ValueError(f"agent {name} YAML config must declare name: {name}")
    model_name = raw_config.get("model", "runtime")
    models = {"runtime": model}
    if not isinstance(model_name, str) or model_name not in models:
        raise ValueError(f"agent {name} has unsupported model identifier")
    resolved_model = models[model_name]

    tool_names = raw_config.get("tools", [])
    if not isinstance(tool_names, list) or any(not isinstance(item, str) for item in tool_names):
        raise ValueError(f"agent {name} tools must be a list of tool-name identifiers")
    tools_by_name = {tool.name: tool for tool in available_tools}
    if len(tool_names) != len(set(tool_names)):
        raise ValueError(f"agent {name} tools must not contain duplicate names")
    missing_tools = sorted(set(tool_names) - tools_by_name.keys())
    if missing_tools:
        raise ValueError(
            f"agent {name} references unavailable tools: {', '.join(missing_tools)}"
        )

    middleware_names = raw_config.get("middleware", ["ticket_runtime"])
    if not isinstance(middleware_names, list) or any(
        not isinstance(item, str) for item in middleware_names
    ):
        raise ValueError(f"agent {name} middleware must be a list of identifiers")
    middleware: list[Any] = []
    for middleware_name in middleware_names:
        if middleware_name != "ticket_runtime":
            raise ValueError(
                f"agent {name} references unknown middleware: {middleware_name}"
            )
        middleware.extend(build_ticket_middleware(resolved_model))

    context_name = raw_config.get("context_schema", "ticket_runtime")
    if context_name is not None and not isinstance(context_name, str):
        raise ValueError(f"agent {name} context_schema must be a string identifier or null")
    context_schemas = {"ticket_runtime": TicketRuntimeContext, None: None}
    if context_name not in context_schemas:
        raise ValueError(f"agent {name} references unknown context schema: {context_name}")

    response_name = raw_config.get("response_format")
    if response_name is not None and not isinstance(response_name, str):
        raise ValueError(f"agent {name} response_format must be a string identifier or null")
    response_formats = {**RESPONSE_FORMATS, None: None}
    if response_name not in response_formats:
        raise ValueError(f"agent {name} references unknown response format: {response_name}")

    return AgentConfig(
        model=resolved_model,
        tools=tuple(tools_by_name[tool_name] for tool_name in tool_names),
        middleware=tuple(middleware),
        context_schema=context_schemas[context_name],
        system_prompt=apply_prompt_template(raw_config["prompt"]),
        response_format=response_formats[response_name],
    )


def build_agent(config: AgentConfig):
    """Construct an agent from resolved YAML configuration."""
    if config.model is None:
        raise ValueError("agent config requires a model")
    kwargs = {
        "model": config.model,
        "tools": list(config.tools),
        "middleware": list(config.middleware),
    }
    if config.context_schema is not None:
        kwargs["context_schema"] = config.context_schema
    if config.system_prompt is not None:
        kwargs["system_prompt"] = config.system_prompt
    if config.response_format is not None:
        kwargs["response_format"] = config.response_format
    return create_agent(**kwargs)


def build_agent_set(*, model: BaseChatModel, tools: Sequence[BaseTool]) -> AgentSet:
    """Load each role's YAML and build all current workflow agents uniformly."""
    if model is None:
        raise ValueError("agent set requires a model")
    available_tools = validate_support_tools(tools)
    agents = {
        name: build_agent(config=_resolve_agent_config(
            name,
            get_agent_soul(name),
            model=model,
            available_tools=available_tools,
        ))
        for name in AGENT_NAMES
    }
    return AgentSet(**agents)


__all__ = [
    "AgentConfig",
    "AgentSet",
    "build_agent",
    "build_agent_set",
    "get_agent_soul",
]
