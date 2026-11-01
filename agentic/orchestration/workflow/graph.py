"""Application-time assembly of specialist agents and the outer workflow graph."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig
import yaml

from agentic.agents.agent import (
    wrap_agent_graph,
)
from agentic.config import AGENTIC_ROOT
from agentic.agents.public_knowledge.state import ValidateResult
from agentic.agents.supervisor.contract import SupervisorResponse
from agentic.capabilities.tools.account_tools import bind_current_user_account_tools
from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.capabilities.tools.clarification import ask_clarification
from agentic.capabilities.tools.feishu_handoff import (
    FeishuTextSender,
    build_feishu_handoff_tool,
)
from agentic.orchestration.workflow.handoff import (
    HANDOFF_NOTICE,
    build_handoff_messages,
)
from agentic.orchestration.workflow.state import (
    WorkflowState, merge_patch, validate_state,
)


def _message_id(message):
    return message.get("id") if isinstance(message, dict) else getattr(message, "id", None)


def _context_ticket_id(context: TicketRuntimeContext) -> str:
    ticket_id = context.ticket_id
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        raise ValueError("workflow context must provide ticket_id")
    return ticket_id


def _tool_message_payload(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _successful_handoff(messages: Any) -> dict[str, str] | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool" or getattr(
            message, "name", None
        ) != "request_human_handoff":
            continue
        result = _tool_message_payload(message)
        if isinstance(result, dict) and result.get("status") == "sent":
            return result
    return None


def _attempted_handoff(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "request_human_handoff"
        and _tool_message_payload(message).get("status") not in {"blocked", "deferred"}
        for message in messages
    )


def _successful_clarification(messages: Any) -> dict[str, str] | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool" or getattr(
            message, "name", None
        ) != "ask_clarification":
            continue
        result = _tool_message_payload(message)
        if not isinstance(result, dict) or result.get("status") != "asked":
            continue
        question = result.get("question")
        if isinstance(question, str) and question.strip():
            return {"question": question.strip()}
    return None


def _attempted_clarification(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "ask_clarification"
        and _tool_message_payload(message).get("status") not in {
            "asked", "deferred", "blocked", "error",
        }
        for message in messages
    )


async def _supervisor_node(
    state: WorkflowState,
    config: RunnableConfig,
    *,
    supervisor,
    context: TicketRuntimeContext,
):
    """Let supervisor answer, clarify, or hand off in this single graph node."""
    validate_state(state)
    result = await asyncio.wait_for(supervisor.ainvoke(
        {"messages": list(state["messages"])},
        config=config,
        context=context,
    ), timeout=60)
    supervisor_messages = result.get("messages", []) if isinstance(result, dict) else []
    handoff_result = _successful_handoff(supervisor_messages)
    clarification_result = _successful_clarification(supervisor_messages)
    if handoff_result is None and _attempted_handoff(supervisor_messages):
        raise RuntimeError("Feishu handoff tool did not confirm successful delivery")
    if clarification_result is None and _attempted_clarification(supervisor_messages):
        raise RuntimeError("Clarification tool did not confirm a customer-facing question")
    if handoff_result is not None and clarification_result is not None:
        raise RuntimeError("Supervisor cannot clarify and hand off in the same ticket run")

    response = None
    if clarification_result is None:
        structured = result.get("structured_response") if isinstance(result, dict) else result
        response = (structured if isinstance(structured, SupervisorResponse)
                    else SupervisorResponse.model_validate(structured))

    if response is not None and response.requires_handoff and handoff_result is None:
        raise RuntimeError("Supervisor requested handoff without calling the Feishu tool")

    ticket_id = _context_ticket_id(context)
    updated = merge_patch(state, {"clarify": {"prompt": None}}, writer="boundary")
    if handoff_result is not None:
        customer_message = HANDOFF_NOTICE
        updated = merge_patch(updated, {"escalation": {
            "summary": handoff_result["summary"],
            "description": handoff_result["description"],
        }}, writer="escalation")
        updated = merge_patch(updated, {"supervisor": {
                "intent": "escalation", "next_node": "finish",
                "message": customer_message,
            }}, writer="boundary")
        messages = list(updated["messages"])
        answer = AIMessage(content=customer_message,
                           id=f"supervisor:{ticket_id}:answer")
        for message in (
            AIMessage(content=json.dumps({
                "summary": handoff_result["summary"],
                "description": handoff_result["description"],
            }, ensure_ascii=False), id=f"escalation:{ticket_id}:result"),
            *build_handoff_messages(updated, ticket_id=ticket_id),
            answer,
        ):
            existing = next((i for i, item in enumerate(messages)
                             if _message_id(item) == message.id), None)
            if existing is None:
                messages.append(message)
            else:
                messages[existing] = message
        updated["messages"] = messages
    else:
        customer_message = (
            clarification_result["question"]
            if clarification_result is not None else response.answer
        )
        updated = merge_patch(updated, {"supervisor": {
                "next_node": "finish", "message": customer_message,
            }}, writer="boundary")
        messages = list(updated["messages"])
        answer = AIMessage(content=customer_message, id=f"supervisor:{ticket_id}:answer")
        existing = next((i for i, item in enumerate(messages)
                         if _message_id(item) == answer.id), None)
        if existing is None:
            messages.append(answer)
        else:
            messages[existing] = answer
        updated["messages"] = messages
    validate_state(updated)
    return updated


def build_workflow(*, checkpointer: BaseCheckpointSaver, supervisor_agent):
    """Compile the workflow with one Supervisor node and tool-based handoff."""
    if checkpointer is None:
        raise ValueError("workflow requires a durable checkpointer")
    if supervisor_agent is None:
        raise ValueError("workflow requires a prebuilt supervisor agent")
    graph = StateGraph(WorkflowState, context_schema=TicketRuntimeContext)
    async def run_supervisor(state, config, runtime: Runtime[TicketRuntimeContext]):
        return await _supervisor_node(
            state,
            config,
            supervisor=supervisor_agent,
            context=runtime.context,
        )
    graph.add_node("supervisor", run_supervisor)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile(checkpointer=checkpointer)


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
        """Build and cache the configured graph using application-scoped resources."""
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
        main_agent = main_module.build_graph(
            app_config=app_config,
            models=models,
            available_tools=available_tools,
            extra_tools=expert_tools,
        )
        self.graph = build_workflow(
            checkpointer=checkpointer,
            supervisor_agent=main_agent,
        )
        return self.graph
