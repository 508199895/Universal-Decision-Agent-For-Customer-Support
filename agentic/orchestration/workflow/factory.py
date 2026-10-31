"""Application-time assembly of specialist agents and the outer workflow graph."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from agentic.agents.agent import get_agent_soul
from agentic.agents.prompt import apply_prompt_template
from agentic.agents.supervisor.contract import SupervisorResponse
from agentic.agents.validate.contract import ValidateResult
from agentic.capabilities.tools.account_tools import bind_current_user_account_tools
from agentic.capabilities.middleware.token_control import (
    TicketRuntimeContext,
    build_ticket_middleware,
)
from agentic.capabilities.tools.support_toolset import validate_support_tools
from agentic.capabilities.tools.clarification import ask_clarification
from agentic.capabilities.tools.feishu_handoff import (
    FeishuTextSender,
    build_feishu_handoff_tool,
)
from agentic.agents.agent_wrappers import (
    build_account_agent_tool, build_public_knowledge_tool,
    build_query_validate_subgraph,
)
from agentic.agents.escalation.handoff import build_handoff_messages
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


def _successful_handoff(messages: Any) -> dict[str, str] | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool" or getattr(
            message, "name", None
        ) != "request_human_handoff":
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("status") == "sent":
            return result
    return None


def _attempted_handoff(messages: Any) -> bool:
    return isinstance(messages, list) and any(
        getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "request_human_handoff"
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
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or result.get("status") != "asked":
            continue
        question = result.get("question")
        if isinstance(question, str) and question.strip():
            return {"question": question.strip()}
    return None


def _attempted_clarification(messages: Any) -> bool:
    return isinstance(messages, list) and any(
        getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "ask_clarification"
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
    structured = result.get("structured_response") if isinstance(result, dict) else result
    response = (structured if isinstance(structured, SupervisorResponse)
                else SupervisorResponse.model_validate(structured))

    supervisor_messages = result.get("messages", []) if isinstance(result, dict) else []
    handoff_result = _successful_handoff(supervisor_messages)
    clarification_result = _successful_clarification(supervisor_messages)
    if handoff_result is None and _attempted_handoff(supervisor_messages):
        raise RuntimeError("Feishu handoff tool did not confirm successful delivery")
    if clarification_result is None and _attempted_clarification(supervisor_messages):
        raise RuntimeError("Clarification tool did not confirm a customer-facing question")
    if handoff_result is not None and clarification_result is not None:
        raise RuntimeError("Supervisor cannot clarify and hand off in the same ticket run")
    if (
        response.requires_handoff
        and handoff_result is None
        and clarification_result is None
    ):
        raise RuntimeError("Supervisor requested handoff without calling the Feishu tool")

    ticket_id = _context_ticket_id(context)
    updated = merge_patch(state, {"clarify": {"prompt": None}}, writer="boundary")
    if handoff_result is not None:
        if response.answer == "":
            customer_message = "当前请求将转交人工核查。"
        else:
            customer_message = response.answer
        if not response.requires_handoff:
            customer_message = "当前请求将转交人工核查。"
        updated = merge_patch(updated, {"escalation": {
            "summary": handoff_result["summary"],
            "description": handoff_result["description"],
        }}, writer="escalation")
        updated = merge_patch(updated, {"ticket": {"status": "handed_off"},
            "supervisor": {
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
        updated = merge_patch(updated, {"ticket": {"status": "completed"},
            "supervisor": {
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
    """Assemble configured agents and compile the ticket graph."""

    def __init__(self) -> None:
        self.graph = None

    def initialize(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        checkpointer: BaseCheckpointSaver,
        handoff_sender: FeishuTextSender | None = None,
    ):
        """Build and cache the graph; model and tool registries stay local to assembly."""
        if self.graph is not None:
            return self.graph
        if model is None or checkpointer is None:
            raise ValueError("workflow initialization requires model and checkpointer")

        available_tools: dict[str, BaseTool] = {}

        def add_tool(tool: BaseTool) -> None:
            available_tools.setdefault(tool.name, tool)

        def add_tools(tools_to_add: Sequence[BaseTool]) -> None:
            for tool in tools_to_add:
                add_tool(tool)

        add_tools(bind_current_user_account_tools(validate_support_tools(tools)))
        add_tool(ask_clarification)
        context_schemas = {"ticket_runtime": TicketRuntimeContext, None: None}
        response_formats = {
            "supervisor_response": SupervisorResponse,
            "validate_result": ValidateResult,
            None: None,
        }

        def build_agent(name: str, allowed_tools: dict[str, BaseTool]):
            config = get_agent_soul(name)
            model_name = config.get("model", "runtime")
            if model_name != "runtime":
                raise ValueError(f"agent {name} references unknown model: {model_name}")

            tool_names = config.get("tools", [])
            if not isinstance(tool_names, list) or any(
                not isinstance(tool_name, str) for tool_name in tool_names
            ):
                raise ValueError(f"agent {name} tools must be a list of tool-name identifiers")
            if len(tool_names) != len(set(tool_names)):
                raise ValueError(f"agent {name} tools must not contain duplicate names")
            missing_tools = sorted(set(tool_names) - allowed_tools.keys())
            if missing_tools:
                raise ValueError(
                    f"agent {name} references unavailable tools: {', '.join(missing_tools)}"
                )

            middleware_names = config.get("middleware", ["ticket_runtime"])
            if not isinstance(middleware_names, list) or any(
                not isinstance(item, str) for item in middleware_names
            ):
                raise ValueError(f"agent {name} middleware must be a list of identifiers")
            middleware = []
            for middleware_name in middleware_names:
                if middleware_name != "ticket_runtime":
                    raise ValueError(
                        f"agent {name} references unknown middleware: {middleware_name}"
                    )
                middleware.extend(build_ticket_middleware(model))

            context_name = config.get("context_schema", "ticket_runtime")
            if (
                context_name is not None and not isinstance(context_name, str)
            ) or context_name not in context_schemas:
                raise ValueError(f"agent {name} references unknown context schema: {context_name}")
            response_name = config.get("response_format")
            if (
                response_name is not None and not isinstance(response_name, str)
            ) or response_name not in response_formats:
                raise ValueError(f"agent {name} references unknown response format: {response_name}")

            kwargs = {
                "model": model,
                "tools": [allowed_tools[item] for item in tool_names],
                "middleware": middleware,
                "system_prompt": apply_prompt_template(config["prompt"]),
            }
            if response_formats[response_name] is not None:
                kwargs["response_format"] = response_formats[response_name]
            if context_schemas[context_name] is not None:
                kwargs["context_schema"] = context_schemas[context_name]
            return create_agent(**kwargs)

        query_agent = build_agent("query", available_tools)
        validate_agent = build_agent("validate", available_tools)
        account_agent = build_agent("account", available_tools)
        handoff_tool = build_feishu_handoff_tool(handoff_sender)
        add_tool(handoff_tool)
        query_validate_subgraph = build_query_validate_subgraph(
            query_agent=query_agent,
            validate_agent=validate_agent,
        )
        agent_tools = (
            build_public_knowledge_tool(subgraph=query_validate_subgraph),
            build_account_agent_tool(account_agent=account_agent),
        )
        add_tools(agent_tools)
        supervisor_agent = build_agent("supervisor", available_tools)
        self.graph = build_workflow(
            checkpointer=checkpointer,
            supervisor_agent=supervisor_agent,
        )
        return self.graph
