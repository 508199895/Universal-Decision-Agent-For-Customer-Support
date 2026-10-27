"""Workflow entry and the current handoff graph."""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import uuid
import inspect
import logging
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import ValidationError

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import RunnableConfig

from agentic.agents.agent import AgentSet, build_agent_set
from agentic.agents.escalation.contract import EscalationMaterial
from agentic.agents.supervisor.contract import SupervisorIntentResult
from agentic.agents.validate.contract import ValidateResult
from agentic.capabilities.middleware.token_control import (
    DEFAULT_MAX_OUTPUT_TOKENS, TICKET_TOKEN_LIMIT, TicketRuntimeContext,
    TicketTokenLimitExceeded,
)
from agentic.capabilities.support_toolset import acollect_support_tools
from agentic.orchestration.handoff import HANDOFF_NOTICE, build_handoff_messages
from agentic.data.tickets.message_journal import TicketJournal
from agentic.data.tickets.ticket_store import (
    TicketAlreadyTerminal,
    TicketEventStore,
    TicketPersistenceError,
)
from agentic.orchestration.workflow_state import (
    StateContractError, WorkflowState, initial_state, merge_patch, validate_state,
)
from agentic.observability.langsmith_tracing import build_trace_config


@dataclass(frozen=True)
class WorkflowDependencies:
    model: BaseChatModel


TERMINAL_TICKET_STATUSES = {"completed", "handed_off"}
_user_locks_guard = Lock()
_user_locks_by_loop: WeakKeyDictionary[Any, dict[str, asyncio.Lock]] = WeakKeyDictionary()


def _public_evidence(messages: list[Any]) -> list[dict[str, str]]:
    """Project only Qdrant records with source references returned by the adapter."""
    evidence = []
    for message in messages:
        name = getattr(message, "name", None)
        content = getattr(message, "content", None)
        if not isinstance(name, str) or "qdrant-find" not in name:
            continue
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text = part if isinstance(part, str) else (
                part.get("text") if isinstance(part, dict) else None
            )
            if not isinstance(text, str):
                continue
            pattern = r"<entry>\s*<content>(.*?)</content>\s*<metadata>(.*?)</metadata>\s*</entry>"
            for match in re.finditer(pattern, text, flags=re.DOTALL):
                try:
                    metadata = json.loads(html.unescape(match.group(2).strip()))
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(metadata, dict):
                    continue
                source_ref = next((
                    metadata.get(key) for key in ("article_id", "source_ref", "uri", "path")
                    if isinstance(metadata.get(key), str) and metadata.get(key).strip()
                ), None)
                body = html.unescape(match.group(1)).strip()
                if source_ref and body:
                    evidence.append({
                        "source_type": "public_knowledge",
                        "source_ref": source_ref,
                        "content": body,
                    })
    return evidence


async def _validate_public_answer(reviewer, *, question: str, answer: str,
                                  evidence: list[dict[str, str]], messages: list[Any],
                                  config: RunnableConfig) -> ValidateResult:
    """Review the answer against retrieved material and actual conversation context."""
    material = {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "conversation": _conversation_text(messages),
    }
    result = await asyncio.wait_for(reviewer.ainvoke(
        {"messages": [HumanMessage(content="校核以下候选：\n" + json.dumps(material, ensure_ascii=False))]},
        config=config,
        context=_ticket_runtime_context(config),
    ), timeout=60)
    structured = result.get("structured_response") if isinstance(result, dict) else result
    return (
        structured if isinstance(structured, ValidateResult)
        else ValidateResult.model_validate(structured)
    )


def _candidate_answer(state: dict[str, Any]) -> str:
    """Require one plain, non-tool assistant answer before content review."""
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        break
    raise StateContractError("single support agent did not produce a valid answer candidate")


async def validate_node(state: WorkflowState, config: RunnableConfig, *, reviewer):
    """Review the support agent's answer against actual state and Qdrant results."""
    validate_state(state)
    draft = _candidate_answer(state)
    evidence = _public_evidence(state["messages"])
    review = await _validate_public_answer(
        reviewer, question=state["ticket"]["input"], answer=draft,
        evidence=evidence, messages=state["messages"], config=config,
    )
    answer = draft
    passed = all((review.context_relevance, review.faithfulness, review.answer_relevance))
    patch = {
        "query": {
            "answer": answer,
            "validated": passed,
            "evidence": evidence,
            "description": "候选答案已完成结构检查和内容校核。",
            "next_node": "supervisor",
        },
        "validate": {
            "context_relevance": review.context_relevance,
            "faithfulness": review.faithfulness,
            "answer_relevance": review.answer_relevance,
            "next_node": "supervisor",
            "feedback": review.feedback or (None if passed else "候选答案未通过校核。"),
        },
    }
    updated = merge_patch(state, patch, writer="boundary")
    if passed:
        updated = merge_patch(updated, {
            "supervisor": {"intent": "query", "message": answer, "next_node": "finish"},
            "ticket": {"status": "completed"},
        }, writer="boundary")
    return updated


async def _safe_validate_node(state: WorkflowState, config: RunnableConfig, *, reviewer):
    if _has_nonpublic_tool_result(state.get("messages", [])):
        return merge_patch(state, {
            "query": {"description": "当前请求包含非公共知识工具结果，转受控交接。"},
        }, writer="boundary")
    try:
        return await validate_node(state, config, reviewer=reviewer)
    except StateContractError as error:
        return merge_patch(state, {
            "query": {"description": f"候选答案结构异常：{error}"},
        }, writer="boundary")
    except ValidationError:
        return merge_patch(state, {
            "query": {"description": "答案校核结果结构异常。"},
        }, writer="boundary")
    except TicketTokenLimitExceeded as error:
        return merge_patch(state, {
            "query": {"description": str(error), "validated": False},
        }, writer="boundary")
    except Exception:
        return merge_patch(state, {
            "query": {"description": "答案校验服务失败。"},
            "validate": {"feedback": "答案校验服务失败。"},
        }, writer="boundary")


async def _supervisor_node(state: WorkflowState, config: RunnableConfig, *, supervisor):
    """Run the current V6 supervisor slice: public-query intent routing."""
    try:
        result = await asyncio.wait_for(supervisor.ainvoke(
            {"messages": [HumanMessage(content=json.dumps({
                "current_request": state["ticket"]["input"],
                "conversation": _conversation_text(state["messages"]),
            }, ensure_ascii=False))]},
            config=config,
            context=_ticket_runtime_context(config),
        ), timeout=60)
        structured = result.get("structured_response") if isinstance(result, dict) else result
        intent = (
            structured if isinstance(structured, SupervisorIntentResult)
            else SupervisorIntentResult.model_validate(structured)
        )
    except Exception:
        return merge_patch(state, {
            "supervisor": {"intent": "unknown", "next_node": "escalation"},
            "query": {"description": "公共知识意图校核失败。"},
        }, writer="boundary")

    patch = {
        "supervisor": {
            "intent": "query" if intent.is_public_knowledge else "escalation",
            "next_node": "query" if intent.is_public_knowledge else "escalation",
        },
    }
    if not intent.is_public_knowledge:
        patch["query"] = {"description": "当前请求超出公共知识问答范围。"}
    return merge_patch(state, patch, writer="boundary")


def _supervisor_route(state: WorkflowState) -> str:
    return "answer" if state["supervisor"].get("next_node") == "query" else "escalation"


def _validate_route(state: WorkflowState) -> str:
    query = state.get("query", {})
    review = state.get("validate", {})
    if query.get("validated") is True and all(
        review.get(field) is True
        for field in ("context_relevance", "faithfulness", "answer_relevance")
    ):
        return "complete"
    return "escalation"


def _has_nonpublic_tool_result(messages: list[Any]) -> bool:
    return any(
        getattr(message, "type", None) == "tool"
        and "qdrant-find" not in (getattr(message, "name", "") or "")
        for message in messages
    )


async def _invoke_support_agent(agent, state, config):
    return await asyncio.wait_for(
        agent.ainvoke(state, config=config, context=_ticket_runtime_context(config)),
        timeout=60,
    )


def _ticket_runtime_context(config: RunnableConfig) -> TicketRuntimeContext:
    journal = (config.get("configurable") or {}).get("ticket_journal")
    if journal is None:
        raise RuntimeError("ticket model call is missing its per-run Journal")
    return TicketRuntimeContext(journal=journal)


def _serialize_user_calls(function):
    """Serialize checkpoint read/update/invoke cycles for each user thread."""
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        user_id = signature.bind_partial(*args, **kwargs).arguments.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        loop = asyncio.get_running_loop()
        with _user_locks_guard:
            locks = _user_locks_by_loop.setdefault(loop, {})
            lock = locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            return await function(*args, **kwargs)

    return wrapped


def _default_dependencies() -> WorkflowDependencies:
    from langchain_openai import ChatOpenAI

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is missing.")
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
    )
    return WorkflowDependencies(model=model)


def _conversation_text(messages: list[Any]) -> str:
    lines = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("type") or message.get("role") or "message"
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "message")
            content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


async def escalation_node(state: WorkflowState, config: RunnableConfig, *, agent):
    """Summarize the full conversation and return a checkpointable handoff state."""
    validate_state(state)
    ticket = state["ticket"]
    if ticket["status"] not in {"processing", "waiting_reply"}:
        raise StateContractError("handoff requires a non-terminal ticket")

    try:
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content="Full ticket conversation:\n" + _conversation_text(state["messages"]))]},
            config=config, context=_ticket_runtime_context(config),
        )
        structured = (
            response.get("structured_response", response)
            if isinstance(response, dict) else response
        )
        material = (
            structured.model_dump() if isinstance(structured, EscalationMaterial)
            else EscalationMaterial.model_validate(structured).model_dump()
        )
        material = {field: value.strip() for field, value in material.items()}
        if not material["summary"] or not material["description"]:
            raise StateContractError("Escalation agent fields must not be blank")
    except TicketTokenLimitExceeded as error:
        material = {
            "summary": f"工单 {ticket['ticket_id']} 因累计 Token 达到上限而转人工处理。",
            "description": (
                f"客户请求：{ticket['input']}\n"
                f"Token 控制：{error}\n"
                "自动处理已停止；请人工依据完整工单消息和已记录工具结果继续核查。"
            ),
        }

    updated = merge_patch(state, {"escalation": material}, writer="escalation")
    updated = merge_patch(updated, {"ticket": {"status": "handed_off"}}, writer="boundary")
    messages = list(updated["messages"])
    generated_message = AIMessage(
        content=json.dumps(material, ensure_ascii=False),
        id=f"escalation:{ticket['ticket_id']}:result",
    )
    for handoff_message in (generated_message, *build_handoff_messages(updated)):
        existing = next(
            (index for index, item in enumerate(messages) if _message_id(item) == handoff_message.id),
            None,
        )
        if existing is None:
            messages.append(handoff_message)
        else:
            messages[existing] = handoff_message
    updated["messages"] = messages
    validate_state(updated)
    return updated


def build_workflow(*, checkpointer: BaseCheckpointSaver, ticket_event_store: TicketEventStore,
                   agents: AgentSet):
    """Compile a workflow from prebuilt agents and durable runtime dependencies."""
    if checkpointer is None or ticket_event_store is None:
        raise ValueError("workflow requires a durable checkpointer and TicketEventStore")
    if agents is None or agents.escalation is None:
        raise ValueError("workflow requires a prebuilt escalation agent")
    async def run_escalation(state, config):
        return await escalation_node(state, config, agent=agents.escalation)

    escalation_builder = StateGraph(WorkflowState)
    escalation_builder.add_node("prepare_handoff", run_escalation)
    escalation_builder.add_edge(START, "prepare_handoff")
    escalation_builder.add_edge("prepare_handoff", END)
    escalation_graph = escalation_builder.compile()

    graph = StateGraph(WorkflowState)
    graph.add_node("escalation", escalation_graph)
    if agents.query is not None:
        if agents.supervisor is None or agents.validate is None:
            raise ValueError(
                "support workflow requires prebuilt supervisor and validate agents"
            )

        async def run_support(state, config):
            try:
                return await _invoke_support_agent(agents.query, state, config)
            except TicketTokenLimitExceeded as error:
                return merge_patch(state, {
                    "query": {"description": str(error), "validated": False},
                }, writer="boundary")

        async def run_supervisor(state, config):
            return await _supervisor_node(
                state, config, supervisor=agents.supervisor,
            )

        async def run_validate(state, config):
            return await _safe_validate_node(
                state, config, reviewer=agents.validate,
            )

        def support_route(state):
            description = state.get("query", {}).get("description") or ""
            return "escalation" if "Token 已达到上限" in description else "review"

        graph.add_node("query", run_support)
        graph.add_node("supervisor", run_supervisor)
        graph.add_node("validate", run_validate)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor", _supervisor_route,
            {"answer": "query", "escalation": "escalation"},
        )
        graph.add_conditional_edges(
            "query", support_route,
            {"review": "validate", "escalation": "escalation"},
        )
        graph.add_conditional_edges(
            "validate", _validate_route,
            {"complete": END, "escalation": "escalation"},
        )
    else:
        graph.add_edge(START, "escalation")
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer)


def _default_agent_database() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "core" / "agent_tickets.db"


@asynccontextmanager
async def persistent_workflow(database: str | os.PathLike, *, agents: AgentSet,
                              ticket_event_store: TicketEventStore | None = None):
    """Open a workflow and keep its saver alive for the context lifetime."""
    store = ticket_event_store or TicketEventStore(database)
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        yield build_workflow(
            checkpointer=checkpointer,
            ticket_event_store=store,
            agents=agents,
        )


@asynccontextmanager
async def default_workflow_runtime():
    """Validate and assemble the static support agent before opening the CLI runtime."""
    event_store = TicketEventStore(_default_agent_database())
    dependencies = _default_dependencies()
    try:
        support_tools = await acollect_support_tools()
        agents = build_agent_set(model=dependencies.model, tools=support_tools)
        async with persistent_workflow(
            event_store.database, agents=agents,
            ticket_event_store=event_store,
        ) as workflow:
            yield workflow, event_store
    finally:
        await _close_model_clients(dependencies.model)


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


@_serialize_user_calls
async def aenter_ticket(external_user_id: str, message: HumanMessage, *, user_id: str,
                        user_name: str | None = None,
                        workflow, ticket_event_store: TicketEventStore,
                        ticket_id: str | None = None, new_ticket: bool = False) -> WorkflowState:
    """Resume the user's latest ticket checkpoint or start a new ticket thread state."""
    if not isinstance(message, HumanMessage) or not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("message must be a non-empty native HumanMessage")
    if not isinstance(message.id, str) or not message.id.strip():
        raise ValueError("customer messages must include a stable native message id")

    thread_id = ticket_event_store.thread_id(user_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await workflow.aget_state(config)
    latest_state = dict(snapshot.values or {})
    needs_upgrade = False
    if latest_state:
        latest_state, needs_upgrade = _normalize_checkpoint_state(latest_state)
        if needs_upgrade:
            validate_state(latest_state)
    latest_ticket = (latest_state.get("ticket", {}) or {}) if latest_state else {}
    latest_ticket_id = latest_ticket.get("ticket_id")
    continue_latest = bool(
        latest_ticket_id and not new_ticket
        and (ticket_id is None or ticket_id == latest_ticket_id)
        and latest_ticket.get("status") not in TERMINAL_TICKET_STATUSES
    )
    record = None
    if continue_latest:
        try:
            record = await asyncio.to_thread(ticket_event_store.get_ticket, latest_ticket_id)
        except Exception as error:
            from agentic.data.tickets.ticket_store import TicketHistoryConflict
            if not isinstance(error, TicketHistoryConflict):
                raise
            record = await asyncio.to_thread(
                ticket_event_store.create_ticket,
                ticket_id=latest_ticket_id, user_id=user_id,
                # State.ticket.status is workflow state, not the overall ticket status.
                status="running",
            )
        if record.user_id != user_id:
            from agentic.data.tickets.ticket_store import TicketHistoryConflict
            raise TicketHistoryConflict("latest checkpoint ticket does not belong to the authenticated user")
        if record.status == "completed":
            raise TicketAlreadyTerminal(f"ticket {record.ticket_id} is completed")
    pending_record = None
    # The CLI's selected ticket ID distinguishes checkpoint continuation from pending recovery.
    if not new_ticket and not continue_latest:
        pending_record = await asyncio.to_thread(
            ticket_event_store.latest_uninitialized_ticket, user_id,
        )
    if not latest_state and pending_record is None:
        checkpoint_record = await asyncio.to_thread(
            ticket_event_store.latest_checkpoint_ticket, user_id,
        )
        if checkpoint_record is not None and checkpoint_record.status != "completed":
            raise TicketPersistenceError(
                f"checkpoint marked initialized but missing for ticket {checkpoint_record.ticket_id}"
            )
    if pending_record is not None:
        pending_record = await asyncio.to_thread(
            ticket_event_store.get_ticket, pending_record.ticket_id,
        )
        if pending_record.status == "completed":
            raise TicketAlreadyTerminal(f"ticket {pending_record.ticket_id} is completed")
        message_events = [
            event for event in await asyncio.to_thread(
                ticket_event_store.events, user_id=user_id, ticket_id=pending_record.ticket_id,
            )
            if event["event_type"] == "ticket.message.received"
        ]
        initial_metadata = message_events[0]["metadata"] if message_events else {}
        if initial_metadata.get("owner_id", external_user_id) != external_user_id:
            from agentic.data.tickets.ticket_store import TicketHistoryConflict
            raise TicketHistoryConflict("pending ticket does not belong to the authenticated external user")
        history = [
            HumanMessage(
                content=event["content"]["content"], id=event["content"].get("id"),
            )
            for event in message_events
            if event["event_type"] == "ticket.message.received"
            and isinstance(event["content"], dict)
            and event["content"].get("type") == "human"
        ]
        if not history:
            raise RuntimeError(
                f"pending ticket {pending_record.ticket_id} has no durable initial message"
            )
        state = initial_state(
            ticket_id=pending_record.ticket_id, owner_id=external_user_id,
            owner_name=initial_metadata.get("owner_name", user_name), message=history[0],
        )
        messages = list(history)
        if not any(_message_id(item) == message.id for item in messages):
            messages.append(message)
        state["messages"] = messages
        state["ticket"]["input"] = message.content
        await _replace_workflow_state(workflow, config, state)
        await asyncio.to_thread(
            ticket_event_store.mark_checkpoint_initialized, pending_record.ticket_id,
        )
        await asyncio.to_thread(ticket_event_store.set_status, pending_record.ticket_id, "running")
        latest_ticket = state["ticket"]
        latest_state = state
        latest_ticket_id = pending_record.ticket_id
        record = pending_record
        continue_latest = True

    if continue_latest:
        if needs_upgrade:
            validate_state(latest_state)
            await _replace_workflow_state(workflow, config, latest_state)
            needs_upgrade = False
        if not await asyncio.to_thread(ticket_event_store.checkpoint_initialized, record.ticket_id):
            await asyncio.to_thread(ticket_event_store.mark_checkpoint_initialized, record.ticket_id)
    else:
        actual_ticket_id = ticket_id or str(uuid.uuid4())
        if ticket_id is not None:
            try:
                await asyncio.to_thread(ticket_event_store.get_ticket, ticket_id)
            except Exception as error:
                from agentic.data.tickets.ticket_store import TicketHistoryConflict
                if not isinstance(error, TicketHistoryConflict):
                    raise
            else:
                from agentic.data.tickets.ticket_store import TicketHistoryConflict
                raise TicketHistoryConflict(f"ticket {ticket_id} already exists and cannot be restarted")
        state = initial_state(
            ticket_id=actual_ticket_id, owner_id=external_user_id,
            owner_name=user_name, message=message,
        )
        record = await asyncio.to_thread(
            ticket_event_store.create_ticket,
            ticket_id=actual_ticket_id, user_id=user_id, status="pending",
            initial_message=message.model_dump(),
            initial_metadata={"owner_id": external_user_id, "owner_name": user_name},
        )
        journal = TicketJournal(
            ticket_event_store, user_id=user_id, ticket_id=actual_ticket_id,
            token_limit=TICKET_TOKEN_LIMIT,
        )
        try:
            # The ticket and its first message are durable before checkpoint creation.
            # If checkpoint creation is interrupted, the next entry rebuilds from the event.
            await _replace_workflow_state(workflow, config, state)
            await asyncio.to_thread(
                ticket_event_store.mark_checkpoint_initialized, actual_ticket_id,
            )
            await asyncio.to_thread(ticket_event_store.set_status, actual_ticket_id, "running")
            await journal.record_lifecycle("run.start", "trace", {"ticket_id": actual_ticket_id})
            result = await _invoke_with_journal(workflow, thread_id, journal)
        except asyncio.CancelledError:
            if await asyncio.to_thread(ticket_event_store.checkpoint_initialized, actual_ticket_id):
                try:
                    await asyncio.to_thread(ticket_event_store.set_status, actual_ticket_id, "failed")
                except Exception:
                    pass
            raise
        except Exception:
            if await asyncio.to_thread(ticket_event_store.checkpoint_initialized, actual_ticket_id):
                try:
                    await asyncio.to_thread(ticket_event_store.set_status, actual_ticket_id, "failed")
                except Exception:
                    pass
            raise
        await _commit_ticket_terminal(ticket_event_store, actual_ticket_id, result)
        return result

    journal = TicketJournal(
        ticket_event_store, user_id=user_id, ticket_id=record.ticket_id,
        token_limit=TICKET_TOKEN_LIMIT,
    )
    try:
        if latest_ticket_id != record.ticket_id:
            from agentic.data.tickets.ticket_store import TicketHistoryConflict
            raise TicketHistoryConflict("checkpoint ticket_id does not match the ticket record")
        latest_ticket = {**latest_ticket, "input": message.content}
        latest_ticket["status"] = "processing"
        # AgentState.messages applies add_messages: append by ID, replacing a
        # prior message with the same stable ID instead of duplicating it.
        await workflow.aupdate_state(
            config, {"ticket": latest_ticket, "messages": [message]}, as_node=START,
        )
        await asyncio.to_thread(ticket_event_store.set_status, record.ticket_id, "running")
        await asyncio.to_thread(ticket_event_store.mark_checkpoint_initialized, record.ticket_id)
        await journal.record_lifecycle("run.start", "trace", {"ticket_id": record.ticket_id})
        result = await _invoke_with_journal(workflow, thread_id, journal)

        # LangGraph returns only after persisting the node output checkpoint. Commit the
        # core terminal state afterward; retries can repair a failed core transaction above.
        workflow_status = result.get("ticket", {}).get("status")
        await _commit_ticket_terminal(ticket_event_store, record.ticket_id, result)

        current = await asyncio.to_thread(ticket_event_store.status, record.ticket_id)
        if current != "completed" or workflow_status not in {"completed", "handed_off"}:
            # waiting_reply remains a resumable workflow state; the ticket's
            # overall processing status remains running until it terminates.
            if workflow_status != "waiting_reply" or current != "running":
                raise RuntimeError("workflow did not persist its overall state")
        return result
    except asyncio.CancelledError:
        try:
            await asyncio.to_thread(ticket_event_store.set_status, record.ticket_id, "failed")
        except Exception:
            pass
        raise
    except Exception:
        try:
            await asyncio.to_thread(ticket_event_store.set_status, record.ticket_id, "failed")
        except Exception:
            pass
        raise


def _message_id(message: Any) -> str | None:
    return message.get("id") if isinstance(message, dict) else getattr(message, "id", None)


async def _commit_ticket_terminal(event_store: TicketEventStore, ticket_id: str,
                                  state: dict[str, Any]) -> None:
    status = state.get("ticket", {}).get("status")
    if status == "handed_off":
        await asyncio.to_thread(
            event_store.commit_handoff, ticket_id,
            state=state, final_message=HANDOFF_NOTICE,
        )
    elif status == "completed":
        await asyncio.to_thread(
            event_store.commit_completed, ticket_id,
            customer_message_for_state(state),
        )


def customer_message_for_state(state: dict[str, Any]) -> str | None:
    """Project only a validated answer or the fixed handoff notice to the customer."""
    ticket = state.get("ticket", {})
    if ticket.get("status") == "handed_off":
        return HANDOFF_NOTICE
    if ticket.get("status") == "waiting_reply":
        return state.get("confirm", {}).get("prompt") or state.get("clarify", {}).get("prompt")
    query = state.get("query", {})
    answer = query.get("answer")
    if ticket.get("status") == "completed" and query.get("validated") is True:
        return answer if isinstance(answer, str) and answer.strip() else None
    return None


def _normalize_checkpoint_state(state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize legacy messages and discard the retired budget state group."""
    normalized = dict(state)
    had_budget = "budget" in normalized
    normalized.pop("budget", None)
    ticket = dict(normalized.get("ticket") or {})
    legacy_messages = ticket.pop("messages", None)
    current_messages = list(normalized.get("messages") or [])
    if legacy_messages is not None:
        by_id = {_message_id(item): item for item in legacy_messages}
        for item in current_messages:
            by_id[_message_id(item)] = item
        normalized["messages"] = list(by_id.values())
    elif "messages" not in normalized:
        normalized["messages"] = current_messages
    normalized["ticket"] = ticket
    return normalized, legacy_messages is not None or "messages" not in state or had_budget


async def _replace_workflow_state(workflow, config: RunnableConfig,
                                  state: dict[str, Any]) -> None:
    """Replace a user's checkpoint, explicitly clearing AgentState.messages first."""
    update = dict(state)
    update["messages"] = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *state["messages"],
    ]
    await workflow.aupdate_state(config, update, as_node=START)


async def _invoke_with_journal(workflow, thread_id: str, journal: TicketJournal,
                               payload=None):
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "ticket_journal": journal},
        "callbacks": [journal],
        "recursion_limit": int(os.getenv("WORKFLOW_RECURSION_LIMIT", "12")),
    }
    trace_config = build_trace_config(journal.ticket_id)
    if trace_config:
        config["run_name"] = trace_config["run_name"]
        config["metadata"] = trace_config["metadata"]
        config["tags"] = trace_config["tags"]
        config["callbacks"].extend(trace_config["callbacks"])
    try:
        result = await workflow.ainvoke(payload, config=config, durability="async")
        await journal.record_lifecycle("run.end", "outputs", {"ticket_id": journal.ticket_id})
        return result
    except asyncio.CancelledError:
        await journal.record_lifecycle("run.cancel", "trace", {"ticket_id": journal.ticket_id})
        raise
    except Exception as error:
        await journal.record_lifecycle(
            "run.error", "error", {"error": str(error)}, {"error_type": type(error).__name__},
        )
        raise
    finally:
        # Usage is one snapshot update, even after graph errors or cancellation.
        try:
            await journal.persist_usage_snapshot()
        except Exception:
            logging.getLogger(__name__).exception(
                "could not persist token usage for ticket %s", journal.ticket_id,
            )
        await journal.flush()


@_serialize_user_calls
async def aresume_ticket(user_id: str, *, workflow, reply=None,
                         ticket_event_store: TicketEventStore):
    from langgraph.types import Command
    from agentic.data.tickets.ticket_store import TicketPersistenceError

    event_store = ticket_event_store
    thread_id = event_store.thread_id(user_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await workflow.aget_state(config)
    if not snapshot.values:
        raise TicketPersistenceError("No checkpoint exists for this ticket")
    ticket = snapshot.values.get("ticket", {}) or {}
    ticket_id = ticket.get("ticket_id")
    record = await asyncio.to_thread(event_store.get_ticket, ticket_id)
    if record.user_id != user_id:
        raise ValueError("checkpoint ticket does not belong to this user")
    if record.status == "completed":
        raise ValueError(f"ticket is terminal: {record.status}")
    if ticket.get("status") in {"completed", "handed_off"}:
        raise ValueError(f"ticket is terminal: {ticket['status']}")
    if not snapshot.next:
        raise ValueError("Ticket has no unfinished execution to resume")
    await asyncio.to_thread(event_store.set_status, ticket_id, "running")
    await asyncio.to_thread(event_store.mark_checkpoint_initialized, ticket_id)
    journal = TicketJournal(
        event_store, user_id=user_id, ticket_id=ticket_id,
        token_limit=TICKET_TOKEN_LIMIT,
    )
    payload = Command(resume=reply) if reply is not None else None
    await journal.record_lifecycle("run.start", "trace", {"ticket_id": ticket_id})
    try:
        result = await _invoke_with_journal(workflow, thread_id, journal, payload=payload)
        workflow_status = result.get("ticket", {}).get("status")
        await _commit_ticket_terminal(event_store, ticket_id, result)
        return result
    except asyncio.CancelledError:
        await asyncio.to_thread(event_store.set_status, ticket_id, "failed")
        raise
    except Exception:
        await asyncio.to_thread(event_store.set_status, ticket_id, "failed")
        raise


