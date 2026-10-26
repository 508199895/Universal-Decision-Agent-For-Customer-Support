"""Workflow entry and the current handoff graph."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import inspect
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import RunnableConfig

from agentic.agents.escalation_agent import build_escalation_agent
from agentic.agents.customer_support_agent import build_customer_support_agent
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
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
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

    response = await agent.ainvoke(
        {"conversation": _conversation_text(state["messages"])}, config=config,
    )
    if not isinstance(response, dict) or set(response) != {"summary", "description"}:
        raise StateContractError("Escalation agent must return exactly summary and description")
    material = {}
    for field in ("summary", "description"):
        value = response[field]
        if not isinstance(value, str) or not value.strip():
            raise StateContractError(f"Escalation agent {field} must be a non-empty string")
        material[field] = value.strip()

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
                   dependencies: WorkflowDependencies | None = None, agent=None,
                   support_agent=None):
    """Compile the support and handoff workflow with shared message state."""
    if checkpointer is None or ticket_event_store is None:
        raise ValueError("workflow requires a durable checkpointer and TicketEventStore")
    if agent is None:
        dependencies = dependencies or _default_dependencies()
        agent = build_escalation_agent(model=dependencies.model)
    async def run_escalation(state, config):
        return await escalation_node(state, config, agent=agent)

    escalation_builder = StateGraph(WorkflowState)
    escalation_builder.add_node("prepare_handoff", run_escalation)
    escalation_builder.add_edge(START, "prepare_handoff")
    escalation_builder.add_edge("prepare_handoff", END)
    escalation_graph = escalation_builder.compile()

    graph = StateGraph(WorkflowState)
    graph.add_node("escalation", escalation_graph)
    if support_agent is not None:
        graph.add_node("customer_support", support_agent)
        graph.add_edge(START, "customer_support")
        graph.add_edge("customer_support", "escalation")
    else:
        graph.add_edge(START, "escalation")
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer)


def _default_agent_database() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "core" / "agent_tickets.db"


@asynccontextmanager
async def persistent_workflow(database: str | os.PathLike, *, dependencies=None, agent=None,
                              ticket_event_store: TicketEventStore | None = None,
                              support_agent=None):
    """Open a workflow and keep its saver alive for the context lifetime."""
    store = ticket_event_store or TicketEventStore(database)
    owns_dependencies = dependencies is None and agent is None
    active_dependencies = dependencies
    if active_dependencies is None and agent is None:
        active_dependencies = _default_dependencies()
    try:
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            yield build_workflow(
                checkpointer=checkpointer,
                ticket_event_store=store,
                dependencies=active_dependencies,
                agent=agent,
                support_agent=support_agent,
            )
    finally:
        if owns_dependencies:
            await _close_model_clients(active_dependencies.model)


@asynccontextmanager
async def default_workflow_runtime():
    """Validate and assemble the static support agent before opening the CLI runtime."""
    event_store = TicketEventStore(_default_agent_database())
    dependencies = _default_dependencies()
    try:
        support_tools = await acollect_support_tools()
        _support_agent = build_customer_support_agent(
            model=dependencies.model, tools=support_tools,
        )
        async with persistent_workflow(
            event_store.database, dependencies=dependencies,
            ticket_event_store=event_store,
            support_agent=_support_agent,
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
        journal = TicketJournal(ticket_event_store, user_id=user_id, ticket_id=actual_ticket_id)
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

    journal = TicketJournal(ticket_event_store, user_id=user_id, ticket_id=record.ticket_id)
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
    """Move messages from the previous nested field into AgentState.messages."""
    normalized = dict(state)
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
    return normalized, legacy_messages is not None or "messages" not in state


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
        "configurable": {"thread_id": thread_id},
        "callbacks": [journal],
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
        # This runs for success, graph errors, and cooperative cancellation.
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
    journal = TicketJournal(event_store, user_id=user_id, ticket_id=ticket_id)
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


