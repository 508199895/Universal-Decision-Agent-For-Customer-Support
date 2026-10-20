"""V5 workflow entry and the current handoff-only graph."""

from __future__ import annotations

import atexit
import asyncio
import json
import os
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agentic.agents.escalation_agent import build_escalation_agent
from agentic.handoff import build_handoff_messages
from agentic.message_journal import TicketMessageJournal
from agentic.ticket_store import TicketStore
from agentic.v5_state import StateContractError, initial_state, merge_patch, validate_state
from logger import get_logger

log = get_logger()


class V5WorkflowState(TypedDict):
    ticket: dict[str, Any]
    supervisor: dict[str, Any]
    query: dict[str, Any]
    clarify: dict[str, Any]
    confirm: dict[str, Any]
    account: dict[str, Any]
    validate: dict[str, Any]
    escalation: dict[str, Any]
    budget: dict[str, Any]


@dataclass(frozen=True)
class WorkflowDependencies:
    model: BaseChatModel


_application_loop = None
_application_loop_lock = Lock()
_default_model: BaseChatModel | None = None


class _ApplicationEventLoop:
    """Keep synchronous callers on one loop so cached async clients remain valid."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = Event()
        self._lock = Lock()
        self._accepting = True
        self._pending: set[Future] = set()
        self._thread = Thread(target=self._run, name="uda-application-event-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Application event loop failed to start.")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    def run(self, coroutine: Coroutine[Any, Any, Any]):
        with self._lock:
            if not self._accepting:
                coroutine.close()
                raise RuntimeError("Application event loop is shutting down.")
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            self._pending.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    def close(self, cleanup_coroutine) -> None:
        if current_thread() is self._thread:
            raise RuntimeError("Application event loop cannot close itself.")
        with self._lock:
            self._accepting = False
            pending = list(self._pending)
        for future in pending:
            try:
                future.result()
            except Exception:
                pass
        try:
            asyncio.run_coroutine_threadsafe(cleanup_coroutine, self.loop).result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join()


def _get_application_loop() -> _ApplicationEventLoop:
    global _application_loop
    with _application_loop_lock:
        if _application_loop is None:
            _application_loop = _ApplicationEventLoop()
        return _application_loop


def _default_dependencies() -> WorkflowDependencies:
    global _default_model
    if _default_model is None:
        from langchain_openai import ChatOpenAI

        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing.")
        _default_model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
        )
    return WorkflowDependencies(model=_default_model)


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


async def escalation_node(state: V5WorkflowState, config: RunnableConfig, *, agent):
    """Summarize the full conversation and return a checkpointable handoff state."""
    validate_state(state)
    ticket = state["ticket"]
    if ticket["status"] not in {"processing", "waiting_reply"}:
        raise StateContractError("handoff requires a non-terminal ticket")

    response = await agent.ainvoke(
        {"conversation": _conversation_text(ticket["messages"])}, config=config,
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
    messages = list(updated["ticket"]["messages"])
    for handoff_message in build_handoff_messages(updated):
        existing = next(
            (index for index, item in enumerate(messages) if _message_id(item) == handoff_message.id),
            None,
        )
        if existing is None:
            messages.append(handoff_message)
        else:
            messages[existing] = handoff_message
    updated = merge_patch(updated, {"ticket": {"messages": messages}}, writer="boundary")
    return updated


def build_workflow(*, checkpointer: BaseCheckpointSaver, ticket_store: TicketStore,
                   dependencies: WorkflowDependencies | None = None, agent=None):
    """Compile the V5 handoff-only graph used while the rest of V5 is built."""
    if checkpointer is None or ticket_store is None:
        raise ValueError("V5 workflow requires a durable checkpointer and TicketStore")
    if agent is None:
        dependencies = dependencies or _default_dependencies()
        agent = build_escalation_agent(model=dependencies.model)
    graph = StateGraph(V5WorkflowState)
    async def run_escalation(state, config):
        return await escalation_node(state, config, agent=agent)
    graph.add_node("escalation", run_escalation)
    graph.add_edge(START, "escalation")
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer)


def _default_core_database() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "core" / "udahub.db"


@asynccontextmanager
async def persistent_workflow(database: str | os.PathLike, *, dependencies=None, agent=None, ticket_store=None):
    store = ticket_store or TicketStore(database)
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        yield build_workflow(
            checkpointer=checkpointer, ticket_store=store, dependencies=dependencies, agent=agent,
        )


async def aenter_ticket(external_user_id: str, message: HumanMessage, *, account_id: str | None,
                        channel: str | None, workflow=None, ticket_store: TicketStore,
                        ticket_id: str | None = None) -> V5WorkflowState:
    """Create/reuse the customer's ticket, checkpoint it, and run handoff."""
    if workflow is None:
        async with persistent_workflow(ticket_store.database, ticket_store=ticket_store) as graph:
            return await aenter_ticket(
                external_user_id, message, account_id=account_id, channel=channel,
                workflow=graph, ticket_store=ticket_store, ticket_id=ticket_id,
            )
    if not isinstance(message, HumanMessage) or not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("message must be a non-empty native HumanMessage")
    if not isinstance(message.id, str) or not message.id.strip():
        raise ValueError("customer messages must include a stable native message id")

    record = await asyncio.to_thread(
        ticket_store.acquire_or_create, external_user_id,
        account_id=account_id, channel=channel, ticket_id=ticket_id, initial_message=message,
    )
    thread_id = ticket_store.thread_id(record.ticket_id)
    journal = TicketMessageJournal(ticket_store, ticket_id=record.ticket_id)
    try:
        if record.created:
            state = initial_state(
                ticket_id=record.ticket_id, owner_id=record.external_user_id,
                owner_name=record.owner_name, message=message,
            )
            result = await _start_new_workflow(state, thread_id, workflow, ticket_store, journal)
        else:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            snapshot = await workflow.aget_state(config)
            initialized = await asyncio.to_thread(ticket_store.checkpoint_initialized, record.ticket_id)
            if not snapshot.values:
                if initialized:
                    from agentic.ticket_store import TicketPersistenceError
                    raise TicketPersistenceError(
                        f"ticket {record.ticket_id} is marked initialized but its checkpoint is missing"
                    )
                original = await asyncio.to_thread(ticket_store.initial_customer_message, record.ticket_id)
                state = initial_state(
                    ticket_id=record.ticket_id, owner_id=record.external_user_id,
                    owner_name=record.owner_name, message=original,
                )
                if message.id != original.id:
                    await asyncio.to_thread(ticket_store.append_customer_message, record.ticket_id, message)
                    state["ticket"]["messages"].append(message)
                    state["ticket"]["input"] = message.content
                result = await _start_new_workflow(state, thread_id, workflow, ticket_store, journal)
            else:
                ticket_state = snapshot.values.get("ticket", {}) or {}
                if ticket_state.get("ticket_id") != record.ticket_id:
                    from agentic.ticket_store import TicketHistoryConflict
                    raise TicketHistoryConflict("checkpoint ticket_id does not match the existing ticket")
                if ticket_state.get("status") == "handed_off":
                    # The graph checkpoint is durable before the core handoff transaction.
                    # If that transaction failed previously, finish it without calling the agent again.
                    result = snapshot.values
                elif ticket_state.get("status") == "completed":
                    from agentic.ticket_store import TicketHistoryConflict
                    raise TicketHistoryConflict("checkpoint is terminal while core ticket is active")
                else:
                    await asyncio.to_thread(ticket_store.mark_checkpoint_initialized, record.ticket_id)
                    await asyncio.to_thread(ticket_store.append_customer_message, record.ticket_id, message)
                    messages = list(ticket_state.get("messages") or [])
                    if not any(_message_id(item) == message.id for item in messages):
                        messages.append(message)
                    updated_ticket = {**ticket_state, "messages": messages, "input": message.content}
                    await workflow.aupdate_state(config, {"ticket": updated_ticket}, as_node=START)
                    result = await _invoke_with_journal(workflow, thread_id, journal)

        # LangGraph returns only after persisting the node output checkpoint. Commit the
        # core terminal state afterward; retries can repair a failed core transaction above.
        if result.get("ticket", {}).get("status") == "handed_off":
            await asyncio.to_thread(ticket_store.commit_handoff, record.ticket_id, state=result)
            log.info("ticket_handed_off", extra={"extra_data": {"ticket_id": record.ticket_id}})

        current = await asyncio.to_thread(ticket_store.status, record.ticket_id)
        if current != "handed_off" or result.get("ticket", {}).get("status") != current:
            raise RuntimeError("V5 handoff did not commit its terminal state")
        return result
    except Exception:
        try:
            await asyncio.to_thread(ticket_store.mark_processing_failed, record.ticket_id)
        except Exception:
            log.exception("ticket_processing_failure_status_persist_failed", extra={"extra_data": {"ticket_id": record.ticket_id}})
        raise


def _message_id(message: Any) -> str | None:
    return message.get("id") if isinstance(message, dict) else getattr(message, "id", None)


async def _start_new_workflow(state, thread_id: str, workflow, ticket_store: TicketStore,
                              journal: TicketMessageJournal):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    existing = await workflow.aget_state(config)
    if existing.values:
        raise ValueError("A checkpoint already exists for this ticket")
    await workflow.aupdate_state(config, state, as_node=START)
    await asyncio.to_thread(ticket_store.mark_checkpoint_initialized, thread_id)
    return await _invoke_with_journal(workflow, thread_id, journal)


async def _invoke_with_journal(workflow, thread_id: str, journal: TicketMessageJournal,
                               payload=None):
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [journal],
    }
    try:
        return await workflow.ainvoke(payload, config=config, durability="sync")
    finally:
        # This runs for success, graph errors, and cooperative cancellation.
        await journal.flush()


async def aresume_ticket(thread_id: str, *, workflow, reply=None, ticket_store=None):
    from langgraph.types import Command
    from agentic.ticket_store import TicketPersistenceError

    if ticket_store is not None:
        status = await asyncio.to_thread(ticket_store.status, thread_id)
        if status in {"completed", "handed_off"}:
            raise ValueError(f"ticket is terminal: {status}")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await workflow.aget_state(config)
    if not snapshot.values:
        raise TicketPersistenceError("No checkpoint exists for this ticket")
    ticket = snapshot.values.get("ticket", {}) or {}
    if ticket.get("ticket_id") != thread_id:
        raise ValueError("checkpoint ticket_id does not match its thread_id")
    if ticket.get("status") in {"completed", "handed_off"}:
        raise ValueError(f"ticket is terminal: {ticket['status']}")
    if not snapshot.next:
        raise ValueError("Ticket has no unfinished execution to resume")
    journal_store = ticket_store or TicketStore(_default_core_database())
    journal = TicketMessageJournal(journal_store, ticket_id=thread_id)
    payload = Command(resume=reply) if reply is not None else None
    return await _invoke_with_journal(workflow, thread_id, journal, payload=payload)


def run_ticket(ticket: dict[str, Any], thread_id: str, *, workflow=None, ticket_store=None):
    """Synchronous adapter for the V5 handoff-only workflow."""
    if thread_id != ticket.get("ticket_id"):
        raise ValueError("thread_id must equal ticket_id")
    store = ticket_store or TicketStore(_default_core_database())
    message = HumanMessage(
        content=ticket.get("content", ""),
        id=ticket.get("message_id") or f"customer:{thread_id}",
    )
    loop = _get_application_loop()
    if workflow is not None:
        return loop.run(aenter_ticket(
            ticket["owner_id"], message, account_id=ticket.get("account_id"),
            channel=ticket.get("channel"), workflow=workflow, ticket_store=store, ticket_id=thread_id,
        ))
    return loop.run(_run_default_entry(ticket, message, thread_id, store))


async def _run_default_entry(ticket, message, thread_id, store):
    async with persistent_workflow(store.database, ticket_store=store) as workflow:
        return await aenter_ticket(
            ticket["owner_id"], message, account_id=ticket.get("account_id"),
            channel=ticket.get("channel"), workflow=workflow, ticket_store=store, ticket_id=thread_id,
        )


async def _close_default_resources() -> None:
    model = _default_model
    if model is None:
        return
    errors = []
    for client, asynchronous in (
        (getattr(model, "root_async_client", None), True),
        (getattr(model, "root_client", None), False),
    ):
        if client is None:
            continue
        try:
            result = client.close()
            if asynchronous:
                await result
        except Exception as error:
            errors.append(error)
    if errors:
        raise errors[0]


def close_default_runtime() -> None:
    global _application_loop, _default_model
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("close_default_runtime() cannot run inside an event loop")
    with _application_loop_lock:
        loop = _application_loop
        if loop is None:
            return
        try:
            loop.close(_close_default_resources())
        finally:
            _default_model = None
            _application_loop = None


def _close_default_runtime_at_exit() -> None:
    try:
        close_default_runtime()
    except Exception:
        log.exception("default_runtime_shutdown_failed")


atexit.register(_close_default_runtime_at_exit)
