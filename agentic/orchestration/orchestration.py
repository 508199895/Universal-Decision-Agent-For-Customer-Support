"""Run ticket requests against durable ticket state and a compiled workflow."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import START
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import RunnableConfig

from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.data.tickets.ticket_journal import TicketJournal
from agentic.data.tickets.TicketEventStore import TicketEventStore
from agentic.data.tickets.TicketStore import (
    TicketStore,
)
from agentic.data.tickets.user_context import reset_current_user_id, set_current_user_id
from agentic.observability.langsmith_tracing import build_trace_config
from agentic.orchestration.ticket.manager import TicketManager, TicketRecord
from agentic.orchestration.ticket.request import TicketStartBody
from agentic.orchestration.workflow.state import WorkflowState, initial_state


logger = logging.getLogger(__name__)


async def astream_workflow(workflow, payload, *, config, context):
    """Stream node updates from one workflow invocation."""
    async for update in workflow.astream(
        payload,
        config=config,
        context=context,
        durability="async",
        stream_mode="updates",
    ):
        yield update


@dataclass(frozen=True)
class TicketWorkflowContext:
    """Long-lived resources and authenticated identity for one ticket task."""

    user_id: str
    external_user_id: str
    user_name: str | None
    workflow: Any
    ticket_store: TicketStore
    ticket_event_store: TicketEventStore


async def start_ticket(body: TicketStartBody, user_id: str, request: Any) -> TicketRecord:
    """Parse ticket input, admit the run, and dispatch ``run_workflow`` in a task."""
    if not isinstance(body, TicketStartBody):
        body = TicketStartBody.model_validate(body)
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    context = _ticket_workflow_context(user_id, request)
    message = _message_from_ticket_body(body)
    metadata_fields = body.metadata or {}
    config_overrides = body.config or {}
    body_context = body.context or {}
    model_name = body_context.get("model_name")
    metadata = {
        **metadata_fields,
        "owner_id": context.external_user_id,
        "owner_name": context.user_name,
    }
    manager = request.app.state.ticket_manager
    record = await manager.create_or_reject(
        user_id=user_id,
        metadata=metadata,
        kwargs={
            "input": body.input,
            "config": config_overrides,
            "context": body_context,
        },
        multitask_strategy=body.multitask_strategy,
        model_name=model_name,
    )
    task = asyncio.create_task(
        run_workflow(
            manager, record, ctx=context, message=message,
            config_overrides=config_overrides,
        ),
        name=f"ticket:{record.ticket_id}",
    )
    record.task = task
    task.add_done_callback(_observe_workflow_task)
    return record


def _ticket_workflow_context(user_id: str, request: Any) -> TicketWorkflowContext:
    """Read runtime resources and authenticated identity from a request-like object."""
    try:
        app_state = request.app.state
        request_state = request.state
    except AttributeError as error:
        raise ValueError("request must expose app.state and state") from error
    user = getattr(request_state, "user", None)
    external_user_id = (
        user.get("external_user_id") or user.get("id")
        if isinstance(user, dict)
        else getattr(user, "external_user_id", None) or getattr(user, "id", None)
    )
    if not isinstance(external_user_id, str) or not external_user_id.strip():
        raise ValueError("request.state.user must provide external_user_id or id")
    user_name = (
        user.get("user_name") or user.get("name")
        if isinstance(user, dict)
        else getattr(user, "user_name", None) or getattr(user, "name", None)
    )
    return TicketWorkflowContext(
        user_id=user_id,
        external_user_id=external_user_id,
        user_name=user_name if isinstance(user_name, str) else None,
        workflow=app_state.workflow,
        ticket_store=app_state.ticket_store,
        ticket_event_store=app_state.ticket_event_store,
    )


def _message_from_ticket_body(body: TicketStartBody) -> HumanMessage:
    messages = (body.input or {}).get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("input.messages must contain the customer's latest message")
    raw_message = messages[-1]
    if isinstance(raw_message, HumanMessage):
        content = raw_message.content
        message_id = raw_message.id
    elif isinstance(raw_message, dict):
        role = raw_message.get("role") or raw_message.get("type")
        if role not in {"user", "human", "HumanMessage"}:
            raise ValueError("input.messages must end with a customer message")
        content = raw_message.get("content")
        message_id = raw_message.get("id")
    else:
        raise ValueError("input.messages must contain a customer message object")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("customer message content must be a non-empty string")
    if not isinstance(message_id, str) or not message_id.strip():
        message_id = f"customer:{uuid.uuid4()}"
    return HumanMessage(content=content, id=message_id)


def _observe_workflow_task(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Background ticket workflow failed",
            exc_info=(type(error), error, error.__traceback__),
        )


async def run_workflow(
    manager: TicketManager,
    record: TicketRecord,
    *,
    ctx: TicketWorkflowContext,
    message: HumanMessage,
    config_overrides: dict[str, Any] | None = None,
) -> WorkflowState:
    """Execute one admitted run, restoring active workflow state by user thread."""
    run_record = record
    user_id = ctx.user_id
    external_user_id = ctx.external_user_id
    user_name = ctx.user_name
    workflow = ctx.workflow
    ticket_event_store = ctx.ticket_event_store
    thread_id = user_id
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    checkpointer = getattr(workflow, "checkpointer", None)
    pre_run_checkpoint = None
    checkpoint_capture_failed = False
    journal: TicketJournal | None = None
    workflow_completed = False
    supervisor_reply: str | None = None
    user_context_token = set_current_user_id(user_id)
    try:
        ticket_id = run_record.ticket_id
        try:
            pre_run_checkpoint = (
                await checkpointer.aget_tuple(config) if checkpointer is not None else None
            )
            checkpoint_capture_failed = False
        except Exception:
            pre_run_checkpoint = None
            checkpoint_capture_failed = True
            logger.warning("Could not capture pre-run checkpoint for ticket rollback", exc_info=True)
        snapshot = await workflow.aget_state(config)
        latest_state = dict(snapshot.values or {})
        latest_ticket = dict(latest_state.get("ticket") or {})
        # Older checkpoints stored these values in State. Drop them while
        # restoring so all subsequent checkpoints follow the current contract.
        latest_ticket.pop("ticket_id", None)
        latest_ticket.pop("status", None)
        journal = TicketJournal(
            event_store=ticket_event_store,
            ticket_id=ticket_id,
        )
        runtime_context = TicketRuntimeContext(
            journal=journal,
            external_user_id=external_user_id,
            ticket_id=ticket_id,
        )
        now = datetime.now(timezone.utc).isoformat()
        await ticket_event_store.put_batch([{
            "ticket_id": ticket_id,
            "event_type": "ticket.created",
            "category": "trace",
            "content": {"ticket_id": ticket_id},
            "metadata": {"owner_id": external_user_id, "owner_name": user_name},
            "created_at": now,
        }, {
            "ticket_id": ticket_id,
            "event_type": "ticket.message.received",
            "category": "message",
            "content": message.model_dump(),
            "metadata": {"source": "ticket_entry", "owner_id": external_user_id,
                         "owner_name": user_name},
            "created_at": now,
        }])

        if latest_state:
            messages = list(latest_state.get("messages") or [])
            if not any(
                (item.get("id") if isinstance(item, dict) else getattr(item, "id", None)) == message.id
                for item in messages
            ):
                messages.append(message)
            latest_ticket.update(
                input=message.content,
                owner_id=external_user_id,
                owner_name=user_name,
            )
            state_update = {
                **latest_state,
                "ticket": latest_ticket,
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages],
            }
        else:
            state_update = initial_state(
                owner_id=external_user_id, owner_name=user_name, message=message,
            )
            state_update["messages"] = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES), *state_update["messages"],
            ]

        try:
            await workflow.aupdate_state(config, state_update, as_node=START)
            await manager.set_status(run_record.ticket_id, "running")
            runnable_config: RunnableConfig = dict(config_overrides or {})
            configurable = runnable_config.get("configurable") or {}
            if not isinstance(configurable, dict):
                raise ValueError("workflow config 'configurable' must be an object")
            runnable_config["configurable"] = {
                **configurable,
                "thread_id": thread_id,
            }
            callbacks = runnable_config.get("callbacks") or []
            if not isinstance(callbacks, list):
                raise ValueError("workflow config 'callbacks' must be a list")
            runnable_config["callbacks"] = [journal, *callbacks]
            runnable_config.setdefault(
                "recursion_limit", int(os.getenv("WORKFLOW_RECURSION_LIMIT", "12")),
            )
            trace_config = build_trace_config(ticket_id)
            if trace_config:
                runnable_config["run_name"] = trace_config["run_name"]
                runnable_config["metadata"] = trace_config["metadata"]
                runnable_config["tags"] = trace_config["tags"]
                runnable_config["callbacks"].extend(trace_config["callbacks"])

            async for update in astream_workflow(
                workflow, None, config=runnable_config, context=runtime_context,
            ):
                if not isinstance(update, dict):
                    continue
                supervisor_state = update.get("supervisor")
                supervisor_group = (
                    supervisor_state.get("supervisor")
                    if isinstance(supervisor_state, dict) else None
                )
                if isinstance(supervisor_group, dict):
                    reply = supervisor_group.get("message")
                    if isinstance(reply, str) and reply.strip():
                        supervisor_reply = reply

            final_snapshot = await workflow.aget_state(runnable_config)
            result = dict(final_snapshot.values or {})
            if not isinstance(supervisor_reply, str) or not supervisor_reply.strip():
                raise RuntimeError("workflow stream did not produce a supervisor reply")
            result["supervisor"] = {
                **dict(result.get("supervisor") or {}),
                "message": supervisor_reply,
            }

            workflow_completed = True
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await manager.set_status(run_record.ticket_id, "error", error=str(error))
            raise

    except asyncio.CancelledError as cancel_error:
        rollback_requested = bool(
            cancel_error.args and cancel_error.args[0] == "rollback"
        )
        if rollback_requested:
            await manager.set_status(
                run_record.ticket_id, "error", error="Rolled back by user",
            )
            try:
                await _rollback_to_pre_run_checkpoint(
                    checkpointer, config, pre_run_checkpoint, checkpoint_capture_failed,
                    cancel_error,
                )
            except Exception:
                logger.warning(
                    "Ticket run %s rollback failed", run_record.ticket_id, exc_info=True,
                )
        else:
            await manager.set_status(run_record.ticket_id, "interrupted")
        raise
    except Exception as error:
        if run_record.status.value in {"pending", "running"}:
            await manager.set_status(run_record.ticket_id, "error", error=str(error))
        raise
    finally:
        try:
            if journal is not None:
                try:
                    await journal.flush()
                except Exception:
                    logger.warning(
                        "Failed to flush journal for ticket %s", ticket_id, exc_info=True,
                    )
                try:
                    completion_data = journal.get_completion_data()
                    if workflow_completed and supervisor_reply:
                        completion_data["last_ai_message"] = supervisor_reply
                    await manager.update_run_completion(
                        ticket_id,
                        status=run_record.status.value,
                        **completion_data,
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist ticket completion for %s",
                        ticket_id,
                        exc_info=True,
                    )
            if workflow_completed and run_record.status.value in {"pending", "running"}:
                await manager.set_status(run_record.ticket_id, "success")
        finally:
            reset_current_user_id(user_context_token)


async def _rollback_to_pre_run_checkpoint(
    checkpointer: Any,
    config: RunnableConfig,
    snapshot: Any,
    capture_failed: bool,
    cancellation: asyncio.CancelledError,
) -> None:
    """Restore the pre-run LangGraph checkpoint when cancellation requests rollback."""
    if not cancellation.args or cancellation.args[0] != "rollback":
        return
    if checkpointer is None or capture_failed:
        logger.warning("Ticket rollback skipped because no reliable checkpoint snapshot is available")
        return
    thread_id = config["configurable"]["thread_id"]
    if snapshot is None:
        await checkpointer.adelete_thread(thread_id)
        return

    checkpoint = dict(snapshot.checkpoint)
    marker = empty_checkpoint()
    checkpoint["id"] = marker["id"]
    checkpoint["ts"] = marker["ts"]
    configurable = snapshot.config.get("configurable", {})
    checkpoint_ns = configurable.get("checkpoint_ns", "")
    restore_config = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
    }
    restored_config = await checkpointer.aput(
        restore_config,
        checkpoint,
        snapshot.metadata or {},
        dict(checkpoint.get("channel_versions") or {}),
    )
    pending_writes = snapshot.pending_writes or []
    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for task_id, channel, value in pending_writes:
        writes_by_task.setdefault(str(task_id), []).append((channel, value))
    for task_id, writes in writes_by_task.items():
        await checkpointer.aput_writes(restored_config, writes, task_id=task_id)




