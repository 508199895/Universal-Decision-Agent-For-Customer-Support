"""Buffered, append-only message events attached to a support ticket."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

from agentic.ticket_store import TicketStore
from logger import get_logger

log = get_logger()


class TicketMessageJournal(AsyncCallbackHandler):
    """Capture selected model/tool messages and flush them in small batches."""

    def __init__(self, ticket_store: TicketStore, *, ticket_id: str,
                 flush_size: int = 20) -> None:
        if flush_size < 1:
            raise ValueError("flush_size must be positive")
        self.ticket_store = ticket_store
        self.ticket_id = ticket_id
        self.flush_size = flush_size
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._input_recorded = False

    async def on_chat_model_start(self, _serialized, messages, **kwargs) -> None:
        async with self._lock:
            if self._input_recorded:
                return
            human = next(
                (message for batch in messages for message in batch
                 if isinstance(message, BaseMessage) and message.type == "human"),
                None,
            )
            if human is None:
                return
            self._input_recorded = True
            self._append_locked(
                event_type="llm.human.input", message=human,
            )
        await self._flush_if_full()

    async def on_llm_end(self, response, **kwargs) -> None:
        async with self._lock:
            for batch in response.generations:
                for generation in batch:
                    message = getattr(generation, "message", None)
                    if isinstance(message, BaseMessage):
                        self._append_locked(
                            event_type="llm.ai.response", message=message,
                        )
        await self._flush_if_full()

    async def on_tool_end(self, output, **kwargs) -> None:
        messages = _tool_messages(output)
        if not messages:
            return
        async with self._lock:
            for message in messages:
                self._append_locked(
                    event_type="llm.tool.result", message=message,
                )
        await self._flush_if_full()

    async def flush(self) -> None:
        """Attempt to persist the buffered batch; event logging never blocks the graph."""
        async with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            try:
                await asyncio.to_thread(self.ticket_store.append_message_events, batch)
            except Exception:
                log.exception(
                    "run_event_flush_failed",
                    extra={"extra_data": {"ticket_id": self.ticket_id}},
                )
                return
            del self._buffer[:len(batch)]

    def _append_locked(self, *, event_type: str, message: BaseMessage) -> None:
        self._buffer.append({
            "ticket_id": self.ticket_id,
            "event_type": event_type,
            "content": _serialize_message(message),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    async def _flush_if_full(self) -> None:
        if len(self._buffer) >= self.flush_size:
            await self.flush()


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    result = {
        "type": message.type,
        "content": message.content,
        "id": message.id or str(uuid.uuid4()),
    }
    if message.additional_kwargs:
        result["additional_kwargs"] = message.additional_kwargs
    if message.response_metadata:
        result["response_metadata"] = message.response_metadata
    if getattr(message, "tool_call_id", None):
        result["tool_call_id"] = message.tool_call_id
    if getattr(message, "name", None):
        result["name"] = message.name
    return result


def _tool_messages(output: Any) -> list[BaseMessage]:
    if isinstance(output, BaseMessage):
        return [output]
    update = getattr(output, "update", None)
    if isinstance(update, dict):
        output = update.get("messages")
    elif isinstance(output, dict):
        output = output.get("messages")
    if isinstance(output, BaseMessage):
        return [output]
    if isinstance(output, (list, tuple)):
        return [message for message in output if isinstance(message, BaseMessage)]
    return []


