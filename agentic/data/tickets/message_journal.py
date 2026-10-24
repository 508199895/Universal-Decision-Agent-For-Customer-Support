"""Buffered LangChain callback journal for ticket events."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

from agentic.data.tickets.ticket_store import TicketEventStore


class TicketJournal(AsyncCallbackHandler):
    """Capture message and workflow events, then append them in batches."""

    def __init__(self, event_store: TicketEventStore, *, user_id: str, ticket_id: str,
                 flush_size: int = 20) -> None:
        if flush_size < 1:
            raise ValueError("flush_size must be positive")
        self.event_store = event_store
        self.user_id = user_id
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
                (message for batch in reversed(messages) for message in reversed(batch)
                 if isinstance(message, BaseMessage)
                 and message.type == "human" and not _is_summary_message(message)),
                None,
            )
            if human is None:
                return
            self._input_recorded = True
            self._append_locked(
                event_type="llm.human.input", category="message", content=human.model_dump(),
            )
        await self._flush_if_full()

    async def on_llm_end(self, response, **kwargs) -> None:
        token_count = 0
        async with self._lock:
            for batch in response.generations:
                for generation in batch:
                    message = getattr(generation, "message", None)
                    if isinstance(message, BaseMessage):
                        metadata = {}
                        if kwargs.get("name"):
                            metadata["caller"] = kwargs["name"]
                        usage = getattr(message, "usage_metadata", None)
                        if usage:
                            metadata["usage"] = usage
                            token_count += _reported_tokens(usage)
                        self._append_locked(
                            event_type="llm.ai.response", category="message",
                            content=message.model_dump(), metadata=metadata,
                        )
        if token_count:
            try:
                await asyncio.to_thread(self.event_store.add_tokens, self.ticket_id, token_count)
            except Exception:
                pass
        await self._flush_if_full()

    async def on_tool_end(self, output, **kwargs) -> None:
        messages = _tool_messages(output)
        if not messages:
            return
        async with self._lock:
            for message in messages:
                self._append_locked(
                    event_type="llm.tool.result", category="message",
                    content=message.model_dump(),
                    metadata={"tool_name": kwargs.get("name")} if kwargs.get("name") else {},
                )
        await self._flush_if_full()

    async def record_lifecycle(self, event_type: str, category: str,
                               content: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        async with self._lock:
            self._append_locked(
                event_type=event_type, category=category, content=content, metadata=metadata or {},
            )
        await self._flush_if_full()

    async def flush(self) -> None:
        """Attempt to persist buffered events; event logging does not block the graph."""
        async with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            try:
                await asyncio.to_thread(self.event_store.append_events, batch)
            except Exception:
                pass
                return
            del self._buffer[:len(batch)]

    def _append_locked(self, *, event_type: str, category: str, content: Any,
                       metadata: dict[str, Any] | None = None) -> None:
        self._buffer.append({
            "user_id": self.user_id,
            "ticket_id": self.ticket_id,
            "event_type": event_type,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    async def _flush_if_full(self) -> None:
        if len(self._buffer) >= self.flush_size:
            await self.flush()


def _is_summary_message(message: BaseMessage) -> bool:
    if getattr(message, "name", None) == "summary":
        return True
    additional = getattr(message, "additional_kwargs", None) or {}
    return isinstance(additional, dict) and bool(additional.get("is_summary"))


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


def _reported_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if type(total) is int and total > 0:
        return total
    prompt = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    return sum(value for value in (prompt, completion) if type(value) is int and value > 0)
