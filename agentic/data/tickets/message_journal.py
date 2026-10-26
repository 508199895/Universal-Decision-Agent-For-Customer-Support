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
                 flush_size: int = 20, token_limit: int = 20_000_000) -> None:
        if flush_size < 1:
            raise ValueError("flush_size must be positive")
        self.event_store = event_store
        self.user_id = user_id
        self.ticket_id = ticket_id
        self.flush_size = flush_size
        if type(token_limit) is not int or token_limit < 1:
            raise ValueError("token_limit must be a positive integer")
        self.token_limit = token_limit
        record = event_store.get_ticket(ticket_id)
        self.input_tokens = record.input_tokens
        self.output_tokens = record.output_tokens
        self.total_tokens = record.tokens
        self.llm_call_count = record.llm_call_count
        self._counted_llm_run_ids: set[str] = set()
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
        usage_for_run = None
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
                            if usage_for_run is None and _usage_values(usage)[2] > 0:
                                usage_for_run = usage
                        self._append_locked(
                            event_type="llm.ai.response", category="message",
                            content=message.model_dump(), metadata=metadata,
                        )
        if usage_for_run is not None:
            run_id = str(kwargs.get("run_id"))
            if run_id not in self._counted_llm_run_ids:
                self._counted_llm_run_ids.add(run_id)
                input_tokens, output_tokens, total_tokens = _usage_values(usage_for_run)
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.total_tokens += total_tokens
                self.llm_call_count += 1
        await self._flush_if_full()

    async def persist_usage_snapshot(self) -> None:
        """Persist the accumulated counters once after the graph run finishes."""
        await asyncio.to_thread(
            self.event_store.save_token_usage,
            self.ticket_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            llm_call_count=self.llm_call_count,
        )

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
    return _usage_values(usage)[2]


def _usage_values(usage: Any) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    input_tokens = input_tokens if type(input_tokens) is int and input_tokens > 0 else 0
    output_tokens = output_tokens if type(output_tokens) is int and output_tokens > 0 else 0
    total_tokens = usage.get("total_tokens", 0)
    if type(total_tokens) is not int or total_tokens < 0:
        total_tokens = 0
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens
