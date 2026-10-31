"""Capture LangChain callbacks as ticket events and completion metrics."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage

from agentic.data.tickets.TicketEventStore import TicketEventStore

logger = logging.getLogger(__name__)


class TicketJournal(AsyncCallbackHandler):
    """Capture callback events and completion metrics for one ticket run."""

    def __init__(
        self,
        event_store: TicketEventStore,
        ticket_id: str,
        *,
        track_token_usage: bool = True,
        flush_threshold: int = 20,
    ) -> None:
        if flush_threshold < 1:
            raise ValueError("flush_threshold must be positive")
        if not ticket_id:
            raise ValueError("ticket_id must be non-empty")
        self._store = event_store
        self.ticket_id = ticket_id
        self._track_tokens = track_token_usage
        self._flush_threshold = flush_threshold
        self._buffer: list[dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        self._llm_call_count = 0
        self._counted_llm_run_ids: set[str] = set()
        self._counted_message_llm_run_ids: set[str] = set()
        self._last_ai_msg: str | None = None
        self._first_human_msg: str | None = None
        self._msg_count = 0
        self._llm_start_times: dict[str, float] = {}
        self._llm_call_index = 0
        self._seen_llm_starts: set[str] = set()

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._put(
                event_type="run.start",
                category="trace",
                content={"chain": (serialized or {}).get("name", "unknown")},
                metadata={"caller": self._identify_caller(tags), **(metadata or {})},
            )
            await self._flush_if_full()

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._put(
                event_type="run.end",
                category="outputs",
                content=outputs,
                metadata={"status": "success"},
            )
            await self._flush_if_full()

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            cancelled = isinstance(error, asyncio.CancelledError)
            self._put(
                event_type="run.cancel" if cancelled else "run.error",
                category="trace" if cancelled else "error",
                content=str(error),
                metadata={"error_type": type(error).__name__},
            )
            await self._flush_if_full()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        self._llm_start_times[run_key] = time.monotonic()
        self._llm_call_index += 1
        self._seen_llm_starts.add(run_key)

        if not self._first_human_msg and messages:
            human = None
            for batch in reversed(messages):
                for message in reversed(batch):
                    if not isinstance(message, BaseMessage) or message.type != "human":
                        continue
                    additional = getattr(message, "additional_kwargs", None) or {}
                    if getattr(message, "name", None) == "summary":
                        continue
                    if isinstance(additional, dict) and additional.get("is_summary"):
                        continue
                    human = message
                    break
                if human is not None:
                    break
            if human is not None:
                caller = self._identify_caller(tags)
                self.set_first_human_message(self._message_text(human))
                self._put(
                    event_type="llm.human.input",
                    category="message",
                    content=human.model_dump(),
                    metadata={"caller": caller},
                )
                self._record_message_summary(human, caller=caller)
        await self._flush_if_full()

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[str(run_id)] = time.monotonic()

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        caller = self._identify_caller(tags)
        start = self._llm_start_times.pop(run_key, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start else None
        call_index = self._llm_call_index
        if run_key not in self._seen_llm_starts:
            self._llm_call_index += 1
            call_index = self._llm_call_index
            self._seen_llm_starts.add(run_key)

        messages: list[BaseMessage] = []
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                if isinstance(message, BaseMessage):
                    messages.append(message)

        for message in messages:
            usage = getattr(message, "usage_metadata", None)
            usage_data = dict(usage) if usage else {}
            self._put(
                event_type="llm.ai.response",
                category="message",
                content=message.model_dump(),
                metadata={
                    "caller": caller,
                    "usage": usage_data,
                    "latency_ms": latency_ms,
                    "llm_call_index": call_index,
                },
            )
            if run_key not in self._counted_message_llm_run_ids:
                self._record_message_summary(message, caller=caller)

            if self._track_tokens:
                input_tokens = usage_data.get("input_tokens", 0) or 0
                output_tokens = usage_data.get("output_tokens", 0) or 0
                total_tokens = usage_data.get("total_tokens", 0) or 0
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens
                if total_tokens > 0 and run_key not in self._counted_llm_run_ids:
                    self._counted_llm_run_ids.add(run_key)
                    self._total_input_tokens += input_tokens
                    self._total_output_tokens += output_tokens
                    self._total_tokens += total_tokens
                    self._llm_call_count += 1

        if messages:
            self._counted_message_llm_run_ids.add(run_key)
        await self._flush_if_full()

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times.pop(str(run_id), None)
        self._put(event_type="llm.error", category="trace", content=str(error))
        await self._flush_if_full()

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.debug("Tool start for node %s, tags=%s", run_id, tags)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(output, BaseMessage):
            messages = [output]
        else:
            update = getattr(output, "update", None)
            if isinstance(update, dict):
                output = update.get("messages")
            elif isinstance(output, dict):
                output = output.get("messages")
            if isinstance(output, BaseMessage):
                messages = [output]
            elif isinstance(output, (list, tuple)):
                messages = [message for message in output if isinstance(message, BaseMessage)]
            else:
                messages = []

        if not messages:
            logger.warning("on_tool_end %s: output has no message", run_id)
            return
        for message in messages:
            self._put(
                event_type="llm.tool.result",
                category="message",
                content=message.model_dump(),
            )
            self._record_message_summary(message)
        await self._flush_if_full()

    def _put(
        self,
        *,
        event_type: str,
        category: str,
        content: Any = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._buffer.append({
            "ticket_id": self.ticket_id,
            "event_type": event_type,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        })

    async def _flush_if_full(self) -> None:
        if len(self._buffer) >= self._flush_threshold:
            try:
                await self.flush()
            except Exception:
                logger.warning(
                    "Failed to flush ticket journal for %s; events remain buffered",
                    self.ticket_id,
                    exc_info=True,
                )

    def _identify_caller(self, tags: list[str] | None) -> str:
        for tag in tags or []:
            if isinstance(tag, str) and (
                tag.startswith("subagent:") or tag.startswith("middleware:") or tag == "lead_agent"
            ):
                return tag
        return "lead_agent"

    def _message_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(block.get("content"), str):
                        parts.append(block["content"])
            return "".join(parts)
        if isinstance(content, Mapping):
            for key in ("text", "content"):
                if isinstance(content.get(key), str):
                    return content[key]
        text = getattr(message, "text", None)
        return text if isinstance(text, str) else ""

    def _record_message_summary(self, message: BaseMessage, *, caller: str | None = None) -> None:
        self._msg_count += 1
        is_ai_message = isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
        if is_ai_message and (caller is None or caller == "lead_agent"):
            text = self._message_text(message).strip()
            if text:
                self._last_ai_msg = text[:2000]

    def set_first_human_message(self, content: str) -> None:
        self._first_human_msg = content[:2000] if content else None

    async def record_middleware(
        self,
        tag: str,
        *,
        name: str,
        hook: str,
        action: str,
        changes: dict[str, Any],
    ) -> None:
        self._put(
            event_type=f"middleware:{tag}",
            category="middleware",
            content={"name": name, "hook": hook, "action": action, "changes": changes},
        )
        await self._flush_if_full()

    async def flush(self) -> None:
        while self._buffer:
            batch = self._buffer[:self._flush_threshold]
            del self._buffer[:self._flush_threshold]
            try:
                await self._store.put_batch(batch)
            except Exception:
                self._buffer = batch + self._buffer
                raise

    def get_completion_data(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_tokens,
            "llm_call_count": self._llm_call_count,
            "message_count": self._msg_count,
            "last_ai_message": self._last_ai_msg,
            "first_human_message": self._first_human_msg,
        }
