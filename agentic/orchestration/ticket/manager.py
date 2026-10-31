"""In-memory manager for active ticket runs with durable TicketStore backing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from agentic.data.tickets.TicketStore import TicketHistoryConflict, TicketStore

logger = logging.getLogger(__name__)


class TicketStatus(StrEnum):
    """Ticket runtime states corresponding to RunManager's RunStatus."""

    pending = "pending"
    running = "running"
    success = "success"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


class MultitaskStrategy(StrEnum):
    """Policy for admitting a run while another run is active."""

    reject = "reject"
    interrupt = "interrupt"
    rollback = "rollback"


class ConflictError(Exception):
    """Raised when reject strategy finds an active ticket run for the user."""


class UnsupportedStrategyError(Exception):
    """Raised when a multitask strategy is not supported by TicketManager."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TicketRecord:
    """Mutable runtime record; each admitted execution gets a new ticket ID."""

    ticket_id: str
    user_id: str
    status: TicketStatus = TicketStatus.pending
    multitask_strategy: MultitaskStrategy = MultitaskStrategy.reject
    metadata: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    task: asyncio.Task | None = None
    error: str | None = None
    model_name: str | None = None


class TicketManager:
    """Manage active ticket runs and persist their serializable run metadata."""

    _ACTIVE_STATUSES = frozenset({TicketStatus.pending, TicketStatus.running})
    _TERMINAL_STATUSES = frozenset({
        TicketStatus.success, TicketStatus.error,
        TicketStatus.timeout, TicketStatus.interrupted,
    })
    _SUPPORTED_STRATEGIES = frozenset(MultitaskStrategy)

    def __init__(self, ticket_store: TicketStore) -> None:
        self._store = ticket_store
        self._tickets: dict[str, TicketRecord] = {}
        self._lock = asyncio.Lock()

    async def _persist_to_store(self, record: TicketRecord) -> None:
        """Best-effort persist serializable run metadata, matching RunManager."""
        try:
            await self._store.put(
                record.ticket_id,
                user_id=record.user_id,
                model_name=record.model_name,
                status=record.status.value,
                multitask_strategy=record.multitask_strategy.value,
                metadata=record.metadata or {},
                kwargs=record.kwargs or {},
                error=record.error,
                created_at=record.created_at,
            )
        except Exception:
            logger.warning("Failed to persist ticket run %s", record.ticket_id, exc_info=True)

    async def update_run_completion(self, ticket_id: str, **kwargs: Any) -> None:
        """Persist journal events, usage, and message summary to TicketStore."""
        if "input_tokens" in kwargs:
            kwargs["total_input_tokens"] = kwargs.pop("input_tokens")
        if "output_tokens" in kwargs:
            kwargs["total_output_tokens"] = kwargs.pop("output_tokens")
        record = self.get(ticket_id)
        if isinstance(kwargs.get("status"), TicketStatus):
            kwargs["status"] = kwargs["status"].value
        kwargs.setdefault(
            "status", record.status.value if record else TicketStatus.running.value,
        )
        try:
            await self._store.update_run_completion(ticket_id, **kwargs)
        except Exception:
            logger.warning("Failed to persist ticket run completion for %s", ticket_id, exc_info=True)
            raise

    def get(self, ticket_id: str) -> TicketRecord | None:
        """Return a process-local ticket runtime record, if it is still active."""
        return self._tickets.get(ticket_id)

    async def list_by_user(self, user_id: str) -> list[TicketRecord]:
        """Return process-local runtime records for a user, newest first."""
        async with self._lock:
            return [ticket for ticket in self._tickets.values() if ticket.user_id == user_id]

    async def set_status(
        self, ticket_id: str, status: TicketStatus | str, *, error: str | None = None,
    ) -> None:
        """Transition a ticket run and persist its runtime status."""
        try:
            status = TicketStatus(status)
        except ValueError:
            raise ValueError(f"unsupported ticket run status: {status}")
        async with self._lock:
            record = self._tickets.get(ticket_id)
            if record is None:
                logger.warning("set_status called for unknown ticket run %s", ticket_id)
                return
            record.status = status
            record.updated_at = _utc_now()
            if error is not None:
                record.error = error
            if status in self._TERMINAL_STATUSES:
                record.task = None
        try:
            await self._store.update_status(ticket_id, status.value, error=error)
        except Exception:
            logger.warning("Failed to persist status for ticket run %s", ticket_id, exc_info=True)
        if status in self._TERMINAL_STATUSES:
            asyncio.create_task(self.cleanup(ticket_id))

    async def cancel(self, ticket_id: str, *, action: str = "interrupt") -> bool:
        """Request interruption or rollback; the worker persists the final status."""
        if action not in {"interrupt", "rollback"}:
            raise ValueError("action must be 'interrupt' or 'rollback'")
        async with self._lock:
            record = self._tickets.get(ticket_id)
            if record is None or record.status not in self._ACTIVE_STATUSES:
                return False
            task = record.task
            if task is not None and task.done():
                return False
            if task is not None:
                task.cancel(action)
            record.status = TicketStatus.interrupted
            record.updated_at = _utc_now()
        return True

    async def create_or_reject(
        self,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        multitask_strategy: MultitaskStrategy | str = MultitaskStrategy.reject,
        model_name: str | None = None,
    ) -> TicketRecord:
        """Atomically admit a user run using RunManager's multitask strategies."""
        if not user_id:
            raise ValueError("user_id must be a non-empty string")
        if multitask_strategy not in self._SUPPORTED_STRATEGIES:
            supported = ", ".join(sorted(strategy.value for strategy in self._SUPPORTED_STRATEGIES))
            raise UnsupportedStrategyError(
                f"unsupported multitask_strategy {multitask_strategy!r}; supported: {supported}"
            )
        multitask_strategy = MultitaskStrategy(multitask_strategy)

        ticket_id = str(uuid.uuid4())
        now = _utc_now()
        async with self._lock:
            inflight = [
                item for item in self._tickets.values()
                if item.user_id == user_id and item.status in self._ACTIVE_STATUSES
            ]
            if multitask_strategy == MultitaskStrategy.reject and inflight:
                raise ConflictError(f"user {user_id} already has an active ticket run")
            if multitask_strategy in {MultitaskStrategy.interrupt, MultitaskStrategy.rollback}:
                for item in inflight:
                    task = item.task
                    if task is not None and not task.done():
                        task.cancel(multitask_strategy.value)
                    item.status = TicketStatus.interrupted
                    item.updated_at = now
                    item.task = None

            record = TicketRecord(
                ticket_id=ticket_id,
                user_id=user_id,
                status=TicketStatus.pending,
                multitask_strategy=multitask_strategy,
                metadata=metadata or {},
                kwargs=kwargs or {},
                task=asyncio.current_task(),
                model_name=model_name,
            )
            self._tickets[ticket_id] = record

        await self._persist_to_store(record)
        return record

    async def has_inflight(self, user_id: str) -> bool:
        """Return whether a user currently has a pending or running ticket run."""
        async with self._lock:
            return any(
                ticket.user_id == user_id and ticket.status in self._ACTIVE_STATUSES
                for ticket in self._tickets.values()
            )

    async def cleanup(
        self,
        ticket_id: str,
        *,
        delay: float = 300,
    ) -> None:
        """Remove a local runtime record after an optional retention delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        async with self._lock:
            self._tickets.pop(ticket_id, None)

