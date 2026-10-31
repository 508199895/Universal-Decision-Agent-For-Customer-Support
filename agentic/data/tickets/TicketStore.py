"""Async persistence for ticket runtime records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic.data.tickets.TicketModel import TicketRow
from agentic.data.tickets.errors import (
    TicketHistoryConflict,
    TicketEventStoreError,
)

TicketStoreError = TicketEventStoreError


def _now() -> datetime:
    return datetime.now(UTC)


class TicketStore:
    """Persist ticket-run metadata through a caller-owned async session factory.

    The store never creates engines, sessions, business-ticket tables, or event
    stores. Its table is created by the application persistence lifecycle.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def _safe_json(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: TicketStore._safe_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TicketStore._safe_json(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return TicketStore._safe_json(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return TicketStore._safe_json(value.dict())
            except Exception:
                pass
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _row_to_dict(row: TicketRow) -> dict[str, Any]:
        result = row.to_dict()
        result["metadata"] = result.pop("metadata_json", {})
        result["kwargs"] = result.pop("kwargs_json", {})
        for key in ("created_at", "updated_at"):
            value = result.get(key)
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result

    async def put(
        self,
        ticket_id: str,
        *,
        user_id: str,
        model_name: str | None = None,
        status: str = "pending",
        multitask_strategy: str = "reject",
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        error: str | None = None,
        created_at: str | None = None,
        follow_up_to_ticket_id: str | None = None,
    ) -> None:
        if not ticket_id or not user_id:
            raise ValueError("ticket_id and user_id must be non-empty")
        created = datetime.fromisoformat(created_at) if created_at else _now()
        async with self._sf() as session:
            row = await session.get(TicketRow, ticket_id)
            if row is None:
                row = TicketRow(
                    ticket_id=ticket_id,
                    user_id=user_id,
                    status=status,
                    model_name=self._normalize_model_name(model_name),
                    multitask_strategy=multitask_strategy,
                    metadata_json=self._safe_json(metadata) or {},
                    kwargs_json=self._safe_json(kwargs) or {},
                    error=error,
                    follow_up_to_ticket_id=follow_up_to_ticket_id,
                    created_at=created,
                    updated_at=_now(),
                )
                session.add(row)
            else:
                if row.user_id != user_id:
                    raise TicketHistoryConflict(
                        f"ticket {ticket_id} already belongs to another user"
                    )
                row.status = status
                row.model_name = self._normalize_model_name(model_name)
                row.multitask_strategy = multitask_strategy
                row.metadata_json = self._safe_json(metadata) or {}
                row.kwargs_json = self._safe_json(kwargs) or {}
                row.error = error
                row.follow_up_to_ticket_id = follow_up_to_ticket_id
                row.updated_at = _now()
            await session.commit()

    @staticmethod
    def _normalize_model_name(model_name: str | None) -> str | None:
        if model_name is None:
            return None
        return str(model_name).strip()[:128]

    async def get(self, ticket_id: str) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(TicketRow, ticket_id)
            return self._row_to_dict(row) if row is not None else None

    async def list_by_user(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        statement = (
            select(TicketRow)
            .where(TicketRow.user_id == user_id)
            .order_by(TicketRow.created_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            rows = (await session.execute(statement)).scalars()
            return [self._row_to_dict(row) for row in rows]

    async def update_status(
        self, ticket_id: str, status: str, *, error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": _now()}
        if error is not None:
            values["error"] = error
        async with self._sf() as session:
            await session.execute(
                update(TicketRow).where(TicketRow.ticket_id == ticket_id).values(**values)
            )
            await session.commit()

    async def delete(self, ticket_id: str) -> None:
        async with self._sf() as session:
            row = await session.get(TicketRow, ticket_id)
            if row is not None:
                await session.delete(row)
                await session.commit()

    async def update_run_completion(
        self,
        ticket_id: str,
        *,
        status: str,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_tokens: int = 0,
        llm_call_count: int = 0,
        message_count: int = 0,
        last_ai_message: str | None = None,
        first_human_message: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "llm_call_count": llm_call_count,
            "message_count": message_count,
            "updated_at": _now(),
        }
        if last_ai_message is not None:
            values["last_ai_message"] = last_ai_message[:2000]
        if first_human_message is not None:
            values["first_human_message"] = first_human_message[:2000]
        if error is not None:
            values["error"] = error
        async with self._sf() as session:
            await session.execute(
                update(TicketRow).where(TicketRow.ticket_id == ticket_id).values(**values)
            )
            await session.commit()

    async def list_pending(self, *, before: str | datetime | None = None) -> list[dict[str, Any]]:
        before_at = _now() if before is None else (
            before if isinstance(before, datetime) else datetime.fromisoformat(before)
        )
        statement = (
            select(TicketRow)
            .where(TicketRow.status == "pending", TicketRow.created_at <= before_at)
            .order_by(TicketRow.created_at.asc())
        )
        async with self._sf() as session:
            rows = (await session.execute(statement)).scalars()
            return [self._row_to_dict(row) for row in rows]

    async def aggregate_tokens_by_user(self, user_id: str) -> dict[str, Any]:
        model = func.coalesce(TicketRow.model_name, "unknown")
        statement = (
            select(
                model.label("model"),
                func.count().label("runs"),
                func.coalesce(func.sum(TicketRow.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(TicketRow.total_input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(TicketRow.total_output_tokens), 0).label("total_output_tokens"),
            )
            .where(TicketRow.user_id == user_id, TicketRow.status.in_(("success", "error")))
            .group_by(model)
        )
        async with self._sf() as session:
            rows = (await session.execute(statement)).all()
        result: dict[str, Any] = {
            "total_tokens": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_runs": 0, "by_model": {},
        }
        for row in rows:
            result["by_model"][row.model] = {"tokens": row.total_tokens, "runs": row.runs}
            result["total_tokens"] += row.total_tokens
            result["total_input_tokens"] += row.total_input_tokens
            result["total_output_tokens"] += row.total_output_tokens
            result["total_runs"] += row.runs
        return result
