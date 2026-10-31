"""SQLAlchemy-backed append-only ticket event store."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic.data.tickets.TicketEventModel import TicketEventRow
from agentic.data.tickets.errors import (
    TicketEventStoreError,
    TicketPersistenceError,
)
from agentic.data.tickets.user_context import require_current_user_id

logger = logging.getLogger(__name__)

CATEGORIES = {"message", "trace", "outputs", "error", "middleware"}


class TicketEventStore:
    """Persist ticket events and scope every operation to the current user."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_trace_content: int = 10240,
    ) -> None:
        if max_trace_content < 1:
            raise ValueError("max_trace_content must be positive")
        self._sf = session_factory
        self._max_trace_content = max_trace_content

    @staticmethod
    def _row_to_dict(row: TicketEventRow) -> dict[str, Any]:
        result = row.to_dict()
        result["metadata"] = result.pop("event_metadata", {})
        created_at = result.get("created_at")
        if isinstance(created_at, datetime):
            result["created_at"] = created_at.isoformat()
        result.pop("event_id", None)

        content = result.get("content", "")
        metadata = result.get("metadata", {})
        if isinstance(content, str) and (
            metadata.get("content_is_json") or metadata.get("content_is_dict")
        ):
            try:
                result["content"] = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                logger.debug(
                    "Failed to deserialize ticket event content for seq=%s",
                    result.get("seq"),
                )
        return result

    def _truncate_trace(
        self, category: str, content: Any, metadata: dict | None,
    ) -> tuple[Any, dict]:
        metadata = metadata or {}
        if category == "trace":
            text = content if isinstance(content, str) else json.dumps(
                content, default=str, ensure_ascii=False,
            )
            encoded = text.encode("utf-8")
            if len(encoded) > self._max_trace_content:
                content = encoded[:self._max_trace_content].decode("utf-8", errors="ignore")
                metadata = {
                    **metadata,
                    "content_truncated": True,
                    "original_byte_length": len(encoded),
                }
        return content, metadata

    @staticmethod
    def _content_to_db(content: Any, metadata: dict | None) -> tuple[str, dict]:
        metadata = metadata or {}
        if isinstance(content, str):
            return content, metadata

        db_content = json.dumps(content, default=str, ensure_ascii=False)
        metadata = {**metadata, "content_is_json": True}
        if isinstance(content, dict):
            metadata["content_is_dict"] = True
        return db_content, metadata

    @staticmethod
    async def _max_seq_for_user(
        session: AsyncSession, user_id: str,
    ) -> int | None:
        statement = (
            select(func.max(TicketEventRow.seq))
            .where(TicketEventRow.user_id == user_id)
            .with_for_update()
        )
        return await session.scalar(statement)

    async def put(
        self,
        *,
        ticket_id: str,
        event_type: str,
        category: str,
        content: Any = "",
        metadata: dict | None = None,
        created_at: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Write a single event and return it with its assigned sequence."""
        user_id = require_current_user_id(method_name="TicketEventStore")
        if not ticket_id or not event_type:
            raise ValueError("ticket_id and event_type must be non-empty")
        if category not in CATEGORIES:
            raise ValueError(f"unknown event category: {category!r}")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("event metadata must be an object")

        content, metadata = self._truncate_trace(category, content, metadata)
        db_content, metadata = self._content_to_db(content, metadata)
        created = (
            datetime.now(UTC)
            if created_at is None
            else created_at
            if isinstance(created_at, datetime)
            else datetime.fromisoformat(created_at)
        )
        try:
            async with self._sf() as session:
                async with session.begin():
                    seq = (await self._max_seq_for_user(session, user_id) or 0) + 1
                    row = TicketEventRow(
                        user_id=user_id,
                        ticket_id=ticket_id,
                        seq=seq,
                        event_type=event_type,
                        category=category,
                        content=db_content,
                        event_metadata=metadata,
                        created_at=created,
                    )
                    session.add(row)
                    await session.flush()
                return self._row_to_dict(row)
        except TicketEventStoreError:
            raise
        except SQLAlchemyError as error:
            raise TicketPersistenceError(f"ticket event persistence failed: {error}") from error

    async def put_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write a batch of events for the current user in one transaction."""
        if not events:
            return []
        user_id = require_current_user_id(method_name="TicketEventStore")
        prepared: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("each event must be an object")
            if "user_id" in event:
                raise ValueError("user_id is resolved from the current context")
            ticket_id = event.get("ticket_id")
            event_type = event.get("event_type")
            category = event.get("category", "trace")
            if not isinstance(ticket_id, str) or not ticket_id:
                raise ValueError("ticket_id must be a non-empty string")
            if not isinstance(event_type, str) or not event_type:
                raise ValueError("event_type must be a non-empty string")
            if category not in CATEGORIES:
                raise ValueError(f"unknown event category: {category!r}")
            metadata = event.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("event metadata must be an object")
            content, metadata = self._truncate_trace(
                category, event.get("content", ""), metadata,
            )
            db_content, metadata = self._content_to_db(content, metadata)
            prepared.append({
                "ticket_id": ticket_id,
                "event_type": event_type,
                "category": category,
                "content": db_content,
                "metadata": metadata,
                "created_at": (
                    datetime.now(UTC)
                    if event.get("created_at") is None
                    else event["created_at"]
                    if isinstance(event["created_at"], datetime)
                    else datetime.fromisoformat(event["created_at"])
                ),
            })

        try:
            async with self._sf() as session:
                async with session.begin():
                    seq = (await self._max_seq_for_user(session, user_id) or 0)
                    rows = []
                    for event in prepared:
                        seq += 1
                        row = TicketEventRow(
                            user_id=user_id,
                            ticket_id=event["ticket_id"],
                            seq=seq,
                            event_type=event["event_type"],
                            category=event["category"],
                            content=event["content"],
                            event_metadata=event["metadata"],
                            created_at=event["created_at"],
                        )
                        session.add(row)
                        rows.append(row)
                    await session.flush()
                return [self._row_to_dict(row) for row in rows]
        except TicketEventStoreError:
            raise
        except SQLAlchemyError as error:
            raise TicketPersistenceError(f"ticket event batch persistence failed: {error}") from error

    async def list_messages(
        self,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        user_id = require_current_user_id(method_name="TicketEventStore")
        statement = select(TicketEventRow).where(
            TicketEventRow.user_id == user_id,
            TicketEventRow.category == "message",
        )
        if before_seq is not None:
            statement = statement.where(TicketEventRow.seq < before_seq)
        if after_seq is not None:
            statement = statement.where(TicketEventRow.seq > after_seq)
        if limit < 1:
            raise ValueError("limit must be positive")
        if after_seq is not None:
            statement = statement.order_by(TicketEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                rows = (await session.execute(statement)).scalars()
                return [self._row_to_dict(row) for row in rows]

        statement = statement.order_by(TicketEventRow.seq.desc()).limit(limit)
        async with self._sf() as session:
            rows = list((await session.execute(statement)).scalars())
            return [self._row_to_dict(row) for row in reversed(rows)]

    async def list_events(
        self,
        ticket_id: str,
        *,
        event_types: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        user_id = require_current_user_id(method_name="TicketEventStore")
        statement = select(TicketEventRow).where(
            TicketEventRow.user_id == user_id,
            TicketEventRow.ticket_id == ticket_id,
        )
        if event_types:
            statement = statement.where(TicketEventRow.event_type.in_(event_types))
        statement = statement.order_by(TicketEventRow.seq.asc()).limit(limit)
        async with self._sf() as session:
            rows = (await session.execute(statement)).scalars()
            return [self._row_to_dict(row) for row in rows]

    async def list_messages_by_ticket(
        self,
        ticket_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        user_id = require_current_user_id(method_name="TicketEventStore")
        statement = select(TicketEventRow).where(
            TicketEventRow.user_id == user_id,
            TicketEventRow.ticket_id == ticket_id,
            TicketEventRow.category == "message",
        )
        if before_seq is not None:
            statement = statement.where(TicketEventRow.seq < before_seq)
        if after_seq is not None:
            statement = statement.where(TicketEventRow.seq > after_seq)
        if limit < 1:
            raise ValueError("limit must be positive")
        if after_seq is not None:
            statement = statement.order_by(TicketEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                rows = (await session.execute(statement)).scalars()
                return [self._row_to_dict(row) for row in rows]

        statement = statement.order_by(TicketEventRow.seq.desc()).limit(limit)
        async with self._sf() as session:
            rows = list((await session.execute(statement)).scalars())
            return [self._row_to_dict(row) for row in reversed(rows)]

    async def count_messages(self) -> int:
        user_id = require_current_user_id(method_name="TicketEventStore")
        statement = select(func.count()).select_from(TicketEventRow).where(
            TicketEventRow.user_id == user_id,
            TicketEventRow.category == "message",
        )
        async with self._sf() as session:
            return await session.scalar(statement) or 0

    async def delete_by_user(self) -> int:
        user_id = require_current_user_id(method_name="TicketEventStore")
        async with self._sf() as session:
            count = await session.scalar(
                select(func.count()).select_from(TicketEventRow).where(
                    TicketEventRow.user_id == user_id,
                )
            ) or 0
            if count:
                await session.execute(delete(TicketEventRow).where(
                    TicketEventRow.user_id == user_id,
                ))
                await session.commit()
            return count

    async def delete_by_ticket(self, ticket_id: str) -> int:
        """Delete the current user's events for one ticket."""
        user_id = require_current_user_id(method_name="TicketEventStore")
        conditions = [
            TicketEventRow.user_id == user_id,
            TicketEventRow.ticket_id == ticket_id,
        ]
        async with self._sf() as session:
            count = await session.scalar(
                select(func.count()).select_from(TicketEventRow).where(*conditions)
            ) or 0
            if count:
                await session.execute(delete(TicketEventRow).where(*conditions))
                await session.commit()
            return count
