"""User-scoped live event delivery for the customer-facing SSE stream."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from agentic.data.tickets.TicketEventStore import TicketEventStore
from agentic.data.tickets.user_context import reset_current_user_id, set_current_user_id

logger = logging.getLogger(__name__)


class UserEventHub:
    """Fan out persisted customer events to current subscribers for each user."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    @asynccontextmanager
    async def subscribe(self, user_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        subscribers = self._subscribers[user_id]
        subscribers.add(queue)
        try:
            yield queue
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(user_id, None)

    def publish(self, user_id: str, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(user_id, ())):
            queue.put_nowait(event)


class TicketEventPublisher:
    """Persist events when possible, then broadcast them to live SSE clients."""

    def __init__(self, event_store: TicketEventStore, hub: UserEventHub) -> None:
        self._event_store = event_store
        self._hub = hub

    async def publish(
        self,
        *,
        user_id: str,
        ticket_id: str,
        event_type: str,
        category: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {**data, "ticket_id": ticket_id}
        user_token = set_current_user_id(user_id)
        try:
            try:
                stored = await self._event_store.put(
                    ticket_id=ticket_id,
                    event_type=event_type,
                    category=category,
                    content=payload,
                )
            except Exception:
                logger.exception(
                    "Could not persist customer event %s for ticket %s; "
                    "publishing it to live subscribers",
                    event_type,
                    ticket_id,
                )
                event = {"event": event_type, "data": payload}
            else:
                event = {
                    "id": str(stored["seq"]),
                    "event": event_type,
                    "data": payload,
                }
        finally:
            reset_current_user_id(user_token)

        self._hub.publish(user_id, event)
        return event


def encode_sse(event: dict[str, Any]) -> str:
    """Encode one event using the standard SSE wire format."""
    event_id = event.get("id")
    event_type = str(event["event"])
    if event_id is not None and any(
        character in str(event_id) + event_type for character in "\r\n\0"
    ):
        raise ValueError("SSE id and event type must be single-line values")
    data = json.dumps(event["data"], ensure_ascii=False, separators=(",", ":"))
    id_line = f"id: {event_id}\n" if event_id is not None else "id:\n"
    return f"{id_line}event: {event_type}\ndata: {data}\n\n"
