"""Shared errors raised by ticket persistence stores."""

from __future__ import annotations


class TicketEventStoreError(RuntimeError):
    """Base error for ticket and event persistence operations."""


class TicketHistoryConflict(TicketEventStoreError):
    """Persisted ticket state conflicts with the requested operation."""


class TicketPersistenceError(TicketEventStoreError):
    """A durable write failed; processing must stop."""
