"""Task-local user identity for ticket persistence operations."""

from __future__ import annotations

from contextvars import ContextVar, Token

_current_user_id: ContextVar[str | None] = ContextVar("agentic_ticket_user_id", default=None)


def set_current_user_id(user_id: str) -> Token[str | None]:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    return _current_user_id.set(user_id)


def reset_current_user_id(token: Token[str | None]) -> None:
    _current_user_id.reset(token)


def require_current_user_id(*, method_name: str) -> str:
    user_id = _current_user_id.get()
    if user_id is None:
        raise RuntimeError(f"{method_name} called without a current user context")
    return user_id
