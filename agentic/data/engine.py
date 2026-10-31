"""Async SQLAlchemy engine and session lifecycle for application persistence."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentic.data.base import Base
from agentic.data.tickets.TicketEventModel import TicketEventRow as _ticket_event_model  # noqa: F401
from agentic.data.tickets.TicketModel import TicketRow as _ticket_model  # noqa: F401

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_engine(database: str | Path) -> AsyncEngine:
    """Create the SQLite engine, initialize ORM tables, and expose its sessions."""
    global _engine, _session_factory

    path = Path(database).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{path.as_posix()}"
    _engine = create_async_engine(url)

    @event.listens_for(_engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    try:
        async with _engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except Exception:
        await close_engine()
        raise
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("persistence engine has not been initialized")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("persistence engine has not been initialized")
    return _session_factory


async def close_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
