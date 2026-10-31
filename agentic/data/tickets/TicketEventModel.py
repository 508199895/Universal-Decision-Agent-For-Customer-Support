"""ORM model for the append-only ticket event stream."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from agentic.data.base import Base


class TicketEventRow(Base):
    __tablename__ = "ticket_events"

    event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    ticket_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tickets.ticket_id"), nullable=False,
    )
    seq: Mapped[int] = mapped_column(nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    event_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        CheckConstraint(
            "category IN ('message', 'trace', 'outputs', 'error', 'middleware')",
            name="ck_ticket_events_category",
        ),
        UniqueConstraint("user_id", "seq", name="uq_ticket_events_user_seq"),
        Index("ix_ticket_events_ticket", "user_id", "ticket_id", "seq"),
    )
