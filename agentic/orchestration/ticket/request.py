"""Project-specific input body for starting a ticket workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TicketStartBody(BaseModel):
    """Subset of the run-start body consumed by the ticket workflow."""

    input: dict[str, Any] | None = Field(
        default=None, description="Workflow input, usually {'messages': [...]}.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Ticket metadata persisted with the record.")
    config: dict[str, Any] | None = Field(default=None, description="RunnableConfig overrides passed to the workflow.")
    context: dict[str, Any] | None = Field(default=None, description="Ticket runtime options; model_name is currently consumed.")
    multitask_strategy: Literal["reject", "interrupt", "rollback"] = Field(
        default="reject", description="Admission policy when this user already has an active run.",
    )
