"""Structured customer-facing response contract for the supervisor."""

from pydantic import BaseModel, ConfigDict, Field


class SupervisorResponse(BaseModel):
    """The supervisor's final answer or explicit handoff decision."""

    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    requires_handoff: bool = False
