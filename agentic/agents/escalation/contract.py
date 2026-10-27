"""Structured output contract for the escalation agent."""

from pydantic import BaseModel, ConfigDict, Field


class EscalationMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    description: str = Field(min_length=1)
