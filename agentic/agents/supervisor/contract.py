"""Structured output contract for public-knowledge intent classification."""

from pydantic import BaseModel, ConfigDict


class SupervisorIntentResult(BaseModel):
    """Structured intent decision currently used by the V6 supervisor node."""

    model_config = ConfigDict(extra="forbid")
    is_public_knowledge: bool
