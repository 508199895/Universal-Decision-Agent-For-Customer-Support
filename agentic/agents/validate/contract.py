"""Structured output contract for public-answer review."""

from pydantic import BaseModel, ConfigDict


class ValidateResult(BaseModel):
    """Structured validation result for the V6 validate node."""

    model_config = ConfigDict(extra="forbid")
    context_relevance: bool
    faithfulness: bool
    answer_relevance: bool
    feedback: str
