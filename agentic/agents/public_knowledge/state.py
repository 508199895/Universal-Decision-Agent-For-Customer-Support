"""State and structured results for the public-knowledge expert graph."""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required


class ValidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_relevance: bool
    faithfulness: bool
    answer_relevance: bool
    feedback: str
    failure_reasons: list[str] = Field(default_factory=list)
    revision_instruction: str = ""


class PublicKnowledgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    answer: str | None
    evidence: list[dict[str, str]]
    validation: ValidateResult
    attempts: int
    failure_summary: str = ""


class PublicKnowledgeState(AgentState, total=False):
    task: Required[str]
    candidate: str
    evidence: list[dict[str, str]]
    attempt: int
    feedback: str
    validation: dict[str, Any]
    validated: bool
    failure_summary: str
    result: PublicKnowledgeResult | None
