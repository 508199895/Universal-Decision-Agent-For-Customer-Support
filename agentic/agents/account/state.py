"""State and structured result for the account expert graph."""

from __future__ import annotations

from typing import Literal

from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict
from typing_extensions import Required


class AccountResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    answer: str | None
    failure_summary: str = ""


class AccountState(AgentState, total=False):
    task: Required[str]
    result: AccountResult | None
