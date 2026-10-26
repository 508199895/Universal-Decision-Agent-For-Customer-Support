from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel


class EscalationMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    description: str = Field(min_length=1)


def build_escalation_agent(*, model: BaseChatModel, middleware=(), context_schema=None):
    """
    Escalation Agent.
    """

    system = (
        "You are the Escalation Agent for UDA-Hub.\n"
        "Read the customer-support conversation and prepare exactly two handoff fields.\n"
        "summary: a concise summary of the customer's request and current outcome.\n"
        "description: a concrete account of relevant conversation, evidence, attempted handling, "
        "known or unresolved business facts, and why human help is needed.\n"
        "Use only facts present in the conversation; preserve uncertainty and do not invent details."
    )

    return create_agent(
        model=model,
        tools=[],
        system_prompt=system,
        response_format=EscalationMaterial,
        middleware=list(middleware),
        context_schema=context_schema,
    )
