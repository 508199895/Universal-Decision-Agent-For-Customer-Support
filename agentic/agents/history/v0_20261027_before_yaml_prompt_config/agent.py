"""Application-lifetime assembly of the customer-support agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agentic.agents.prompt import (
    escalation_system_prompt,
    public_answer_reviewer_system_prompt,
    public_intent_system_prompt,
)
from agentic.agents.escalation.contract import EscalationMaterial
from agentic.agents.public_answer_reviewer.contract import PublicAnswerReview
from agentic.agents.public_intent.contract import PublicKnowledgeIntent
from agentic.capabilities.middleware.token_control import (
    TicketRuntimeContext,
    build_ticket_middleware,
)
from agentic.capabilities.support_toolset import validate_support_tools


@dataclass(frozen=True)
class AgentSet:
    """Prebuilt agent runnables shared by workflows in one runtime."""

    customer_support: Any | None = None
    escalation: Any | None = None
    public_intent: Any | None = None
    public_answer_reviewer: Any | None = None


# Keep the structured response contracts available through the assembly module.
__all__ = [
    "AgentSet", "build_agent_set", "build_customer_support_agent",
]


def build_customer_support_agent(
    *, model: BaseChatModel, tools: Sequence[BaseTool], middleware=(), context_schema=None,
):
    """Assemble the primary tool-calling agent with its fixed startup toolset."""
    if model is None:
        raise ValueError("customer-support agent requires a model")
    selected_tools = validate_support_tools(tools)
    return create_agent(
        model=model, tools=list(selected_tools), middleware=list(middleware),
        context_schema=context_schema,
    )


def build_escalation_agent(*, model: BaseChatModel, middleware=(), context_schema=None):
    if model is None:
        raise ValueError("escalation agent requires a model")
    return create_agent(
        model=model,
        tools=[],
        system_prompt=escalation_system_prompt(),
        response_format=EscalationMaterial,
        middleware=list(middleware),
        context_schema=context_schema,
    )


def build_public_intent_agent(*, model: BaseChatModel, middleware=(), context_schema=None):
    if model is None:
        raise ValueError("public-knowledge intent agent requires a model")
    return create_agent(
        model=model,
        tools=[],
        system_prompt=public_intent_system_prompt(),
        response_format=PublicKnowledgeIntent,
        middleware=list(middleware),
        context_schema=context_schema,
    )


def build_public_answer_reviewer(*, model: BaseChatModel, middleware=(), context_schema=None):
    if model is None:
        raise ValueError("public-answer reviewer requires a model")
    return create_agent(
        model=model,
        tools=[],
        system_prompt=public_answer_reviewer_system_prompt(),
        response_format=PublicAnswerReview,
        middleware=list(middleware),
        context_schema=context_schema,
    )


def build_agent_set(*, model: BaseChatModel, tools: Sequence[BaseTool]) -> AgentSet:
    """Build each agent once before compiling the workflow that uses them."""
    if model is None:
        raise ValueError("agent set requires a model")
    return AgentSet(
        customer_support=build_customer_support_agent(
            model=model,
            tools=tools,
            middleware=build_ticket_middleware(model),
            context_schema=TicketRuntimeContext,
        ),
        escalation=build_escalation_agent(
            model=model,
            middleware=build_ticket_middleware(model),
            context_schema=TicketRuntimeContext,
        ),
        public_intent=build_public_intent_agent(
            model=model,
            middleware=build_ticket_middleware(model),
            context_schema=TicketRuntimeContext,
        ),
        public_answer_reviewer=build_public_answer_reviewer(
            model=model,
            middleware=build_ticket_middleware(model),
            context_schema=TicketRuntimeContext,
        ),
    )
