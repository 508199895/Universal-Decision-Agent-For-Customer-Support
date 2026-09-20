from __future__ import annotations

from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_escalation_agent(*, model: BaseChatModel) -> Runnable:
    """
    Escalation Agent.
    """

    system = (
        "You are the Escalation Agent for UDA-Hub.\n"
        "Prepare a structured handoff summary for a human agent.\n\n"
        "You MUST return JSON with:\n"
        "  - summary_for_human\n"
        "  - recommended_department\n"
        "  - proposed_next_steps\n"
        "  - include_prior_resolution_notes (true/false)\n\n"
        "Return ONLY valid JSON."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Ticket content: {ticket_content}\n"
                "Intake summary: {intake_summary}\n"
                "Sentiment: {sentiment}\n"
                "Classification: {classification}\n"
                "Resolver notes: {resolver_notes}\n\n"
                "Return ONLY JSON."
            ),
        ]
    )

    chain: Runnable = prompt | model.with_structured_output(method="json_mode")
    return chain
