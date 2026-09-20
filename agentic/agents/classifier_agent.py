from __future__ import annotations

from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_classifier_agent(*, model: BaseChatModel) -> Runnable:
    """
    Classifier Agent.
    """

    system = (
        "You are the Classifier Agent for UDA-Hub.\n"
        "Classify the ticket into:\n"
        "  - issue_type: login, billing, reservation, subscription, technical, refund, other\n"
        "  - urgency: low, medium, high\n"
        "  - complexity: low, medium, high\n"
        "  - should_escalate_immediately: true/false\n"
        "  - rationale: explanation\n\n"
        "Return ONLY valid JSON with these fields."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Ticket content: {ticket_content}\n"
                "Normalized issue: {normalized_issue}\n"
                "Sentiment: {sentiment}\n"
                "Channel: {channel}\n"
                "Tags: {tags}\n\n"
                "Return ONLY JSON."
            ),
        ]
    )

    chain: Runnable = prompt | model.with_structured_output(method="json_mode")
    return chain
