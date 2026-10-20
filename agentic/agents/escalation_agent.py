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
        "Read the customer-support conversation and prepare exactly two handoff fields.\n"
        "summary: a concise summary of the customer's request and current outcome.\n"
        "description: a concrete account of relevant conversation, evidence, attempted handling, "
        "known or unresolved business facts, and why human help is needed.\n"
        "Use only facts present in the conversation; preserve uncertainty and do not invent details.\n\n"
        'Return JSON with exactly these keys: "summary" and "description".\n'
        "Return ONLY valid JSON."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Full ticket conversation:\n{conversation}\n\n"
                "Return ONLY JSON."
            ),
        ]
    )

    chain: Runnable = prompt | model.with_structured_output(method="json_mode")
    return chain
