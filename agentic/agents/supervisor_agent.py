from __future__ import annotations

from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_supervisor_agent(*, model: BaseChatModel) -> Runnable:
    """
    Supervisor Agent.

    Decides the next step in the workflow:
      - "resolver"    -> try to resolve using KB + tools
      - "escalation"  -> prepare human handoff
      - "done"        -> ticket is resolved, nothing else to do
    """

    # Resolver-first policy:
    # - If resolver_status is None (no attempt yet)        -> "resolver"
    # - If resolver_status == "needs_escalation"          -> "escalation"
    # - If resolver_status == "resolved" and confidence>=0.7 -> "done"
    # - Otherwise, fall back:
    #       if urgency is high and complexity is high -> "escalation"
    #       else -> "resolver"
    system = (
        "You are the Supervisor Agent for UDA-Hub.\n"
        "Your job is to decide which specialized agent should handle the ticket next.\n\n"
        "POSSIBLE next_step values:\n"
        "  - \"resolver\"\n"
        "  - \"escalation\"\n"
        "  - \"done\"\n\n"
        "Decision policy:\n"
        "1) If resolver_status is null / None (no resolver attempt yet):\n"
        "      next_step = \"resolver\".\n"
        "2) If resolver_status == \"needs_escalation\":\n"
        "      next_step = \"escalation\".\n"
        "3) If resolver_status == \"resolved\" AND resolver_confidence >= 0.7:\n"
        "      next_step = \"done\".\n"
        "4) Otherwise, use urgency/complexity as a fallback:\n"
        "      - If urgency is \"high\" AND complexity is \"high\": next_step = \"escalation\".\n"
        "      - Else: next_step = \"resolver\".\n\n"
        "You MUST return JSON with:\n"
        "  - next_step: one of \"resolver\", \"escalation\", or \"done\"\n"
        "  - reason: brief explanation of why you chose that step\n\n"
        "Return ONLY valid JSON."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Summary: {summary}\n"
                "Issue type: {issue_type}\n"
                "Urgency: {urgency}\n"
                "Complexity: {complexity}\n"
                "Resolver status: {resolver_status}\n"
                "Resolver confidence: {resolver_confidence}\n\n"
                "Return ONLY JSON."
            ),
        ]
    )

    chain: Runnable = prompt | model.with_structured_output(method="json_mode")
    return chain
