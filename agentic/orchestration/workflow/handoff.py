"""Deterministic human-handoff material and customer notice projection."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from agentic.orchestration.workflow.state import validate_state


HANDOFF_NOTICE = "已转人工，等待人工客服回应"


def build_handoff_messages(
    state: dict[str, Any],
    *,
    ticket_id: str,
) -> tuple[SystemMessage, SystemMessage]:
    """Persist the two escalation fields and the fixed customer notice."""
    validate_state(state)
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        raise ValueError("handoff requires a context ticket_id")
    supervisor = state.get("supervisor", {})
    if supervisor.get("intent") != "escalation":
        raise ValueError("handoff requires supervisor.intent='escalation'")
    escalation = state.get("escalation", {})
    if not escalation.get("description") or not escalation.get("summary"):
        raise ValueError("handoff requires complete escalation description and summary")
    description = escalation["description"].strip()
    summary = escalation["summary"].strip()
    content = f"交接摘要：{summary}\n\n具体描述：{description}"
    return (
        SystemMessage(content=content, id=f"handoff:{ticket_id}:material"),
        SystemMessage(content=HANDOFF_NOTICE, id=f"handoff:{ticket_id}:notice"),
    )
