"""Deterministic human-handoff material and customer notice projection."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from agentic.v5_state import validate_state


HANDOFF_NOTICE = "已整理好后续处理材料，自动处理已结束；目前尚未确认人工已接手。"


def build_handoff_messages(
    state: dict[str, Any],
) -> tuple[SystemMessage, SystemMessage]:
    """Persist the two escalation fields and the fixed customer notice."""
    validate_state(state)
    ticket = state.get("ticket")
    if not isinstance(ticket, dict) or not isinstance(ticket.get("ticket_id"), str):
        raise ValueError("handoff requires a ticket_id")
    if ticket.get("status") != "handed_off":
        raise ValueError("handoff requires ticket.status='handed_off'")
    escalation = state.get("escalation", {})
    if not escalation.get("description") or not escalation.get("summary"):
        raise ValueError("handoff requires complete escalation description and summary")
    ticket_id = ticket["ticket_id"]
    description = escalation["description"].strip()
    summary = escalation["summary"].strip()
    content = f"交接摘要：{summary}\n\n具体描述：{description}"
    return (
        SystemMessage(content=content, id=f"handoff:{ticket_id}:material"),
        SystemMessage(content=HANDOFF_NOTICE, id=f"handoff:{ticket_id}:notice"),
    )
