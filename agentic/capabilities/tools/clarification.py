"""Supervisor clarification tool."""

from __future__ import annotations

import json

from langchain.tools import tool


@tool("ask_clarification")
async def ask_clarification(question: str) -> str:
    """Ask one focused customer question when required information is missing.

    Use this tool when the request cannot be handled reliably without a specific
    detail from the customer. The question must be customer-facing and concise.
    Submit this tool by itself. The runtime defers it when it is combined with
    other tools, and ends the Agent run when it is called alone.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("clarification question must be a non-empty string")
    return json.dumps({
        "status": "asked",
        "question": question.strip(),
    }, ensure_ascii=False)
