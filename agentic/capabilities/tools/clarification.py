"""Supervisor clarification tool."""

from __future__ import annotations

import json

from langchain.tools import tool


@tool("ask_clarification")
async def ask_clarification(question: str) -> str:
    """Ask the customer one focused question and finish this ticket run."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("clarification question must be a non-empty string")
    return json.dumps({
        "status": "asked",
        "question": question.strip(),
    }, ensure_ascii=False)
