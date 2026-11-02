"""Supervisor clarification tool."""

from __future__ import annotations

from langgraph.config import get_stream_writer
from langchain.tools import tool

SAFE_TERMINAL_FAILURE_REPLY = "抱歉，暂时无法完成这一步，请稍后重试。"


@tool("ask_clarification", return_direct=True)
async def ask_clarification(question: str) -> str:
    """Ask one focused customer question when required information is missing.

    Use this tool when the request cannot be handled reliably without a specific
    detail from the customer. The question must be customer-facing and concise.
    Call this tool by itself, never in the same batch as another tool. It ends
    the Agent run immediately and sends the question to the customer.
    """
    writer = get_stream_writer()
    try:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("clarification question must be a non-empty string")
        question = question.strip()
        writer({
            "event": "terminal_response",
            "tool": "ask_clarification",
            "status": "success",
            "message": question,
        })
        return question
    except Exception:
        writer({
            "event": "terminal_response",
            "tool": "ask_clarification",
            "status": "failure",
            "message": SAFE_TERMINAL_FAILURE_REPLY,
        })
        return SAFE_TERMINAL_FAILURE_REPLY
