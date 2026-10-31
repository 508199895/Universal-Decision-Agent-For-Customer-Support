"""Tool for delivering Supervisor-prepared handoff material to Feishu."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from agentic.capabilities.middleware.token_control import TicketRuntimeContext


FeishuTextSender = Callable[[str], Awaitable[str]]


def build_feishu_handoff_tool(sender: FeishuTextSender | None) -> BaseTool:
    """Build a sender that uses ticket identity from trusted runtime context."""
    @tool("request_human_handoff")
    async def request_human_handoff(
        summary: str,
        description: str,
        runtime: ToolRuntime[TicketRuntimeContext],
    ) -> str:
        """Send the Supervisor-prepared handoff summary and description to Feishu."""
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        ticket_id = runtime.context.ticket_id
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError("handoff requires a context ticket_id")
        if sender is None:
            raise RuntimeError("Feishu handoff sender is not configured")

        summary = summary.strip()
        description = description.strip()
        message = (
            f"人工交接工单：{ticket_id.strip()}\n\n"
            f"交接摘要：\n{summary}\n\n"
            f"交接详情：\n{description}"
        )
        message_id = await sender(message)
        return json.dumps({
            "status": "sent",
            "ticket_id": ticket_id.strip(),
            "message_id": message_id,
            "summary": summary,
            "description": description,
        }, ensure_ascii=False)

    return request_human_handoff
