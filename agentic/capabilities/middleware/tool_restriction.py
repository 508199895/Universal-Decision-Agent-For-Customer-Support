"""Keep terminal customer-response tools out of parallel tool batches."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import ToolMessage


class ToolRestrictionMiddleware(AgentMiddleware):
    """Reject batches that mix a terminal response tool with another tool."""

    _TERMINAL_TOOLS = {"ask_clarification", "request_human_handoff"}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return self._restrict(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        return self._restrict(state)

    @classmethod
    def _restrict(cls, state) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None
        tool_calls = getattr(messages[-1], "tool_calls", None)
        if not isinstance(tool_calls, list) or len(tool_calls) < 2:
            return None
        if not any(
            isinstance(call, dict) and call.get("name") in cls._TERMINAL_TOOLS
            for call in tool_calls
        ):
            return None

        tool_messages = []
        for call in tool_calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not isinstance(call.get("name"), str)
            ):
                raise ValueError("each tool call must have a name and id")
            tool_messages.append(ToolMessage(
                id=f"tool-restriction:{call['id']}",
                name=call["name"],
                tool_call_id=call["id"],
                content=json.dumps({
                    "status": "blocked",
                    "message": (
                        "澄清和转人工工具必须单独调用；本批工具均未执行。"
                        "请重新判断，并单独发起一个工具调用。"
                    ),
                }, ensure_ascii=False),
            ))
        return {"messages": tool_messages, "jump_to": "model"}
