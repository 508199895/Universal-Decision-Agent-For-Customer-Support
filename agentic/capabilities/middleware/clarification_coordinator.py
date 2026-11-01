"""Coordinate clarification tool calls before the Agent executes a tool batch."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import ToolMessage


class ClarificationCoordinatorMiddleware(AgentMiddleware):
    """Reject mixed clarification batches and end on a standalone clarification."""

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return self._coordinate(state)

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        return self._coordinate(state)

    @staticmethod
    def _coordinate(state) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        ai_message = messages[-1]
        tool_calls = getattr(ai_message, "tool_calls", None)
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        clarification_calls = [
            call for call in tool_calls
            if isinstance(call, dict) and call.get("name") == "ask_clarification"
        ]
        if not clarification_calls:
            return None

        for call in tool_calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not isinstance(call.get("name"), str)
            ):
                raise ValueError("each coordinated tool call must have a name and id")

        if len(tool_calls) > 1:
            messages = []
            clarification_ids = {call["id"] for call in clarification_calls}
            for call in tool_calls:
                is_clarification = call["id"] in clarification_ids
                messages.append(ToolMessage(
                    id=f"clarification-coordinator:{call['id']}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    content=json.dumps({
                        "status": "deferred" if is_clarification else "blocked",
                        "message": (
                            "澄清调用与其他工具不能同批执行；本批所有工具均未执行。"
                            "请重新判断下一步，并一次只发起一个工具调用。"
                        ),
                    }, ensure_ascii=False),
                ))
            return {"messages": messages, "jump_to": "model"}

        call = clarification_calls[0]
        args = call.get("args")
        question = args.get("question") if isinstance(args, dict) else None
        if not isinstance(question, str) or not question.strip():
            result = {
                "status": "error",
                "message": "澄清问题必须是非空字符串，请修正后重试。",
            }
            jump_to = "model"
        else:
            result = {"status": "asked", "question": question.strip()}
            jump_to = "end"

        return {
            "messages": [ToolMessage(
                id=f"clarification-coordinator:{call['id']}",
                name="ask_clarification",
                tool_call_id=call["id"],
                content=json.dumps(result, ensure_ascii=False),
            )],
            "jump_to": jump_to,
        }
