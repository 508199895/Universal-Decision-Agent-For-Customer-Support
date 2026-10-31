"""Build account query tools that scope every call to the ticket owner."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from agentic.capabilities.middleware.token_control import TicketRuntimeContext


ACCOUNT_TOOL_NAMES = frozenset({
    "account_account_get_user",
    "account_account_get_user_reservations",
})


def _decode_result(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _profile_belongs_to_user(profile: Any, external_user_id: str) -> bool:
    if not isinstance(profile, dict):
        return False
    external_profile = profile.get("external_user")
    core_profile = profile.get("core_user")
    if external_profile is None and core_profile is None:
        return False
    return not (
        external_profile is not None
        and (not isinstance(external_profile, dict)
             or external_profile.get("user_id") != external_user_id)
    ) and not (
        core_profile is not None
        and (not isinstance(core_profile, dict)
             or core_profile.get("external_user_id") != external_user_id)
    )


def bind_current_user_account_tools(tools: Iterable[BaseTool]) -> tuple[BaseTool, ...]:
    """Wrap account tools so the model cannot choose the queried user ID."""
    source_tools = tuple(tools)
    sources_by_name = {source.name: source for source in source_tools}
    bound = []
    for source in source_tools:
        if source.name not in ACCOUNT_TOOL_NAMES:
            bound.append(source)
            continue

        def make_wrapper(account_tool: BaseTool):
            @tool(account_tool.name, description=account_tool.description)
            async def read_current_user(runtime: ToolRuntime[TicketRuntimeContext]) -> str:
                external_user_id = runtime.context.external_user_id
                if not isinstance(external_user_id, str) or not external_user_id:
                    raise RuntimeError("account query is missing the trusted ticket owner")
                try:
                    if account_tool.name == "account_account_get_user_reservations":
                        profile_tool = sources_by_name.get("account_account_get_user")
                        if profile_tool is None:
                            raise RuntimeError("account profile tool is unavailable")
                        profile = _decode_result(await profile_tool.ainvoke(
                            {"external_user_id": external_user_id},
                        ))
                        if not _profile_belongs_to_user(profile, external_user_id):
                            return json.dumps({
                                "error": "当前账户资料未能确认，预约记录无法核实。",
                            }, ensure_ascii=False)

                    result = _decode_result(await account_tool.ainvoke(
                        {"external_user_id": external_user_id},
                    ))
                    if account_tool.name == "account_account_get_user":
                        if not _profile_belongs_to_user(result, external_user_id):
                            raise ValueError("account profile is outside the current-user scope")
                    else:
                        if not isinstance(result, list) or any(
                            not isinstance(reservation, dict)
                            or reservation.get("user_id") != external_user_id
                            for reservation in result
                        ):
                            raise ValueError("account reservations are outside the current-user scope")
                    return json.dumps(result, ensure_ascii=False, default=str)
                except Exception:
                    return json.dumps({"error": "账户查询失败或结果范围无法核实。"}, ensure_ascii=False)

            return read_current_user

        bound.append(make_wrapper(source))
    return tuple(bound)



