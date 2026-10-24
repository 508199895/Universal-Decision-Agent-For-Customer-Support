import json
import pytest
from typing import Any, Dict

from agentic.capabilities.tools.account_client import aget_account_tools


def _get_default_cultpass_user() -> Dict[str, Any]:
    """
    Helper: load the first CultPass user from data/external/cultpass_users.jsonl.
    """
    path = "data/external/cultpass_users.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    user = json.loads(first_line)
    return user  # expects keys like "id", "name"


async def _get_account_tool_by_name_fragment(fragment: str):
    tools = await aget_account_tools()
    for tool in tools:
        if fragment in tool.name:
            return tool
    raise AssertionError(f"Account tool containing '{fragment}' not found")


@pytest.mark.asyncio
async def test_account_get_user_returns_profile_for_known_user():
    """
    Account MCP server should return a user profile for a known external_user_id.
    """
    user = _get_default_cultpass_user()
    external_user_id = user["id"]

    get_user_tool = await _get_account_tool_by_name_fragment("account_get_user")
    result = await get_user_tool.ainvoke({"external_user_id": external_user_id})

    # Tools currently return JSON strings – normalize to dict
    if isinstance(result, str):
        assert result != "", "account_get_user returned empty string"
        parsed = json.loads(result)
    else:
        parsed = result

    assert isinstance(parsed, dict)

    # At least check some identifying properties exist
    assert "external_user" in parsed or "core_user" in parsed
    assert "reservation_count" in parsed or "ticket_count" in parsed


@pytest.mark.asyncio
async def test_account_get_user_reservations_returns_list():
    """
    Account MCP server should return a list of reservations for a known user.
    """
    user = _get_default_cultpass_user()
    external_user_id = user["id"]

    get_res_tool = await _get_account_tool_by_name_fragment(
        "account_get_user_reservations"
    )
    result = await get_res_tool.ainvoke({"external_user_id": external_user_id})

    # Normalize JSON-string → Python list
    if isinstance(result, str):
        assert result != "", "account_get_user_reservations returned empty string"
        parsed = json.loads(result)
    else:
        parsed = result

    assert isinstance(parsed, list)

    if parsed:
        first = parsed[0]
        assert isinstance(first, dict)
        assert "reservation_id" in first
        assert "experience_id" in first
