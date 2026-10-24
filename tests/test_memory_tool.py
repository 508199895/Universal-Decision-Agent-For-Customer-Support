import json
import uuid
import pytest
from typing import Any, Dict

from agentic.capabilities.tools.memory_client import aget_memory_tools


def _get_default_cultpass_user() -> Dict[str, Any]:
    """
    Helper: load the first CultPass user from data/external/cultpass_users.jsonl.
    """
    path = "data/external/cultpass_users.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    user = json.loads(first_line)
    return user


async def _get_memory_tool_by_name_fragment(fragment: str):
    tools = await aget_memory_tools()
    for tool in tools:
        if fragment in tool.name:
            return tool
    raise AssertionError(f"Memory tool containing '{fragment}' not found")


@pytest.mark.asyncio
async def test_memory_write_and_search_roundtrip():
    """
    Memory MCP server should allow writing a memory and retrieving it via search.
    """
    user = _get_default_cultpass_user()
    external_user_id = user["id"]
    ticket_id = f"TEST-MEM-{uuid.uuid4().hex[:8]}"

    memory_write = await _get_memory_tool_by_name_fragment("memory_write")
    memory_search = await _get_memory_tool_by_name_fragment("memory_search")

    content = "Resolved login issue for CultPass user during test."
    metadata = {"issue_type": "login", "source": "pytest"}

    # Write memory (should not raise ToolException now)
    write_result = await memory_write.ainvoke(
        {
            "external_user_id": external_user_id,
            "ticket_id": ticket_id,
            "content": content,
            "metadata": metadata,
        }
    )

    # Don't rely heavily on the exact return shape; just ensure it responded
    assert write_result is not None

    # Search memory
    search_results = await memory_search.ainvoke(
        {
            "external_user_id": external_user_id,
            "query": "login",
            "limit": 10,
        }
    )

    # Normalize JSON-string → list
    if isinstance(search_results, str):
        assert search_results.strip() != "", "Memory search returned an empty string"
        parsed = json.loads(search_results)
    else:
        parsed = search_results

    # We now REQUIRE that at least one result exists after writing
    assert isinstance(parsed, list), "Memory search did not return a list"
    assert len(parsed) >= 1, "Memory search returned no results after write"

    # Ideally, one of the entries mentions "login"
    joined = " ".join(str(r) for r in parsed)
    assert "login" in joined.lower()
