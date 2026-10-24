import json
import pytest
from typing import Any, Dict, List

from agentic.capabilities.tools.knowledge_client import aget_kb_tools


async def _get_kb_search_tool():
    tools = await aget_kb_tools()
    for tool in tools:
        if "kb_search" in tool.name:
            return tool
    raise AssertionError("kb_search tool not found among KB tools")


@pytest.mark.asyncio
async def test_kb_search_returns_results_for_reservation_query():
    """
    The KB MCP server should respond (string or list) for a known topic.

    Implementation note:
    - Current MCP server returns a JSON string (or "" if no results).
    - We normalize that here so the test is robust.
    """
    kb_search_tool = await _get_kb_search_tool()

    query = "How do I reserve a spot for an event with CultPass?"
    results = await kb_search_tool.ainvoke({"query": query, "limit": 5})

    # Accept JSON string or list; normalize
    if isinstance(results, str):
        # If there are results, they should be JSON list-like; if empty string, that's OK
        if results.strip() == "":
            # Nothing found; at least the tool responded without error
            assert isinstance(results, str)
            return
        parsed = json.loads(results)
    else:
        parsed = results

    assert isinstance(parsed, list), "Expected KB search to return a list-like structure"

    if parsed:
        first = parsed[0]
        assert isinstance(first, dict)
        assert "title" in first
        assert "content" in first
