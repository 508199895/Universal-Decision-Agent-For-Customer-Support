import pytest

from agentic.tools.qdrant_client import aget_qdrant_tools


async def _get_qdrant_tool_by_name_fragment(fragment: str):
    tools = await aget_qdrant_tools()
    for tool in tools:
        if fragment in tool.name:
            return tool
    raise AssertionError(f"Qdrant tool containing '{fragment}' not found")


@pytest.mark.asyncio
async def test_qdrant_find_tool_is_available():
    """
    The running Qdrant MCP server should expose its read-only search tool.
    """
    qdrant_find = await _get_qdrant_tool_by_name_fragment("qdrant-find")

    assert qdrant_find is not None


@pytest.mark.asyncio
async def test_qdrant_store_tool_is_available():
    """
    The running Qdrant MCP server should expose its store tool without writing data.
    """
    qdrant_store = await _get_qdrant_tool_by_name_fragment("qdrant-store")

    assert qdrant_store is not None
