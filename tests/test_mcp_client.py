from __future__ import annotations

import sys

import pytest
from langchain_core.tools import StructuredTool

from agentic.tools import mcp_client


def test_connections_config_targets_mcp_v2_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_MCP_URL", "http://qdrant-mcp.test/mcp")

    config = mcp_client._build_connections_config()
    servers = config["mcpServers"]

    assert set(servers) == {"kb", "account", "memory", "qdrant"}
    assert servers["kb"]["command"] == sys.executable
    assert servers["qdrant"] == {
        "transport": "streamable-http",
        "url": "http://qdrant-mcp.test/mcp",
    }


@pytest.mark.asyncio
async def test_tools_are_filtered_by_mcp_server_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def return_text_block(**_arguments: object) -> list[dict[str, str]]:
        return [{"type": "text", "text": "ok"}]

    def fake_tool(name: str) -> StructuredTool:
        return StructuredTool(
            name=name,
            description="test tool",
            args_schema={"type": "object", "properties": {}},
            coroutine=return_text_block,
        )

    class FakeAdapter:
        async def list_tools(self):
            return [
                fake_tool("kb_kb_search"),
                fake_tool("account_account_get_user"),
                fake_tool("qdrant_qdrant-find"),
            ]

    async def fake_get_adapter() -> FakeAdapter:
        return FakeAdapter()

    monkeypatch.setattr(mcp_client, "aget_client", fake_get_adapter)

    tools = await mcp_client.aget_tools_for_servers("qdrant")

    assert [tool.name for tool in tools] == ["qdrant_qdrant-find"]
    assert await tools[0].ainvoke({}) == "ok"
