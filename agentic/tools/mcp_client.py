
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool, StructuredTool

# Global adapter instance (lazy init)
_client: MCPAdapter | None = None

# Resolve project root (uda-hub/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SERVER_TOOL_PREFIXES = {
    "kb": ("kb_",),
    "account": ("account_",),
    "memory": ("memory_",),
    "qdrant": ("qdrant_",),
}


def _build_connections_config() -> Dict[str, Any]:
    """
    Build a canonical multi-server MCP configuration for MCPAdapter.

    We use stdio transport and launch each FastMCP server as a subprocess
    with the current Python interpreter. Qdrant is reached over Streamable HTTP.
    """
    return {
        "mcpServers": {
            "kb": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_ROOT / "mcp_services" / "kb" / "server.py")],
            },
            "account": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_ROOT / "mcp_services" / "account" / "server.py")],
            },
            "memory": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_ROOT / "mcp_services" / "memory" / "server.py")],
            },
            "qdrant": {
                "transport": "streamable-http",
                "url": os.getenv(
                    "QDRANT_MCP_URL",
                    "http://127.0.0.1:8000/mcp",
                ),
            },
        }
    }


async def aget_client() -> MCPAdapter:
    """
    Async helper to initialize (once) and return the global MCPAdapter.

    Safe to use inside Jupyter and async code.
    """
    global _client
    if _client is None:
        _client = MCPAdapter(_build_connections_config())
    return _client


def _normalize_text_result(result: Any) -> Any:
    """Preserve the project's pre-MCPAdapter behavior for text-only tools."""
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], dict)
        and result[0].get("type") == "text"
        and isinstance(result[0].get("text"), str)
    ):
        return result[0]["text"]
    return result


def _with_text_result_compatibility(tool: BaseTool) -> BaseTool:
    """Wrap an MCPAdapter tool so one text content block is returned as a string."""

    async def invoke_tool(**arguments: Any) -> Any:
        return _normalize_text_result(await tool.ainvoke(arguments))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=invoke_tool,
        return_direct=tool.return_direct,
        tags=tool.tags,
        metadata=tool.metadata,
    )


async def aget_tools_for_servers(*servers: str) -> List[BaseTool]:
    """
    Async helper: fetch LangChain tools from one or more MCP servers.

    MCPAdapter namespaces tools from a multi-server config as
    `<server>_<upstream_tool>`, so requested servers are filtered by namespace.
    """
    client = await aget_client()
    all_tools = [
        _with_text_result_compatibility(tool)
        for tool in await client.list_tools()
    ]

    if not servers:
        return all_tools

    prefixes = tuple(
        prefix
        for server in servers
        for prefix in _SERVER_TOOL_PREFIXES.get(server, (f"{server}_",))
    )
    filtered = [t for t in all_tools if t.name.startswith(prefixes)]
    return filtered


def get_tools_for_servers(*servers: str) -> List[BaseTool]:
    """
    Synchronous wrapper to fetch tools from one or more MCP servers.

    Use this in scripts (like 03_agentic_app.py), NOT inside Jupyter notebooks.

        tools = get_tools_for_servers("kb")

    In Jupyter, prefer `await aget_tools_for_servers("kb")`.
    """
    return asyncio.run(aget_tools_for_servers(*servers))
