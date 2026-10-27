
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool, StructuredTool

# Global adapter instance (lazy init)
_client: MCPAdapter | None = None
_client_config: Dict[str, Any] | None = None
_tools_cache: List[BaseTool] | None = None
_tools_cache_lock = asyncio.Lock()

# Resolve project root (uda-hub/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    with the current Python interpreter. The bundled Qdrant server uses stdio.
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
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_ROOT / "mcp_services" / "qdrant" / "server.py")],
            },
        }
    }


async def aget_client() -> MCPAdapter:
    """
    Async helper to initialize (once) and return the global MCPAdapter.

    Safe to use inside Jupyter and async code.
    """
    global _client, _client_config, _tools_cache
    config = _build_connections_config()
    if _client is None or config != _client_config:
        _client = MCPAdapter(config)
        _client_config = config
        _tools_cache = None
    return _client


def clear_client() -> None:
    """Drop the cached adapter after all MCP tool calls have completed."""
    global _client, _client_config, _tools_cache
    _client = None
    _client_config = None
    _tools_cache = None


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
    global _tools_cache
    client = await aget_client()
    if _tools_cache is None:
        async with _tools_cache_lock:
            if _tools_cache is None:
                _tools_cache = [
                    _with_text_result_compatibility(tool)
                    for tool in await client.list_tools()
                ]
    all_tools = _tools_cache

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

    Use this in scripts (like agentic.app.cli), NOT inside Jupyter notebooks.

        tools = get_tools_for_servers("kb")

    In Jupyter, prefer `await aget_tools_for_servers("kb")`.
    """
    return asyncio.run(aget_tools_for_servers(*servers))
