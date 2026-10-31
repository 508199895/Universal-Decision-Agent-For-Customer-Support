from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool

from agentic.capabilities.mcp.client import aget_tools_for_servers, get_tools_for_servers


# ---------- Async versions ----------

async def aget_memory_tools() -> List[BaseTool]:
    return await aget_tools_for_servers("memory")


async def aget_memory_write_tool() -> Optional[BaseTool]:
    tools = await aget_memory_tools()
    for tool in tools:
        if "memory_write" in tool.name:
            return tool
    return None


async def aget_memory_search_tool() -> Optional[BaseTool]:
    tools = await aget_memory_tools()
    for tool in tools:
        if "memory_search" in tool.name:
            return tool
    return None


async def aget_memory_get_all_tool() -> Optional[BaseTool]:
    tools = await aget_memory_tools()
    for tool in tools:
        if "memory_get_all" in tool.name:
            return tool
    return None


# ---------- Sync versions (for scripts) ----------

def get_memory_tools() -> List[BaseTool]:
    return get_tools_for_servers("memory")


def get_memory_write_tool() -> Optional[BaseTool]:
    tools = get_memory_tools()
    for tool in tools:
        if "memory_write" in tool.name:
            return tool
    return None


def get_memory_search_tool() -> Optional[BaseTool]:
    tools = get_memory_tools()
    for tool in tools:
        if "memory_search" in tool.name:
            return tool
    return None


def get_memory_get_all_tool() -> Optional[BaseTool]:
    tools = get_memory_tools()
    for tool in get_memory_tools():
        if "memory_get_all" in tool.name:
            return tool
    return None
