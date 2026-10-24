from __future__ import annotations

import asyncio
from typing import List, Optional

from langchain_core.tools import BaseTool

from agentic.integrations.mcp.client import aget_tools_for_servers, get_tools_for_servers


# ---------- Async versions (for Jupyter / async flows) ----------

async def aget_kb_tools() -> List[BaseTool]:
    """
    Async: return all LangChain tools from the KB MCP server.
    """
    return await aget_tools_for_servers("kb")


async def aget_kb_search_tool() -> Optional[BaseTool]:
    tools = await aget_kb_tools()
    for tool in tools:
        if "kb_search" in tool.name:
            return tool
    return None


async def aget_kb_get_tool() -> Optional[BaseTool]:
    tools = await aget_kb_tools()
    for tool in tools:
        if "kb_get" in tool.name:
            return tool
    return None


# ---------- Sync versions (for scripts only) ----------

def get_kb_tools() -> List[BaseTool]:
    """
    Sync: return all KB tools.
    Use this in scripts (not in Jupyter).
    """
    return get_tools_for_servers("kb")


def get_kb_search_tool() -> Optional[BaseTool]:
    tools = get_kb_tools()
    for tool in tools:
        if "kb_search" in tool.name:
            return tool
    return None


def get_kb_get_tool() -> Optional[BaseTool]:
    tools = get_kb_tools()
    for tool in tools:
        if "kb_get" in tool.name:
            return tool
    return None
