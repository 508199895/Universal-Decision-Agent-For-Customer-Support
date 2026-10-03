from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool

from .mcp_client import aget_tools_for_servers, get_tools_for_servers


# ---------- Async versions ----------

async def aget_qdrant_tools() -> List[BaseTool]:
    return await aget_tools_for_servers("qdrant")


async def aget_qdrant_find_tool() -> Optional[BaseTool]:
    tools = await aget_qdrant_tools()
    for tool in tools:
        if "qdrant-find" in tool.name:
            return tool
    return None


async def aget_qdrant_store_tool() -> Optional[BaseTool]:
    tools = await aget_qdrant_tools()
    for tool in tools:
        if "qdrant-store" in tool.name:
            return tool
    return None


# ---------- Sync versions (for scripts) ----------

def get_qdrant_tools() -> List[BaseTool]:
    return get_tools_for_servers("qdrant")


def get_qdrant_find_tool() -> Optional[BaseTool]:
    tools = get_qdrant_tools()
    for tool in tools:
        if "qdrant-find" in tool.name:
            return tool
    return None


def get_qdrant_store_tool() -> Optional[BaseTool]:
    tools = get_qdrant_tools()
    for tool in tools:
        if "qdrant-store" in tool.name:
            return tool
    return None
