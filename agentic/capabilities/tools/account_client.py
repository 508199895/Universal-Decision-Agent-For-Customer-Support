from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool

from agentic.integrations.mcp.client import aget_tools_for_servers, get_tools_for_servers


# ---------- Async versions ----------

async def aget_account_tools() -> List[BaseTool]:
    return await aget_tools_for_servers("account")


async def aget_account_get_user_tool() -> Optional[BaseTool]:
    tools = await aget_account_tools()
    for tool in tools:
        if "account_get_user" in tool.name:
            return tool
    return None


async def aget_account_get_user_reservations_tool() -> Optional[BaseTool]:
    tools = await aget_account_tools()
    for tool in tools:
        if "account_get_user_reservations" in tool.name:
            return tool
    return None


# ---------- Sync versions (for scripts) ----------

def get_account_tools() -> List[BaseTool]:
    return get_tools_for_servers("account")


def get_account_get_user_tool() -> Optional[BaseTool]:
    tools = get_account_tools()
    for tool in tools:
        if "account_get_user" in tool.name:
            return tool
    return None


def get_account_get_user_reservations_tool() -> Optional[BaseTool]:
    tools = get_account_tools()
    for tool in tools:
        if "account_get_user_reservations" in tool.name:
            return tool
    return None
