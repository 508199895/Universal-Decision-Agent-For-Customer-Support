"""Static tool collection for the currently selected customer-support scenes."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from langchain_core.tools import BaseTool

from agentic.integrations.mcp.client import aget_tools_for_servers


# These are the current Qdrant RAG search and account query capabilities.
# Legacy KB MCP and Qdrant administration tools are not customer-facing tools.
REQUIRED_SUPPORT_TOOL_NAMES = frozenset(
    {
        "qdrant_qdrant-find",
        "account_account_get_user",
        "account_account_get_user_reservations",
    }
)


class SupportToolConfigurationError(ValueError):
    """The configured MCP tools cannot support the selected support scenes."""


def validate_support_tools(tools: Iterable[BaseTool]) -> tuple[BaseTool, ...]:
    """Validate and freeze the selected customer-support tools once at startup."""
    collected = tuple(tools)
    names = [tool.name for tool in collected]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise SupportToolConfigurationError(
            "Duplicate MCP tool names: " + ", ".join(duplicates)
        )

    by_name = {tool.name: tool for tool in collected}
    missing = sorted(REQUIRED_SUPPORT_TOOL_NAMES - by_name.keys())
    if missing:
        raise SupportToolConfigurationError(
            "Required customer-support MCP tools are unavailable: " + ", ".join(missing)
        )
    return tuple(by_name[name] for name in sorted(REQUIRED_SUPPORT_TOOL_NAMES))


async def acollect_support_tools() -> tuple[BaseTool, ...]:
    """Collect configured tools a single time and select the support allowlist."""
    try:
        tools = await aget_tools_for_servers()
    except Exception as error:
        raise SupportToolConfigurationError(
            "Could not collect customer-support tools from configured MCP servers "
            "(qdrant, account)"
        ) from error
    return validate_support_tools(tools)
