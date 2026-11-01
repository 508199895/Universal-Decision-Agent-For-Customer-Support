"""Static tool collection for the currently selected customer-support scenes."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from langchain_core.tools import BaseTool

from agentic.capabilities.mcp.client import aget_tools_for_servers


# Memory and Qdrant administration tools are intentionally not part of the
# customer-facing agent; customer knowledge is retrieved through Qdrant find.
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


async def acollect_support_tools(
    enabled_tool_names: Iterable[str] | None = None,
) -> tuple[BaseTool, ...]:
    """Collect the configured support MCP tools once at application startup.

    With no explicit names, retain the legacy support-tool validation path.
    The config-driven path only requires globally enabled MCP tools.
    """
    if enabled_tool_names is not None:
        requested = set(enabled_tool_names)
        collection_names = set(requested)
        if "account_account_get_user_reservations" in requested:
            # The reservation wrapper uses the profile tool internally to verify scope.
            collection_names.add("account_account_get_user")
        server_by_prefix = {
            "account_": "account",
            "qdrant_": "qdrant",
        }
        servers = sorted({
            server for name in collection_names
            for prefix, server in server_by_prefix.items()
            if name.startswith(prefix)
        })
        unsupported = sorted(
            name for name in collection_names
            if not any(name.startswith(prefix) for prefix in server_by_prefix)
        )
        if unsupported:
            raise SupportToolConfigurationError(
                "Unsupported configured MCP tools: " + ", ".join(unsupported)
            )
        if not requested:
            return ()
        try:
            tools = tuple(await aget_tools_for_servers(*servers))
        except Exception as error:
            raise SupportToolConfigurationError(
                "Could not collect globally enabled support tools from configured MCP servers"
            ) from error
        names = [tool.name for tool in tools]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise SupportToolConfigurationError(
                "Duplicate MCP tool names: " + ", ".join(duplicates)
            )
        by_name = {item.name: item for item in tools}
        missing = sorted(collection_names - by_name.keys())
        if missing:
            raise SupportToolConfigurationError(
                "Configured MCP tools are unavailable: " + ", ".join(missing)
            )
        return tuple(by_name[name] for name in sorted(collection_names))

    try:
        tools = await aget_tools_for_servers()
    except Exception as error:
        raise SupportToolConfigurationError(
            "Could not collect customer-support tools from configured MCP servers "
            "(qdrant, account)"
        ) from error
    return validate_support_tools(tools)
