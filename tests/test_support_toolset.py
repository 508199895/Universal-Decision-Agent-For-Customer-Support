from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agentic.agents import customer_support_agent
from agentic.capabilities import support_toolset


def _tool(name: str) -> StructuredTool:
    async def invoke(**_arguments):
        return name

    return StructuredTool(
        name=name,
        description="test tool",
        args_schema={"type": "object", "properties": {}},
        coroutine=invoke,
    )


def _configured_tools():
    return [_tool(name) for name in support_toolset.REQUIRED_SUPPORT_TOOL_NAMES]


def test_tool_collection_requires_every_selected_capability():
    tools = _configured_tools()
    tools.pop()

    with pytest.raises(support_toolset.SupportToolConfigurationError, match="unavailable"):
        support_toolset.validate_support_tools(tools)


def test_tool_collection_rejects_duplicate_names():
    tools = _configured_tools()
    tools.append(tools[0])

    with pytest.raises(support_toolset.SupportToolConfigurationError, match="Duplicate"):
        support_toolset.validate_support_tools(tools)


def test_collection_is_once_and_excludes_non_support_tools(monkeypatch):
    calls = []
    tools = [*_configured_tools(), _tool("memory_memory_search"), _tool("qdrant_qdrant-store")]

    async def collect():
        calls.append("collect")
        return tools

    monkeypatch.setattr(support_toolset, "aget_tools_for_servers", collect)
    selected = asyncio.run(support_toolset.acollect_support_tools())

    assert calls == ["collect"]
    assert {tool.name for tool in selected} == support_toolset.REQUIRED_SUPPORT_TOOL_NAMES


def test_agent_calls_qdrant_rag_and_ignores_unselected_tools():
    class ControlledToolCallingModel(GenericFakeChatModel):
        def bind_tools(self, tools, **_kwargs):
            assert {tool.name for tool in tools} == support_toolset.REQUIRED_SUPPORT_TOOL_NAMES
            return self

    async def return_answer(**_arguments):
        return "account data"

    qdrant_tool = StructuredTool(
        name="qdrant_qdrant-find",
        description="Search the existing Qdrant RAG collection",
        args_schema={"type": "object", "properties": {}},
        coroutine=return_answer,
    )
    model = ControlledToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "qdrant_qdrant-find", "args": {}, "id": "call-rag",
        }]),
        AIMessage(content="I found your account data."),
    ]))
    tools = [tool for tool in _configured_tools() if tool.name != qdrant_tool.name]
    tools.append(qdrant_tool)
    agent = customer_support_agent.build_customer_support_agent(
        model=model,
        tools=[*tools, _tool("memory_memory_search")],
    )

    result = asyncio.run(agent.ainvoke({"messages": [("human", "Check my account")]}))

    assert result["messages"][-2].content == "account data"
    assert result["messages"][-1].content == "I found your account data."
