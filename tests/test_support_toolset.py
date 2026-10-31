from __future__ import annotations

import asyncio
import json

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agentic.agents.agent import AGENT_NAMES, get_agent_soul
from agentic.capabilities.tools.account_tools import bind_current_user_account_tools
from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.capabilities.tools import support_toolset


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


def test_agent_tool_allowlists_are_limited_by_role(monkeypatch):
    expected_tools = {
        "supervisor": {
            "ask_clarification",
            "query_public_knowledge",
            "lookup_current_user_account",
        },
        "query": {
            "qdrant_qdrant-find",
        },
        "validate": set(),
        "account": {
            "account_account_get_user",
            "account_account_get_user_reservations",
        },
        "escalation": set(),
    }

    for name in AGENT_NAMES:
        config = get_agent_soul(name)
        assert set(config.get("tools", [])) == expected_tools[name]


def test_missing_tools_config_defaults_to_no_tools(monkeypatch):
    raw_config = dict(get_agent_soul("query"))
    raw_config.pop("tools")
    assert raw_config.get("tools", []) == []


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
    agent = create_agent(
        model=model,
        tools=support_toolset.validate_support_tools(
            [*tools, _tool("memory_memory_search")],
        ),
    )

    result = asyncio.run(agent.ainvoke({"messages": [("human", "Check my account")]}))

    assert result["messages"][-2].content == "account data"
    assert result["messages"][-1].content == "I found your account data."


def test_account_tools_hide_identity_argument_and_use_ticket_context():
    calls = []

    async def read_profile(external_user_id: str):
        calls.append(external_user_id)
        return '{"external_user":{"user_id":"' + external_user_id + '"}}'

    raw_tool = StructuredTool(
        name="account_account_get_user",
        description="Read the current user's profile",
        args_schema={
            "type": "object",
            "properties": {"external_user_id": {"type": "string"}},
            "required": ["external_user_id"],
        },
        coroutine=read_profile,
    )
    wrapped = bind_current_user_account_tools([raw_tool])[0]

    class ToolCallingModel(GenericFakeChatModel):
        def bind_tools(self, tools, **_kwargs):
            assert {tool.name for tool in tools} == {"account_account_get_user"}
            assert "external_user_id" not in tools[0].tool_call_schema.model_fields
            return self

    model = ToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "account_account_get_user", "args": {}, "id": "call-profile",
        }]),
        AIMessage(content="Profile checked."),
    ]))
    agent = create_agent(
        model=model, tools=(wrapped,), context_schema=TicketRuntimeContext,
    )

    result = asyncio.run(agent.ainvoke(
        {"messages": [("human", "Check my profile")]},
        context=TicketRuntimeContext(journal=object(), external_user_id="trusted-user"),
    ))

    assert calls == ["trusted-user"]
    assert "trusted-user" in result["messages"][-2].content


def test_account_reservation_tool_returns_current_user_list_for_agent_matching():
    calls = []
    async def read_profile(external_user_id: str):
        calls.append(("profile", external_user_id))
        return json.dumps({
            "external_user": {"user_id": external_user_id},
            "core_user": None,
        })

    async def list_reservations(external_user_id: str):
        calls.append(external_user_id)
        return [
            {"reservation_id": "23e1c1", "user_id": external_user_id, "status": "confirmed"},
            {"reservation_id": "24008a", "user_id": external_user_id, "status": "cancelled"},
        ]

    raw_tool = StructuredTool(
        name="account_account_get_user_reservations",
        description="Read current-user reservations",
        args_schema={
            "type": "object",
            "properties": {"external_user_id": {"type": "string"}},
            "required": ["external_user_id"],
        },
        coroutine=list_reservations,
    )
    raw_profile_tool = StructuredTool(
        name="account_account_get_user",
        description="Read current-user profile",
        args_schema={
            "type": "object",
            "properties": {"external_user_id": {"type": "string"}},
            "required": ["external_user_id"],
        },
        coroutine=read_profile,
    )
    wrapped = next(
        item for item in bind_current_user_account_tools([raw_profile_tool, raw_tool])
        if item.name == "account_account_get_user_reservations"
    )

    class ToolCallingModel(GenericFakeChatModel):
        def bind_tools(self, tools, **_kwargs):
            assert "external_user_id" not in tools[0].tool_call_schema.model_fields
            return self

    model = ToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "account_account_get_user_reservations", "args": {}, "id": "call-reservations",
        }]),
        AIMessage(content="Reservation checked."),
    ]))
    agent = create_agent(
        model=model, tools=(wrapped,), context_schema=TicketRuntimeContext,
    )

    result = asyncio.run(agent.ainvoke(
        {"messages": [("human", "What is my reservation 23e1c1 status?")]},
        context=TicketRuntimeContext(
            journal=object(),
            external_user_id="trusted-user",
        ),
    ))

    assert calls == [("profile", "trusted-user"), "trusted-user"]
    assert json.loads(result["messages"][-2].content) == [
        {"reservation_id": "23e1c1", "user_id": "trusted-user", "status": "confirmed"},
        {"reservation_id": "24008a", "user_id": "trusted-user", "status": "cancelled"},
    ]


def test_reservation_query_does_not_treat_unknown_user_as_no_reservations():
    calls = []

    async def read_unknown_profile(external_user_id: str):
        calls.append("profile")
        return json.dumps({"external_user": None, "core_user": None})

    async def read_reservations(external_user_id: str):
        calls.append("reservations")
        return []

    schema = {
        "type": "object",
        "properties": {"external_user_id": {"type": "string"}},
        "required": ["external_user_id"],
    }
    wrapped = bind_current_user_account_tools([
        StructuredTool(
            name="account_account_get_user",
            description="Read current-user profile",
            args_schema=schema,
            coroutine=read_unknown_profile,
        ),
        StructuredTool(
            name="account_account_get_user_reservations",
            description="Read current-user reservations",
            args_schema=schema,
            coroutine=read_reservations,
        ),
    ])
    reservation_tool = next(
        item for item in wrapped if item.name == "account_account_get_user_reservations"
    )

    class ToolCallingModel(GenericFakeChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    model = ToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "account_account_get_user_reservations", "args": {}, "id": "call-unknown-reservations",
        }]),
        AIMessage(content="The account could not be confirmed."),
    ]))
    agent = create_agent(
        model=model, tools=(reservation_tool,), context_schema=TicketRuntimeContext,
    )

    result = asyncio.run(agent.ainvoke(
        {"messages": [("human", "Do I have any reservations?")]},
        context=TicketRuntimeContext(journal=object(), external_user_id="unknown-user"),
    ))

    assert calls == ["profile"]
    assert json.loads(result["messages"][-2].content)["error"]


def test_account_tool_blocks_profile_data_for_another_user_before_model_sees_it():
    async def read_foreign_profile(external_user_id: str):
        return json.dumps({
            "external_user": {"user_id": "another-user", "email": "private@example.test"},
            "core_user": None,
        })

    raw_tool = StructuredTool(
        name="account_account_get_user",
        description="Read the current user's profile",
        args_schema={
            "type": "object",
            "properties": {"external_user_id": {"type": "string"}},
            "required": ["external_user_id"],
        },
        coroutine=read_foreign_profile,
    )
    wrapped = bind_current_user_account_tools([raw_tool])[0]

    class ToolCallingModel(GenericFakeChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    model = ToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "account_account_get_user", "args": {}, "id": "call-foreign-profile",
        }]),
        AIMessage(content="I could not verify that profile."),
    ]))
    agent = create_agent(
        model=model, tools=(wrapped,), context_schema=TicketRuntimeContext,
    )

    result = asyncio.run(agent.ainvoke(
        {"messages": [("human", "Check my profile")]},
        context=TicketRuntimeContext(journal=object(), external_user_id="trusted-user"),
    ))

    tool_result = result["messages"][-2]
    assert tool_result.type == "tool"
    assert "private@example.test" not in tool_result.content
