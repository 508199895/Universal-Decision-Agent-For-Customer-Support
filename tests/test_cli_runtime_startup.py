import sys
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic.app import cli
from agentic.capabilities.support_toolset import SupportToolConfigurationError
from agentic.orchestration import workflow


def test_default_runtime_opens_checkpointer_for_async_session(tmp_path, monkeypatch):
    checkpointers = []
    support_agents = []
    graph = object()
    monkeypatch.setattr(
        workflow,
        "build_workflow",
        lambda **kwargs: checkpointers.append(kwargs["checkpointer"]) or graph,
    )
    monkeypatch.setattr(workflow, "_default_agent_database", lambda: tmp_path / "agent.sqlite")
    monkeypatch.setattr(
        workflow,
        "_default_dependencies",
        lambda: workflow.WorkflowDependencies(model=object()),
    )
    monkeypatch.setattr(workflow, "acollect_support_tools", lambda: _resolved(("tool",)))
    monkeypatch.setattr(
        workflow,
        "build_customer_support_agent",
        lambda **kwargs: support_agents.append(kwargs) or object(),
    )

    async def exercise_runtime():
        async with workflow.default_workflow_runtime() as (opened_graph, event_store):
            assert opened_graph is graph
            assert event_store.database == str(tmp_path / "agent.sqlite")
            assert len(checkpointers) == 1
            assert isinstance(checkpointers[0], AsyncSqliteSaver)
            assert len(support_agents) == 1
            assert support_agents[0]["tools"] == ("tool",)

    asyncio.run(exercise_runtime())


async def _resolved(value):
    return value


def test_support_tool_configuration_failure_stops_before_workflow_startup(tmp_path, monkeypatch):
    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(workflow, "_default_agent_database", lambda: tmp_path / "agent.sqlite")
    monkeypatch.setattr(
        workflow,
        "_default_dependencies",
        lambda: workflow.WorkflowDependencies(model=SimpleNamespace(root_client=client)),
    )

    async def fail_collection():
        raise SupportToolConfigurationError("required support tool is missing")

    async def should_not_start_workflow(*_args, **_kwargs):
        raise AssertionError("workflow startup must follow successful tool collection")

    monkeypatch.setattr(workflow, "acollect_support_tools", fail_collection)
    monkeypatch.setattr(workflow, "persistent_workflow", should_not_start_workflow)

    async def exercise_runtime():
        with pytest.raises(SupportToolConfigurationError, match="missing"):
            async with workflow.default_workflow_runtime():
                pytest.fail("customer input must not be accepted after configuration failure")

    asyncio.run(exercise_runtime())
    assert client.closed


def test_cli_keeps_runtime_open_until_chat_loop_finishes(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["uda", "--mode", "chat"])

    @asynccontextmanager
    async def runtime():
        calls.append("open")
        yield object(), object()
        calls.append("close")

    async def chat_loop(graph, event_store):
        assert graph is not None
        assert event_store is not None
        calls.append("chat")

    monkeypatch.setattr(cli, "default_workflow_runtime", runtime)
    monkeypatch.setattr(cli, "interactive_chat_loop", chat_loop)

    asyncio.run(cli.main())

    assert calls == ["open", "chat", "close"]
