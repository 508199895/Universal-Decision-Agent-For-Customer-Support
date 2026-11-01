import sys
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic.app import cli
from agentic.capabilities.tools.support_toolset import SupportToolConfigurationError
from agentic.lifespan import runtime as workflow_runtime


def test_default_runtime_opens_checkpointer_for_async_session(tmp_path, monkeypatch):
    checkpointers = []
    workflow_initializations = []
    graph = object()
    class MiddlewareModel:
        _llm_type = "fake-chat"

        def with_retry(self):
            return self
    monkeypatch.setattr(workflow_runtime, "_default_agent_database", lambda: tmp_path / "agent.sqlite")
    monkeypatch.setattr(workflow_runtime, "_default_model", lambda _config=None: MiddlewareModel())
    monkeypatch.setattr(workflow_runtime, "acollect_support_tools", lambda _names=None: _resolved(("tool",)))
    class FakeWorkflowManager:
        def initialize(self, **kwargs):
            checkpointers.append(kwargs["checkpointer"])
            workflow_initializations.append(kwargs)
            return graph

    monkeypatch.setattr(workflow_runtime, "WorkflowManager", FakeWorkflowManager)

    async def exercise_runtime():
        async with workflow_runtime.default_workflow_runtime() as (opened_graph, ticket_manager):
            assert opened_graph is graph
            assert ticket_manager.ticket_store.database == str(tmp_path / "agent.sqlite")
            assert len(checkpointers) == 1
            assert isinstance(checkpointers[0], AsyncSqliteSaver)
            assert len(workflow_initializations) == 1
            assert workflow_initializations[0]["tools"] == ["tool"]

    asyncio.run(exercise_runtime())


async def _resolved(value):
    return value


def test_support_tool_configuration_failure_stops_before_workflow_startup(tmp_path, monkeypatch):
    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(workflow_runtime, "_default_agent_database", lambda: tmp_path / "agent.sqlite")
    monkeypatch.setattr(workflow_runtime, "_default_model", lambda _config=None: SimpleNamespace(root_client=client))

    async def fail_collection(_names=None):
        raise SupportToolConfigurationError("required support tool is missing")

    async def should_not_start_workflow(*_args, **_kwargs):
        raise AssertionError("workflow startup must follow successful tool collection")

    monkeypatch.setattr(workflow_runtime, "acollect_support_tools", fail_collection)
    monkeypatch.setattr(workflow_runtime, "WorkflowManager", should_not_start_workflow)

    async def exercise_runtime():
        with pytest.raises(SupportToolConfigurationError, match="missing"):
            async with workflow_runtime.default_workflow_runtime():
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
