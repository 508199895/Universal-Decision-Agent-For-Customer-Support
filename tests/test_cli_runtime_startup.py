import sys
import asyncio
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic.app import cli
from agentic.orchestration import workflow


def test_default_runtime_opens_checkpointer_for_async_session(tmp_path, monkeypatch):
    checkpointers = []
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

    async def exercise_runtime():
        async with workflow.default_workflow_runtime() as (opened_graph, event_store):
            assert opened_graph is graph
            assert event_store.database == str(tmp_path / "agent.sqlite")
            assert len(checkpointers) == 1
            assert isinstance(checkpointers[0], AsyncSqliteSaver)

    asyncio.run(exercise_runtime())


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
