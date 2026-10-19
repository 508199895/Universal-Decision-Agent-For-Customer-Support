"""Entry regression with controllable model/tools and real checkpoint storage."""
import asyncio
import os
import subprocess
import sys

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing import TypedDict


class TicketModel(FakeListChatModel):
    def with_structured_output(self, schema=None, **kwargs):
        return self | JsonOutputParser()


def test_workflow_import_needs_neither_credentials_nor_network():
    probe = """
import os
import socket
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False
os.environ.pop('DEEPSEEK_API_KEY', None)
def reject_network(*args, **kwargs):
    raise AssertionError('import attempted network access')
socket.socket.connect = reject_network
import agentic.workflow
import dump_mermaid
print('offline import OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True,
        text=True, timeout=30, env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "offline import OK" in result.stdout


def test_sync_ticket_runs_reuse_the_application_event_loop(monkeypatch):
    import agentic.workflow as workflow_module

    class AsyncClient:
        closed_loop = None

        async def close(self):
            self.closed_loop = asyncio.get_running_loop()

    class SyncClient:
        closed = False

        def close(self):
            self.closed = True

    class LoopBoundWorkflow:
        loops = []

        async def ainvoke(self, initial_state, config):
            current_loop = asyncio.get_running_loop()
            if self.loops and self.loops[-1].is_closed():
                raise RuntimeError("Event loop is closed")
            self.loops.append(current_loop)
            return initial_state

    cached_workflow = LoopBoundWorkflow()
    async_client = AsyncClient()
    sync_client = SyncClient()
    workflow_module._default_model = type(
        "DefaultModelResources", (),
        {"root_async_client": async_client, "root_client": sync_client},
    )()
    monkeypatch.setattr(workflow_module, "_default_workflow", lambda: cached_workflow)

    try:
        first = workflow_module.run_ticket({"ticket_id": "ticket-a"}, "ticket-a")
        second = workflow_module.run_ticket({"ticket_id": "ticket-b"}, "ticket-b")
    finally:
        monkeypatch.undo()
        close_runtime = getattr(workflow_module, "close_default_runtime", None)
        if close_runtime is not None:
            close_runtime()

    assert first["ticket"]["ticket_id"] == "ticket-a"
    assert second["ticket"]["ticket_id"] == "ticket-b"
    assert len(cached_workflow.loops) == 2
    assert cached_workflow.loops[0] is cached_workflow.loops[1]
    assert async_client.closed_loop is cached_workflow.loops[0]
    assert sync_client.closed
    assert cached_workflow.loops[0].is_closed()


def test_application_loop_closes_coroutine_when_shutting_down():
    from agentic.workflow import _ApplicationEventLoop

    application_loop = _ApplicationEventLoop()

    async def pending():
        return None

    coroutine = pending()
    application_loop._accepting = False
    try:
        try:
            application_loop.run(coroutine)
        except RuntimeError as error:
            assert "shutting down" in str(error)
        else:
            raise AssertionError("shutting down loop accepted a coroutine")
        assert coroutine.cr_frame is None
    finally:
        application_loop.loop.call_soon_threadsafe(application_loop.loop.stop)
        application_loop._thread.join()


def test_default_resource_cleanup_closes_sync_client_after_async_error(monkeypatch):
    import agentic.workflow as workflow_module

    class AsyncClient:
        async def close(self):
            raise ValueError("async close failed")

    class SyncClient:
        closed = False

        def close(self):
            self.closed = True

    sync_client = SyncClient()
    model = type("DefaultModelResources", (), {
        "root_async_client": AsyncClient(), "root_client": sync_client,
    })()
    monkeypatch.setattr(workflow_module, "_default_model", model)

    try:
        asyncio.run(workflow_module._close_default_resources())
    except ValueError as error:
        assert "async close failed" in str(error)
    else:
        raise AssertionError("async close error was swallowed")

    assert sync_client.closed


def test_run_and_reload_ticket_with_injected_dependencies(tmp_path):
    from agentic.workflow import WorkflowDependencies, arun_ticket, build_workflow

    @tool
    async def kb_search(query: str, limit: int) -> list[dict]:
        """Return known test evidence for a reservation question."""
        return [{
            "article_id": "test-article", "title": "Reserve experience",
            "content": "Reserve an experience through the app.", "score": 3,
        }]

    async def kb_tools():
        return [kb_search]

    ticket = {
        "ticket_id": "isolated-ticket", "content": "Reserve experience",
        "owner_id": "test-customer", "channel": "chat",
    }
    responses = [
        '{"summary":"Reserve experience","normalized_issue":"Reserve experience"}',
        '{"issue_type":"reservation","urgency":"low","complexity":"low"}',
        '{"next_step":"resolver","reason":"answer the question"}',
        'Use the app to reserve an experience.',
        '{"next_step":"done","reason":"resolved"}',
    ]
    database = str(tmp_path / "workflow.sqlite")

    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            dependencies = WorkflowDependencies(
                model=TicketModel(responses=responses), kb_tools=kb_tools,
            )
            workflow = build_workflow(dependencies=dependencies, checkpointer=saver)
            result = await arun_ticket(ticket, "isolated-ticket", workflow=workflow)
            assert result["resolution"]["status"] == "resolved"
            assert result["resolution"]["answer"] == "Use the app to reserve an experience."
            assert result["resolution"]["used_kb_articles"] == ["test-article"]
            assert result["supervisor"]["next_step"] == "done"

    asyncio.run(exercise())


class ClarificationState(TypedDict, total=False):
    ticket: dict
    reply: str


def test_paused_ticket_resumes_from_sqlite_with_new_workflow_instance(tmp_path):
    from agentic.workflow import arun_ticket, aresume_ticket

    database = str(tmp_path / "paused-ticket.sqlite")
    ticket = {"ticket_id": "paused-ticket", "content": "Please clarify"}

    def ask_customer(state: ClarificationState):
        reply = interrupt({"prompt": state["ticket"]["content"]})
        return {"reply": reply}

    async def make_workflow(saver):
        builder = StateGraph(ClarificationState)
        builder.add_node("ask_customer", ask_customer)
        builder.add_edge(START, "ask_customer")
        builder.add_edge("ask_customer", END)
        return builder.compile(checkpointer=saver)

    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            workflow = await make_workflow(saver)
            paused = await arun_ticket(ticket, "paused-ticket", workflow=workflow)
            assert paused["ticket"] == ticket
            assert paused["__interrupt__"][0].value == {"prompt": "Please clarify"}

        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            workflow = await make_workflow(saver)
            resumed = await aresume_ticket(
                "paused-ticket", workflow=workflow, reply="Customer supplied details",
            )
            assert resumed["ticket"] == ticket
            assert resumed["reply"] == "Customer supplied details"

    asyncio.run(exercise())


def test_resume_rejects_a_ticket_without_a_pending_interrupt(tmp_path):
    from agentic.workflow import aresume_ticket

    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "completed-ticket.sqlite")
        ) as saver:
            builder = StateGraph(ClarificationState)
            builder.add_node("finish", lambda state: {"reply": "already complete"})
            builder.add_edge(START, "finish")
            builder.add_edge("finish", END)
            workflow = builder.compile(checkpointer=saver)
            await workflow.ainvoke(
                {"ticket": {"ticket_id": "completed-ticket"}},
                {"configurable": {"thread_id": "completed-ticket"}},
            )
            try:
                await aresume_ticket("completed-ticket", workflow=workflow, reply="late reply")
            except ValueError as error:
                assert "not waiting" in str(error)
            else:
                raise AssertionError("completed ticket accepted a resume reply")

    asyncio.run(exercise())
