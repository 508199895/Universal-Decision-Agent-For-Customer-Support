import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessageChunk, HumanMessage

from agentic.app.event_stream import TicketEventPublisher, UserEventHub, encode_sse
from agentic.app.fastapi_server import stream_customer_events
from agentic.data.tickets.user_context import require_current_user_id
from agentic.orchestration.ticket.manager import TicketRecord, TicketStatus
from agentic.orchestration.ticket.request import TicketStartBody
from agentic.orchestration import orchestration
from agentic.orchestration.orchestration import TicketWorkflowContext, run_workflow


def test_user_event_hub_only_delivers_to_subscribers_for_that_user():
    async def exercise():
        hub = UserEventHub()
        async with hub.subscribe("user-1") as first, hub.subscribe("user-2") as second:
            event = {"id": "1", "event": "message.accepted", "data": {"ticket_id": "t1"}}
            hub.publish("user-1", event)

            assert await asyncio.wait_for(first.get(), timeout=1) == event
            assert second.empty()

    asyncio.run(exercise())


def test_ticket_event_publisher_persists_before_broadcasting():
    calls = []

    class EventStore:
        async def put(self, **event):
            calls.append(("persist", require_current_user_id(method_name="test"), event))
            return {"seq": 7}

    class Hub:
        def publish(self, user_id, event):
            calls.append(("publish", user_id, event))

    async def exercise():
        publisher = TicketEventPublisher(EventStore(), Hub())
        event = await publisher.publish(
            user_id="user-1",
            ticket_id="ticket-1",
            event_type="message.accepted",
            category="outputs",
            data={"status": "pending"},
        )
        assert event == {
            "id": "7",
            "event": "message.accepted",
            "data": {"ticket_id": "ticket-1", "status": "pending"},
        }

    asyncio.run(exercise())

    assert [call[0] for call in calls] == ["persist", "publish"]
    assert calls[0][1] == "user-1"
    assert calls[1][1] == "user-1"


def test_ticket_event_publisher_broadcasts_live_event_when_persistence_fails():
    published = []

    class EventStore:
        async def put(self, **_event):
            raise RuntimeError("database unavailable")

    class Hub:
        def publish(self, user_id, event):
            published.append((user_id, event))

    async def exercise():
        publisher = TicketEventPublisher(EventStore(), Hub())
        event = await publisher.publish(
            user_id="user-1",
            ticket_id="ticket-1",
            event_type="assistant.message",
            category="message",
            data={"message": "hello"},
        )

        assert event == {
            "event": "assistant.message",
            "data": {"message": "hello", "ticket_id": "ticket-1"},
        }

    asyncio.run(exercise())
    assert published == [("user-1", {
        "event": "assistant.message",
        "data": {"message": "hello", "ticket_id": "ticket-1"},
    })]


def test_sse_encoding_includes_id_type_and_json_data():
    encoded = encode_sse({
        "id": "9",
        "event": "assistant.message",
        "data": {"ticket_id": "ticket-1", "message": "你好"},
    })

    assert encoded.startswith("id: 9\nevent: assistant.message\ndata: ")
    assert json.loads(encoded.split("data: ", 1)[1].split("\n", 1)[0]) == {
        "ticket_id": "ticket-1",
        "message": "你好",
    }
    assert encode_sse({
        "event": "assistant.message",
        "data": {"ticket_id": "ticket-2", "message": "暂未保存"},
    }).startswith("id:\nevent: assistant.message\ndata: ")


def test_sse_endpoint_validates_user_and_streams_only_user_events():
    async def exercise():
        hub = UserEventHub()

        class Session:
            async def scalar(self, _statement):
                return object()

        class SessionFactory:
            @asynccontextmanager
            async def __call__(self):
                yield Session()

        class Request:
            def __init__(self):
                self.app = SimpleNamespace(state=SimpleNamespace(
                    user_session_factory=SessionFactory(),
                    ticket_event_hub=hub,
                ))
                self.disconnect_checks = 0

            async def is_disconnected(self):
                self.disconnect_checks += 1
                return self.disconnect_checks > 1

        response = await stream_customer_events(Request(), user_id="user-1")
        body = response.body_iterator
        assert await anext(body) == ": connected\n\n"
        hub.publish("user-1", {
            "id": "3", "event": "message.accepted", "data": {"ticket_id": "ticket-1"},
        })
        assert await anext(body) == (
            "id: 3\nevent: message.accepted\n"
            'data: {"ticket_id":"ticket-1"}\n\n'
        )
        with pytest.raises(StopAsyncIteration):
            await anext(body)

    asyncio.run(exercise())


def test_sse_endpoint_rejects_unknown_user():
    async def exercise():
        class Session:
            async def scalar(self, _statement):
                return None

        class SessionFactory:
            @asynccontextmanager
            async def __call__(self):
                yield Session()

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            user_session_factory=SessionFactory(),
        )))
        with pytest.raises(HTTPException) as error:
            await stream_customer_events(request, user_id="unknown")
        assert error.value.status_code == 404

    asyncio.run(exercise())


def test_start_ticket_persists_acceptance_before_dispatching_workflow(monkeypatch):
    calls = []
    record = TicketRecord(ticket_id="ticket-1", user_id="user-1")

    class Manager:
        async def create_or_reject(self, **_kwargs):
            return record

        async def set_status(self, *_args, **_kwargs):
            pytest.fail("successful acceptance should not mark the ticket as failed")

    class Publisher:
        async def publish(self, **event):
            calls.append(("persist_and_publish", event))

    async def finish_workflow(*_args, **_kwargs):
        calls.append(("workflow", None))

    monkeypatch.setattr(orchestration, "run_workflow", finish_workflow)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            ticket_manager=Manager(),
            workflow=object(),
            ticket_store=object(),
            ticket_event_store=object(),
            ticket_event_publisher=Publisher(),
        )),
        state=SimpleNamespace(user={
            "external_user_id": "external-1",
            "user_name": "Customer",
        }),
    )

    async def exercise():
        started = await orchestration.start_ticket(
            TicketStartBody(input={"messages": [{"role": "user", "content": "hello"}]}),
            "user-1",
            request,
        )
        assert calls[0][0] == "persist_and_publish"
        assert calls[0][1] == {
            "user_id": "user-1",
            "ticket_id": "ticket-1",
            "event_type": "message.accepted",
            "category": "outputs",
            "data": {"status": "pending"},
        }
        await started.task
        assert calls[1][0] == "workflow"

    asyncio.run(exercise())


def test_workflow_publishes_customer_reply_before_success_status(monkeypatch):
    calls = []

    class Workflow:
        def __init__(self):
            self.snapshots = iter([
                SimpleNamespace(values={}),
                SimpleNamespace(values={"supervisor": {"message": None}}),
            ])

        async def aget_state(self, _config):
            return next(self.snapshots)

        async def aupdate_state(self, *_args, **_kwargs):
            return None

        async def astream(self, *_args, **kwargs):
            assert kwargs["stream_mode"] == ["messages", "custom"]
            assert not kwargs.get("subgraphs", False)
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="中间内容", id="internal-1"), {"langgraph_node": "model"}),
            }
            yield {
                "type": "messages",
                "data": (
                    AIMessageChunk(
                        content="内部工具参数",
                        id="tool-1",
                        tool_calls=[{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}],
                    ),
                    {"langgraph_node": "model"},
                ),
            }
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="工具节点结果", id="tools-1"), {"langgraph_node": "tools"}),
            }
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="客户答复", id="final-1"), {"langgraph_node": "model"}),
            }

    class Manager:
        async def set_status(self, _ticket_id, status, **_kwargs):
            record.status = TicketStatus(status)
            calls.append(("status", status))

        async def update_run_completion(self, _ticket_id, **_kwargs):
            return None

    class EventStore:
        async def put_batch(self, _events):
            return []

    class Publisher:
        async def publish(self, **event):
            calls.append(("event", event))

    monkeypatch.setattr(
        "agentic.orchestration.orchestration.build_trace_config",
        lambda _ticket_id: None,
    )
    record = TicketRecord(ticket_id="ticket-1", user_id="user-1")
    context = TicketWorkflowContext(
        user_id="user-1",
        external_user_id="external-1",
        user_name=None,
        ticket_event_store=EventStore(),
    )
    workflow = Workflow()
    event_publisher = Publisher()

    async def exercise():
        result = await run_workflow(
            Manager(),
            record,
            ctx=context,
            message=HumanMessage(content="问题", id="customer-message"),
            workflow=workflow,
            event_publisher=event_publisher,
        )
        assert result["supervisor"]["message"] == "客户答复"

    asyncio.run(exercise())

    assert [call[0] for call in calls] == ["status", "event", "event", "status", "event"]
    assert calls[1][1]["event_type"] == "ticket.status"
    assert calls[1][1]["data"] == {"status": "running"}
    assert calls[2][1]["event_type"] == "assistant.message"
    assert calls[2][1]["category"] == "message"
    assert calls[2][1]["data"] == {"message": "客户答复"}
    assert calls[3][1] == "success"
    assert calls[4][1]["event_type"] == "ticket.status"
    assert calls[4][1]["data"] == {"status": "success"}
