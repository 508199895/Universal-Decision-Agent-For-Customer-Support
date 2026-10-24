import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
import pytest

from agentic.data.tickets.ticket_store import TicketEventStore, TicketPersistenceError
from agentic.orchestration.workflow import aenter_ticket, build_workflow
from agentic.orchestration.workflow_state import WorkflowState


class HandoffAgent:
    def __init__(self):
        self.payloads = []

    async def ainvoke(self, payload, config=None):
        self.payloads.append(payload)
        return {
            "summary": "客户请求取消预订，处理结果尚未确认。",
            "description": "对话记录了取消请求；当前无法确认取消是否提交成功。",
        }


async def _enter(database, store, agent, *, ticket_id, message, new_ticket=True, workflow=None,
                 support_agent=None):
    if workflow is not None:
        return await aenter_ticket(
            "external-1", message, user_id="user-1", user_name="One",
            workflow=workflow, ticket_event_store=store,
            ticket_id=ticket_id, new_ticket=new_ticket,
        )
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_workflow(
            checkpointer=checkpointer, ticket_event_store=store, agent=agent,
            support_agent=support_agent,
        )
        return await aenter_ticket(
            "external-1", message, user_id="user-1", user_name="One",
            workflow=graph, ticket_event_store=store,
            ticket_id=ticket_id, new_ticket=new_ticket,
        )


def test_handoff_uses_user_thread_and_shared_agent_state_messages(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)
    agent = HandoffAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    result = asyncio.run(_enter(
        database, store, agent, ticket_id="ticket-1", message=message,
    ))

    assert result["ticket"]["ticket_id"] == "ticket-1"
    assert result["ticket"]["status"] == "handed_off"
    assert "messages" not in result["ticket"]
    assert "[human] 请取消我的预订。" in agent.payloads[0]["conversation"]
    assert [(item.id, item.type) for item in result["messages"]] == [
        ("customer-1", "human"),
        ("escalation:ticket-1:result", "ai"),
        ("handoff:ticket-1:material", "system"),
        ("handoff:ticket-1:notice", "system"),
    ]
    assert store.status("ticket-1") == "completed"
    events = store.events(user_id="user-1", ticket_id="ticket-1")
    assert [event["event_type"] for event in events][:3] == [
        "ticket.created", "ticket.message.received", "checkpoint.initialized",
    ]
    assert events[-1]["event_type"] == "run.end"

    async def read_checkpoint():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer, ticket_event_store=store, agent=agent,
            )
            return await graph.aget_state({"configurable": {"thread_id": "user-1"}})

    snapshot = asyncio.run(read_checkpoint())
    assert snapshot.values["ticket"]["ticket_id"] == "ticket-1"
    assert "messages" not in snapshot.values["ticket"]


def test_customer_support_agent_runs_before_handoff_and_shares_messages(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)
    handoff_agent = HandoffAgent()
    support_calls = []

    async def answer_with_tool_evidence(state):
        support_calls.append(list(state["messages"]))
        return {"messages": [AIMessage(content="已查询知识库", id="support-agent-result")]}

    support_builder = StateGraph(WorkflowState)
    support_builder.add_node("support", answer_with_tool_evidence)
    support_builder.add_edge(START, "support")
    support_builder.add_edge("support", END)
    support_agent = support_builder.compile()

    result = asyncio.run(_enter(
        database, store, handoff_agent, ticket_id="ticket-support",
        message=HumanMessage(content="怎么退订？", id="customer-support"),
        support_agent=support_agent,
    ))

    assert len(support_calls) == 1
    assert [message.id for message in support_calls[0]] == ["customer-support"]
    assert "[ai] 已查询知识库" in handoff_agent.payloads[0]["conversation"]
    assert [message.id for message in result["messages"]][:2] == [
        "customer-support", "support-agent-result",
    ]


def test_model_failure_keeps_ticket_checkpoint_and_retry_does_not_duplicate_message(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)

    class FailOnceAgent(HandoffAgent):
        attempts = 0

        async def ainvoke(self, payload, config=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary model failure")
            return await super().ainvoke(payload, config=config)

    agent = FailOnceAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    try:
        asyncio.run(_enter(database, store, agent, ticket_id="ticket-1", message=message))
    except RuntimeError as error:
        assert "temporary model failure" in str(error)
    else:
        raise AssertionError("the first model call should fail")

    assert store.status("ticket-1") == "failed"
    result = asyncio.run(_enter(
        database, store, agent, ticket_id="ticket-1", message=message, new_ticket=False,
    ))

    assert agent.attempts == 2
    assert result["ticket"]["status"] == "handed_off"
    assert store.status("ticket-1") == "completed"
    assert [item.id for item in result["messages"]].count("customer-1") == 1


def test_initial_checkpoint_interruption_is_rebuilt_from_ticket_events(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)
    agent = HandoffAgent()
    message = HumanMessage(content="请取消我的预订。", id="customer-1")

    class FailingCheckpointUpdate:
        def __init__(self, graph):
            self.graph = graph

        async def aget_state(self, config):
            return await self.graph.aget_state(config)

        async def aupdate_state(self, *_args, **_kwargs):
            raise OSError("checkpoint write failed")

    async def fail_initial_write():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer, ticket_event_store=store, agent=agent,
            )
            try:
                await aenter_ticket(
                    "external-1", message, user_id="user-1", user_name="One",
                    workflow=FailingCheckpointUpdate(graph), ticket_event_store=store,
                    ticket_id="ticket-1", new_ticket=True,
                )
            except OSError as error:
                assert "checkpoint write failed" in str(error)
            else:
                raise AssertionError("initial checkpoint write should fail")

    asyncio.run(fail_initial_write())
    assert store.status("ticket-1") == "pending"
    assert not store.checkpoint_initialized("ticket-1")

    result = asyncio.run(_enter(
        database, store, agent, ticket_id="ticket-1", message=message, new_ticket=False,
    ))

    assert result["ticket"]["ticket_id"] == "ticket-1"
    assert result["ticket"]["status"] == "handed_off"
    assert store.status("ticket-1") == "completed"
    assert [item.id for item in result["messages"]].count("customer-1") == 1


def test_initialized_ticket_without_checkpoint_reports_recovery_error(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)
    store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", status="running",
        initial_message=HumanMessage(content="already started", id="customer-1").model_dump(),
    )
    store.mark_checkpoint_initialized("ticket-1")
    agent = HandoffAgent()

    async def enter_without_checkpoint():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer, ticket_event_store=store, agent=agent,
            )
            with pytest.raises(TicketPersistenceError, match="checkpoint marked initialized but missing"):
                await aenter_ticket(
                    "external-1", HumanMessage(content="new input", id="customer-2"),
                    user_id="user-1", workflow=graph, ticket_event_store=store,
                )

    asyncio.run(enter_without_checkpoint())
    assert store.get_ticket("ticket-1").status == "running"
    assert agent.payloads == []


def test_new_ticket_replaces_previous_users_messages_in_checkpoint(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)
    agent = HandoffAgent()
    first = asyncio.run(_enter(
        database, store, agent, ticket_id="ticket-1",
        message=HumanMessage(content="first", id="message-1"),
    ))
    second = asyncio.run(_enter(
        database, store, agent, ticket_id="ticket-2",
        message=HumanMessage(content="second", id="message-2"), new_ticket=True,
    ))

    assert first["ticket"]["ticket_id"] == "ticket-1"
    assert second["ticket"]["ticket_id"] == "ticket-2"
    assert [message.id for message in second["messages"]] == [
        "message-2", "escalation:ticket-2:result",
        "handoff:ticket-2:material", "handoff:ticket-2:notice",
    ]
    assert store.get_ticket("ticket-1").status == "completed"
    assert store.get_ticket("ticket-2").status == "completed"
