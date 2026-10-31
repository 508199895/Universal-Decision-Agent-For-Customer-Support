import sqlite3

import pytest
from typing_extensions import is_typeddict
from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agentic.data.tickets.TicketStore import TicketStore
from agentic.data.tickets.errors import (
    TicketHistoryConflict,
    TicketPersistenceError,
)
from agentic.orchestration.workflow.state import (
    StateContractError,
    WorkflowState,
    initial_state,
    merge_patch,
)


def test_ticket_schema_includes_cumulative_model_usage_fields(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketStore(database)

    connection = sqlite3.connect(database)
    ticket_columns = [row[1] for row in connection.execute("PRAGMA table_info(tickets)")]
    event_columns = [row[1] for row in connection.execute("PRAGMA table_info(ticket_events)")]
    connection.close()

    assert ticket_columns == [
        "ticket_id", "user_id", "status", "tokens", "input_tokens", "output_tokens",
        "llm_call_count", "created_at", "updated_at",
    ]
    assert event_columns == [
        "event_id", "user_id", "ticket_id", "seq", "event_type", "category",
        "content", "metadata", "created_at",
    ]
    assert store.thread_id("user-1") == "user-1"


def test_ticket_creation_persists_first_message_atomically_and_allows_multiple_active_tickets(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    first = store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", status="pending",
        initial_message=HumanMessage(content="first", id="message-1").model_dump(),
        initial_metadata={"owner_id": "external-1", "owner_name": "One"},
    )
    second = store.create_ticket(
        ticket_id="ticket-2", user_id="user-1", status="pending",
        initial_message=HumanMessage(content="second", id="message-2").model_dump(),
    )

    assert first.created and second.created
    assert first.status == second.status == "pending"
    assert [event["seq"] for event in store.event_store.events(user_id="user-1")] == [1, 2, 3, 4]
    assert [event["event_type"] for event in store.event_store.events(user_id="user-1", ticket_id="ticket-1")] == [
        "ticket.created", "ticket.message.received",
    ]
    first_message = store.event_store.events(user_id="user-1", ticket_id="ticket-1")[1]
    assert first_message["content"]["id"] == "message-1"
    assert first_message["metadata"]["owner_id"] == "external-1"


def test_checkpoint_initialization_marker_is_idempotent_and_scoped_to_pending_tickets(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", initial_message={"content": "hello", "id": "m1"},
    )

    assert store.latest_uninitialized_ticket("user-1").ticket_id == "ticket-1"
    store.mark_checkpoint_initialized("ticket-1")
    store.mark_checkpoint_initialized("ticket-1")
    assert store.checkpoint_initialized("ticket-1")
    assert store.latest_uninitialized_ticket("user-1") is None
    assert [event["event_type"] for event in store.event_store.events(user_id="user-1")] == [
        "ticket.created", "ticket.message.received", "checkpoint.initialized",
    ]


def test_event_sequence_is_user_scoped_and_event_owner_is_checked(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1")
    store.create_ticket(ticket_id="ticket-2", user_id="user-2")
    store.event_store.append_events([{
        "user_id": "user-1", "ticket_id": "ticket-1", "event_type": "run.start",
        "category": "trace", "content": {}, "metadata": {}, "created_at": "2026-10-22T00:00:00Z",
    }])

    assert [item["seq"] for item in store.event_store.events(user_id="user-1")] == [1, 2]
    assert [item["seq"] for item in store.event_store.events(user_id="user-2")] == [1]
    with pytest.raises(TicketHistoryConflict, match="does not belong"):
        store.event_store.append_events([{
            "user_id": "user-2", "ticket_id": "ticket-1", "event_type": "run.start",
            "category": "trace", "content": {}, "metadata": {}, "created_at": "2026-10-22T00:00:01Z",
        }])


def test_ticket_status_and_token_updates_follow_the_new_contract(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1", status="running")
    store.save_token_usage(
        "ticket-1", input_tokens=12, output_tokens=8, total_tokens=20, llm_call_count=2,
    )
    record = store.get_ticket("ticket-1")
    assert (record.tokens, record.input_tokens, record.output_tokens, record.llm_call_count) == (20, 12, 8, 2)

    assert store.commit_completed("ticket-1", final_message="处理完成。")
    assert not store.commit_completed("ticket-1", final_message="处理完成。")
    assert store.status("ticket-1") == "completed"


def test_terminal_status_and_customer_notice_commit_in_one_transaction(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketStore(database)
    store.create_ticket(ticket_id="ticket-1", user_id="user-1", status="running")

    assert store.commit_completed("ticket-1", final_message="处理已完成。")
    assert store.status("ticket-1") == "completed"
    terminal_events = [
        event for event in store.event_store.events(user_id="user-1", ticket_id="ticket-1")
        if event["event_type"] == "ticket.completed"
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0]["content"] == {"message": "处理已完成。"}
    assert not store.commit_completed("ticket-1", final_message="处理已完成。")
    assert len([
        event for event in store.event_store.events(user_id="user-1", ticket_id="ticket-1")
        if event["event_type"] == "ticket.completed"
    ]) == 1


def test_terminal_event_failure_rolls_back_completed_status(tmp_path):
    database = tmp_path / "agent.sqlite"
    store = TicketStore(database)
    store.create_ticket(ticket_id="ticket-1", user_id="user-1", status="running")
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TRIGGER reject_terminal_event BEFORE INSERT ON ticket_events
           WHEN NEW.event_type = 'ticket.completed'
           BEGIN SELECT RAISE(ABORT, 'terminal event rejected'); END"""
    )
    connection.commit()
    connection.close()

    with pytest.raises(TicketPersistenceError, match="terminal status persistence failed"):
        store.commit_completed("ticket-1", final_message="处理已完成。")

    assert store.status("ticket-1") == "running"
    assert not [
        event for event in store.event_store.events(user_id="user-1", ticket_id="ticket-1")
        if event["event_type"] == "ticket.completed"
    ]


def test_completed_ticket_requires_customer_facing_message(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="ticket-1", user_id="user-1", status="running")

    with pytest.raises(TicketHistoryConflict, match="missing its customer-facing"):
        store.commit_completed("ticket-1")

    assert store.status("ticket-1") == "running"


def test_workflow_state_inherits_agent_state_and_keeps_messages_at_the_top_level():
    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )

    assert is_typeddict(WorkflowState)
    assert set(AgentState.__annotations__) <= set(WorkflowState.__annotations__)
    assert set(AgentState.__required_keys__) <= WorkflowState.__required_keys__
    assert "messages" in WorkflowState.__annotations__
    assert {
        "ticket", "supervisor", "query", "clarify", "confirm", "account",
        "validate", "escalation",
    } <= WorkflowState.__required_keys__
    assert set(state) == {
        "messages", "ticket", "supervisor", "query", "clarify", "confirm", "account",
        "validate", "escalation",
    }
    assert "messages" not in state["ticket"]
    with pytest.raises(StateContractError, match="unknown fields"):
        merge_patch(state, {"ticket": {"unexpected": "value"}}, writer="boundary")
    with pytest.raises(StateContractError, match="cannot update group ticket"):
        merge_patch(state, {"ticket": {"owner_id": "attacker"}}, writer="supervisor")
    updated = merge_patch(state, {"query": {"evidence": []}}, writer="query")
    assert updated["query"]["evidence"] == []


def test_state_patch_retains_untouched_fields_and_rejects_invalid_values():
    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )
    state["supervisor"]["query_instruction"] = "prior instruction"
    patched = merge_patch(
        state,
        {"supervisor": {"query_instruction": None}, "query": {"evidence": []}},
        writer="boundary",
    )
    assert patched["ticket"]["input"] == "hello"
    assert patched["supervisor"]["query_instruction"] is None
    assert patched["query"]["evidence"] == []
    with pytest.raises(StateContractError, match="invalid supervisor.intent"):
        merge_patch(state, {"supervisor": {"intent": "DONE"}}, writer="supervisor")
    with pytest.raises(StateContractError, match="unknown state group"):
        merge_patch(state, {"budget": {"outer_nodes_used": 1}}, writer="boundary")


def test_compiled_subgraph_shares_parent_agent_state_messages():
    state = initial_state(
        ticket_id="ticket-1", owner_id="external-1", owner_name=None,
        message=HumanMessage(content="hello", id="message-1"),
    )

    def child_node(_state):
        return {"messages": [AIMessage(content="reply", id="reply-1")]}

    child_builder = StateGraph(WorkflowState)
    child_builder.add_node("child", child_node)
    child_builder.add_edge(START, "child")
    child_builder.add_edge("child", END)

    parent_builder = StateGraph(WorkflowState)
    parent_builder.add_node("child_graph", child_builder.compile())
    parent_builder.add_edge(START, "child_graph")
    parent_builder.add_edge("child_graph", END)

    result = parent_builder.compile().invoke(state)
    assert [(message.id, message.type) for message in result["messages"]] == [
        ("message-1", "human"), ("reply-1", "ai"),
    ]
