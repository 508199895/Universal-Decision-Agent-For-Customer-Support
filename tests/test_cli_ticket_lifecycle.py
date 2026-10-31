import asyncio

import pytest

from agentic.app.cli import (
    _continuation_ticket_id,
    _should_start_new_ticket,
)
from agentic.data.tickets.TicketStore import TicketStore
from agentic.agents.escalation.handoff import HANDOFF_NOTICE
from agentic.orchestration.orchestration import customer_message_for_state


def test_cli_selects_only_unfinished_ticket_candidates(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    assert _continuation_ticket_id(store, "user-1") is None

    store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", initial_message={"content": "start", "id": "m1"},
    )
    assert _continuation_ticket_id(store, "user-1") == "ticket-1"

    store.mark_checkpoint_initialized("ticket-1")
    store.set_status("ticket-1", "running")
    assert _continuation_ticket_id(store, "user-1") == "ticket-1"

    store.commit_completed("ticket-1", final_message="done")
    assert _continuation_ticket_id(store, "user-1") is None


def test_cli_uses_the_latest_checkpoint_ticket_not_an_older_open_ticket(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="older", user_id="user-1", status="running")
    store.mark_checkpoint_initialized("older")
    store.create_ticket(ticket_id="latest", user_id="user-1", status="running")
    store.mark_checkpoint_initialized("latest")
    store.commit_completed("latest", final_message="done")

    assert store.status("older") == "running"
    assert _continuation_ticket_id(store, "user-1") is None


def test_cli_falls_back_to_pending_when_latest_checkpoint_ticket_is_completed(tmp_path):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(ticket_id="checkpoint", user_id="user-1", status="running")
    store.mark_checkpoint_initialized("checkpoint")
    store.commit_completed("checkpoint", final_message="done")
    store.create_ticket(ticket_id="pending", user_id="user-1", status="pending")

    assert _continuation_ticket_id(store, "user-1") == "pending"


def test_chat_starts_new_ticket_without_prompt_when_none_is_unfinished(tmp_path, monkeypatch):
    store = TicketStore(tmp_path / "agent.sqlite")

    def unexpected_prompt(_prompt):
        raise AssertionError("CLI should not ask when no unfinished ticket exists")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    assert asyncio.run(_should_start_new_ticket(store, "user-1")) == (True, None)


@pytest.mark.parametrize(("answer", "expected"), [("是", (False, "ticket-1")), ("否", (True, None))])
def test_chat_uses_the_fixed_prompt_to_choose_ticket_lifecycle(
    tmp_path, monkeypatch, answer, expected,
):
    store = TicketStore(tmp_path / "agent.sqlite")
    store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", initial_message={"content": "start", "id": "m1"},
    )
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or answer)

    assert asyncio.run(_should_start_new_ticket(store, "user-1")) == expected
    assert prompts == ["检测到有未完成工单，是否继续？"]


def test_cli_projects_only_the_active_customer_message():
    assert customer_message_for_state({
        "ticket": {"status": "waiting_reply"},
        "confirm": {"prompt": "确认取消预约？"},
        "clarify": {"prompt": "请提供预约编号。"},
    }) == "确认取消预约？"
    assert customer_message_for_state({
        "ticket": {"status": "waiting_reply"},
        "confirm": {"prompt": None},
        "clarify": {"prompt": "请提供预约编号。"},
    }) == "请提供预约编号。"
    assert customer_message_for_state({
        "ticket": {"status": "waiting_reply"},
        "confirm": {"prompt": None},
        "clarify": {"prompt": None},
        "supervisor": {"message": "此前的已完成答案。"},
    }) is None
    assert customer_message_for_state({
        "ticket": {"status": "handed_off"},
    }) == HANDOFF_NOTICE
    assert customer_message_for_state({
        "ticket": {"status": "completed"},
        "query": {"answer": "已验证答案。", "validated": True},
        "supervisor": {"message": None},
    }) is None
    assert customer_message_for_state({
        "ticket": {"status": "completed"},
        "query": {"answer": "已验证答案。", "validated": True},
        "supervisor": {"message": "supervisor 组织后的答案。"},
    }) == "supervisor 组织后的答案。"
    assert customer_message_for_state({
        "ticket": {"status": "completed"},
        "query": {"answer": "未校核内容。", "validated": False},
    }) is None
