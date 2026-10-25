import pytest

from agentic.app.cli import _parse_chat_input
from agentic.app.cli import _has_unfinished_ticket
from agentic.data.tickets.ticket_store import TicketEventStore
from agentic.orchestration.handoff import HANDOFF_NOTICE
from agentic.orchestration.workflow import customer_message_for_state


def test_cli_defaults_to_a_new_ticket_and_allows_explicit_continuation():
    assert _parse_chat_input("查询我的预约") == ("查询我的预约", True)
    assert _parse_chat_input(":new 再问一个问题") == ("再问一个问题", True)
    assert _parse_chat_input(":continue 补充一下") == ("补充一下", False)


@pytest.mark.parametrize("value", [":new", ":continue", ":unknown hi"])
def test_cli_rejects_empty_or_unknown_lifecycle_commands(value):
    with pytest.raises(ValueError):
        _parse_chat_input(value)


def test_cli_only_continues_when_an_unfinished_ticket_exists(tmp_path):
    store = TicketEventStore(tmp_path / "agent.sqlite")
    assert not _has_unfinished_ticket(store, "user-1")

    store.create_ticket(
        ticket_id="ticket-1", user_id="user-1", initial_message={"content": "start", "id": "m1"},
    )
    assert _has_unfinished_ticket(store, "user-1")

    store.mark_checkpoint_initialized("ticket-1")
    store.set_status("ticket-1", "running")
    assert _has_unfinished_ticket(store, "user-1")

    store.commit_completed("ticket-1", final_message="done")
    assert not _has_unfinished_ticket(store, "user-1")


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
    }) == "已验证答案。"
    assert customer_message_for_state({
        "ticket": {"status": "completed"},
        "query": {"answer": "未校核内容。", "validated": False},
    }) is None
