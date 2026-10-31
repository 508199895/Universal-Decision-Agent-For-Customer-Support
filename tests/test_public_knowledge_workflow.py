import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic.data.tickets.TicketStore import TicketStore
from agentic.agents.escalation.handoff import HANDOFF_NOTICE
from agentic.orchestration.workflow.factory import (
    ValidateResult,
    SupervisorIntentResult,
    build_workflow,
)
from agentic.agents.agent_wrappers import _public_evidence
from agentic.orchestration.ticket.manager import TicketManager
from agentic.orchestration.orchestration import customer_message_for_state, run_ticket
from agentic.orchestration.workflow.state import WorkflowState


class _StructuredModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    async def ainvoke(self, _messages, config=None):
        return next(self.responses)


class _StructuredAgent:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema

    async def ainvoke(self, _state, config=None, context=None):
        return {"structured_response": await self.model.with_structured_output(self.schema).ainvoke([])}


class _HandoffAgent:
    async def ainvoke(self, _payload, config=None, context=None):
        return {
            "summary": "客户提出公共知识问题，自动校核未能确认答案。",
            "description": "候选答案未通过可靠性校核，未向客户交付候选内容。",
        }


def _support_agent(answer, tool_messages=()):
    async def respond(_state):
        return {"messages": [*tool_messages, AIMessage(content=answer, id="candidate-draft")]}

    builder = StateGraph(WorkflowState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def _run(
    tmp_path, model, *, draft="候选草稿。", question="退订规则是什么？",
    account_tool_messages=(),
):
    database = tmp_path / "agent.sqlite"
    store = TicketStore(database)

    async def execute():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer,
                escalation_agent=_HandoffAgent(),
                supervisor_agent=_StructuredAgent(model, SupervisorIntentResult),
            )
            result = await run_ticket(TicketManager(store),
                "external-1", HumanMessage(content=question, id="customer-1"),
                user_id="user-1", workflow=graph,     ticket_id="ticket-1", new_ticket=True,
            )
            return result

    return asyncio.run(execute()), store


def test_public_answer_is_committed_only_after_all_three_checks_pass(tmp_path):
    model = _StructuredModel([
        SupervisorIntentResult(is_public_knowledge=True),
        ValidateResult(
            context_relevance=True, faithfulness=True,
            answer_relevance=True, feedback="",
        ),
    ])

    state, store = _run(tmp_path, model, draft="已核实的退订规则。")

    assert model.schemas == [SupervisorIntentResult, ValidateResult]
    assert state["ticket"]["status"] == "completed"
    assert state["query"]["validated"] is True
    assert state["supervisor"]["message"] == "已核实的退订规则。"
    assert customer_message_for_state(state) == "已核实的退订规则。"
    assert store.status("ticket-1") == "completed"
    events = store.event_store.events(user_id="user-1", ticket_id="ticket-1")
    assert events[-1]["event_type"] == "ticket.completed"
    assert events[-1]["content"]["message"] == "已核实的退订规则。"


def test_public_answer_that_fails_review_is_handed_off_without_leaking_candidate(tmp_path):
    model = _StructuredModel([
        SupervisorIntentResult(is_public_knowledge=True),
        ValidateResult(
            context_relevance=True, faithfulness=False,
            answer_relevance=True, feedback="材料没有支持该承诺。",
        ),
    ])

    state, store = _run(tmp_path, model, draft="未经证实的退订承诺。")

    assert state["ticket"]["status"] == "handed_off"
    assert state["query"]["validated"] is False
    assert customer_message_for_state(state) == HANDOFF_NOTICE
    assert "未经证实的退订承诺" not in customer_message_for_state(state)
    assert store.status("ticket-1") == "completed"


def test_invalid_review_structure_hands_off(tmp_path):
    model = _StructuredModel([
        SupervisorIntentResult(is_public_knowledge=True),
        {"unexpected": "not an answer"},
    ])

    state, _store = _run(tmp_path, model)

    assert model.schemas == [SupervisorIntentResult, ValidateResult]
    assert state["ticket"]["status"] == "handed_off"
    assert state["validate"]["context_relevance"] is None
    assert state["query"]["validated"] is None
    assert customer_message_for_state(state) == HANDOFF_NOTICE


def test_account_action_intent_bypasses_public_answer_generation(tmp_path):
    model = _StructuredModel([SupervisorIntentResult(is_public_knowledge=False)])

    state, _store = _run(tmp_path, model, question="请取消我的预订")

    assert model.schemas == [SupervisorIntentResult]
    assert state["ticket"]["status"] == "handed_off"
    assert state["supervisor"]["intent"] == "escalation"
    assert customer_message_for_state(state) == HANDOFF_NOTICE


def _account_message(name, result, tool_call_id):
    return ToolMessage(
        name=name,
        content=json.dumps(result, ensure_ascii=False),
        tool_call_id=tool_call_id,
        id=f"result-{tool_call_id}",
    )


def test_account_profile_query_returns_source_backed_tool_evidence(tmp_path):
    model = _StructuredModel([
        SupervisorIntentResult(is_public_knowledge=False),
        ValidateResult(
            context_relevance=True, faithfulness=True,
            answer_relevance=True, feedback="",
        ),
    ])
    messages = [
        _account_message("account_account_get_user", {
            "external_user": {"user_id": "external-1", "name": "One"},
            "core_user": None, "reservation_count": 2, "ticket_count": 0,
        }, "profile"),
        _account_message("account_account_get_user_reservations", [
            {"reservation_id": "23e1c1", "user_id": "external-1", "status": "confirmed"},
        ], "reservations"),
    ]
    state, store = _run(
        tmp_path, model, question="请查询我的账户资料", account_tool_messages=messages,
        draft="账户资料显示姓名为 One。",
    )

    assert state["ticket"]["status"] == "completed"
    assert [item["source_ref"] for item in state["query"]["evidence"]] == [
        "account_get_user", "account_get_user_reservations",
    ]
    assert customer_message_for_state(state) == "账户资料显示姓名为 One。"
    assert store.event_store.events(user_id="user-1", ticket_id="ticket-1")[-1]["content"]["message"] == customer_message_for_state(state)


def test_account_query_failure_and_unknown_user_do_not_claim_success(tmp_path):
    (tmp_path / "failed").mkdir()
    (tmp_path / "unknown").mkdir()
    failed, _ = _run(
        tmp_path / "failed",
        _StructuredModel([
            SupervisorIntentResult(is_public_knowledge=False),
            ValidateResult(
                context_relevance=False, faithfulness=False,
                answer_relevance=False, feedback="账户查询未得到资料。",
            ),
        ]),
        question="我的预约状态是什么？",
        draft="无法从账户服务核实预约状态。",
    )
    assert failed["ticket"]["status"] == "handed_off"
    assert customer_message_for_state(failed) == HANDOFF_NOTICE

    unknown, _ = _run(
        tmp_path / "unknown",
        _StructuredModel([SupervisorIntentResult(is_public_knowledge=False)]),
        question="我的账户资料是什么？",
        account_tool_messages=[_account_message(
            "account_account_get_user", {"external_user": None, "core_user": None}, "unknown",
        )],
    )
    assert unknown["ticket"]["status"] == "handed_off"
    assert customer_message_for_state(unknown) == HANDOFF_NOTICE


def test_account_query_rejects_reservation_rows_for_another_user(tmp_path):
    state, _ = _run(
        tmp_path,
        _StructuredModel([SupervisorIntentResult(is_public_knowledge=False)]),
        question="我的预约状态是什么？",
        account_tool_messages=[_account_message(
            "account_account_get_user_reservations", [
                {"reservation_id": "23e1c1", "user_id": "external-2", "status": "secret"},
            ], "foreign-reservation",
        )],
    )
    assert state["ticket"]["status"] == "handed_off"
    assert "secret" not in customer_message_for_state(state)


def test_account_query_with_missing_requested_field_keeps_the_gap(tmp_path):
    state, _ = _run(
        tmp_path,
        _StructuredModel([
            SupervisorIntentResult(is_public_knowledge=False),
            ValidateResult(
                context_relevance=True, faithfulness=False,
                answer_relevance=True, feedback="账户资料未返回邮箱。",
            ),
        ]),
        question="我的邮箱是什么？",
        account_tool_messages=[_account_message(
            "account_account_get_user", {
                "external_user": {"user_id": "external-1", "name": "One", "email": None},
                "core_user": None,
            }, "profile-no-email",
        )],
        draft="账户资料未返回邮箱，尚不能核实。",
    )
    assert state["ticket"]["status"] == "handed_off"
    assert customer_message_for_state(state) == HANDOFF_NOTICE


def test_account_status_query_does_not_substitute_reservation_data(tmp_path):
    state, _ = _run(
        tmp_path,
        _StructuredModel([
            SupervisorIntentResult(is_public_knowledge=False),
            ValidateResult(
                context_relevance=True, faithfulness=False,
                answer_relevance=True, feedback="当前工具没有订阅状态资料。",
            ),
        ]),
        question="What is my subscription status?",
        draft="当前工具没有订阅状态资料，暂时无法核实。",
    )

    assert state["ticket"]["status"] == "handed_off"
    assert state["query"]["evidence"] == []
    assert customer_message_for_state(state) == HANDOFF_NOTICE


def test_qdrant_evidence_requires_source_reference_from_result_metadata():
    evidence = _public_evidence([
        ToolMessage(
            name="qdrant_qdrant-find", tool_call_id="call-id-is-not-a-source",
            content=[
                "Results for the query",
                '<entry><content>Cancellation rules.</content>'
                '<metadata>{"article_id":"policy-17"}</metadata></entry>',
                '<entry><content>No reference.</content><metadata>{}</metadata></entry>',
            ],
        ),
    ])

    assert evidence == [{
        "source_type": "public_knowledge",
        "source_ref": "policy-17",
        "content": "Cancellation rules.",
    }]


