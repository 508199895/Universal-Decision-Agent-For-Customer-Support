import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic.data.tickets.ticket_store import TicketEventStore
from agentic.agents.agent import AgentSet
from agentic.orchestration.handoff import HANDOFF_NOTICE
from agentic.orchestration.workflow import (
    ValidateResult,
    SupervisorIntentResult,
    _public_evidence,
    aenter_ticket,
    build_workflow,
    customer_message_for_state,
)
from agentic.orchestration.workflow_state import WorkflowState


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


def _support_agent(answer):
    async def respond(_state):
        return {"messages": [AIMessage(content=answer, id="candidate-draft")]}

    builder = StateGraph(WorkflowState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def _run(tmp_path, model, *, draft="候选草稿。", question="退订规则是什么？"):
    database = tmp_path / "agent.sqlite"
    store = TicketEventStore(database)

    async def execute():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
            graph = build_workflow(
                checkpointer=checkpointer,
                ticket_event_store=store,
                agents=AgentSet(
                    escalation=_HandoffAgent(),
                    query=_support_agent(draft),
                    supervisor=_StructuredAgent(model, SupervisorIntentResult),
                    validate=_StructuredAgent(model, ValidateResult),
                ),
            )
            result = await aenter_ticket(
                "external-1", HumanMessage(content=question, id="customer-1"),
                user_id="user-1", workflow=graph, ticket_event_store=store,
                ticket_id="ticket-1", new_ticket=True,
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
    events = store.events(user_id="user-1", ticket_id="ticket-1")
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
