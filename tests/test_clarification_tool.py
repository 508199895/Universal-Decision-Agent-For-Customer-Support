import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic.agents.supervisor.contract import SupervisorResponse
from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.capabilities.tools.clarification import ask_clarification
from agentic.orchestration.workflow.graph import _supervisor_node
from agentic.orchestration.workflow.state import initial_state


def test_ask_clarification_returns_the_customer_facing_question():
    result = asyncio.run(ask_clarification.ainvoke({
        "question": "请提供需要取消的预约编号。",
    }))

    assert json.loads(result) == {
        "status": "asked",
        "question": "请提供需要取消的预约编号。",
    }


def test_ask_clarification_rejects_an_empty_question():
    with pytest.raises(ValueError, match="non-empty string"):
        asyncio.run(ask_clarification.ainvoke({"question": "  "}))


def test_successful_clarification_finishes_ticket_with_the_question_as_answer():
    question = "请提供需要取消的预约编号。"

    class ClarifyingSupervisor:
        async def ainvoke(self, _payload, config=None, context=None):
            return {
                "messages": [
                    ToolMessage(
                        name="ask_clarification",
                        tool_call_id="clarification-call-1",
                        content=json.dumps({
                            "status": "asked",
                            "question": question,
                        }, ensure_ascii=False),
                    ),
                ],
                "structured_response": SupervisorResponse(
                    answer="已请求补充信息。",
                ),
            }

    state = initial_state(
        owner_id="external-user-1",
        owner_name="客户",
        message=HumanMessage(content="请帮我取消预约。", id="customer-1"),
    )
    result = asyncio.run(_supervisor_node(
        state,
        {},
        supervisor=ClarifyingSupervisor(),
        context=TicketRuntimeContext(ticket_id="ticket-1"),
    ))

    assert result["supervisor"]["next_node"] == "finish"
    assert result["supervisor"]["message"] == question
    assert result["messages"][-1].content == question
    assert result["messages"][-1].id == "supervisor:ticket-1:answer"
