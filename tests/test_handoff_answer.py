import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from agentic.agents.supervisor.contract import SupervisorResponse
from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.orchestration.ticket.manager import TicketStatus
from agentic.orchestration.orchestration import TicketWorkflowContext, run_workflow
from agentic.orchestration.workflow.graph import _supervisor_node
from agentic.orchestration.workflow.handoff import HANDOFF_NOTICE
from agentic.orchestration.workflow.state import initial_state


def test_successful_handoff_uses_send_result_for_customer_answer():
    class HandoffSupervisor:
        async def ainvoke(self, _payload, config=None, context=None):
            return {
                "messages": [
                    ToolMessage(
                        name="request_human_handoff",
                        tool_call_id="handoff-call-1",
                        content=json.dumps({
                            "status": "sent",
                            "ticket_id": "ticket-1",
                            "message_id": "feishu-message-1",
                            "summary": "取消预约请求",
                            "description": "客户要求取消预约，尚未执行。",
                        }, ensure_ascii=False),
                    ),
                ],
                "structured_response": SupervisorResponse(
                    answer="CultPass 预约可在个人中心取消。",
                    requires_handoff=True,
                ),
            }

    state = initial_state(
        owner_id="external-user-1",
        owner_name="客户",
        message=HumanMessage(content="请帮我取消预订。", id="customer-1"),
    )
    result = asyncio.run(_supervisor_node(
        state,
        {},
        supervisor=HandoffSupervisor(),
        context=TicketRuntimeContext(ticket_id="ticket-1"),
    ))

    assert result["supervisor"]["intent"] == "escalation"
    assert result["supervisor"]["message"] == HANDOFF_NOTICE
    assert result["messages"][-1].content == HANDOFF_NOTICE


def test_final_supervisor_answer_is_persisted_as_last_ai_message(monkeypatch):
    answer = "已转人工，等待人工客服回应"

    class FakeWorkflow:
        def __init__(self):
            self.snapshots = iter([
                SimpleNamespace(values={}),
                SimpleNamespace(values={
                    "supervisor": {"message": answer},
                }),
            ])

        async def aget_state(self, _config):
            return next(self.snapshots)

        async def aupdate_state(self, *_args, **_kwargs):
            return None

        async def astream(self, *_args, **_kwargs):
            yield {"supervisor": {"supervisor": {"message": answer}}}

    class FakeManager:
        def __init__(self, record):
            self.record = record
            self.completion = None

        async def set_status(self, _ticket_id, status, **_kwargs):
            self.record.status = TicketStatus(status)

        async def update_run_completion(self, _ticket_id, **completion):
            self.completion = completion

    class FakeEventStore:
        async def put_batch(self, _events):
            return None

    monkeypatch.setattr(
        "agentic.orchestration.orchestration.build_trace_config",
        lambda _ticket_id: None,
    )
    record = SimpleNamespace(
        ticket_id="ticket-1",
        user_id="user-1",
        status=TicketStatus.pending,
    )
    manager = FakeManager(record)
    context = TicketWorkflowContext(
        user_id="user-1",
        external_user_id="external-user-1",
        user_name=None,
        ticket_event_store=FakeEventStore(),
    )
    workflow = FakeWorkflow()

    result = asyncio.run(run_workflow(
        manager,
        record,
        ctx=context,
        message=HumanMessage(content="请帮我取消预订。", id="customer-1"),
        workflow=workflow,
    ))

    assert result["supervisor"]["message"] == answer
    assert manager.completion["last_ai_message"] == answer
    assert record.status == TicketStatus.success
