"""Exercise RAG, account, clarification, and handoff scenarios through /tickets."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic.app import fastapi_server
from agentic.data.tickets.ticket_journal import TicketJournal
from agentic.lifespan import runtime


SCENARIOS = (
    {
        "name": "rag",
        "user_index": 0,
        "message": "CultPass 应用中如何修改邮箱？",
        "tool": "query_public_knowledge",
        "workflow_status": "completed",
    },
    {
        "name": "account",
        "user_index": 1,
        "message": "请查询我的账户资料和预约情况。",
        "tool": "lookup_current_user_account",
        "workflow_status": "completed",
    },
    {
        "name": "handoff",
        "user_index": 2,
        "message": "请帮我取消我的预订。",
        "tool": "request_human_handoff",
        "workflow_status": "handed_off",
    },
    {
        "name": "clarification",
        "user_index": 0,
        "message": "我想查询一下相关信息。",
        "tool": "ask_clarification",
        "workflow_status": "completed",
    },
)


def _test_user_ids() -> list[str]:
    database = Path(os.getenv(
        "UDAHUB_DB_PATH", PROJECT_ROOT / "data" / "core" / "udahub.db",
    )).expanduser()
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT user_id FROM users WHERE external_user_id IS NOT NULL "
            "ORDER BY user_id LIMIT 3"
        ).fetchall()
    required_users = max(scenario["user_index"] for scenario in SCENARIOS) + 1
    if len(rows) < required_users:
        raise RuntimeError(
            f"Need {required_users} existing users with linked external accounts"
        )
    return [row[0] for row in rows]


async def run() -> None:
    users = _test_user_ids()
    delivered: list[str] = []
    tasks: dict[str, asyncio.Task] = {}
    observed_tools: dict[str, set[str]] = {}
    tool_run_names: dict[str, tuple[str, str]] = {}
    tool_outputs: dict[str, dict[str, list[Any]]] = {}
    original_database_factory = runtime._default_agent_database
    original_start_ticket = fastapi_server.start_ticket
    original_send = fastapi_server.send_feishu_text
    original_tool_start = TicketJournal.on_tool_start
    original_tool_end = TicketJournal.on_tool_end

    async def observe_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name")
        if isinstance(name, str) and self.ticket_id:
            observed_tools.setdefault(self.ticket_id, set()).add(name)
            run_id = kwargs.get("run_id")
            if run_id is not None:
                tool_run_names[str(run_id)] = (self.ticket_id, name)
        return await original_tool_start(self, serialized, input_str, **kwargs)

    async def capture_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        identity = tool_run_names.pop(str(run_id), None) if run_id is not None else None
        if identity is not None:
            ticket_id, name = identity
            tool_outputs.setdefault(ticket_id, {}).setdefault(name, []).append(output)
        return await original_tool_end(self, output, **kwargs)

    async def capture_start_ticket(*args: Any, **kwargs: Any):
        record = await original_start_ticket(*args, **kwargs)
        if record.task is not None:
            tasks[record.ticket_id] = record.task
        return record

    async def capture_feishu_delivery(channel, chat_id: str, message: str) -> str:
        message_id = await original_send(channel, chat_id, message)
        delivered.append(message_id)
        return message_id

    fastapi_server.start_ticket = capture_start_ticket
    fastapi_server.send_feishu_text = capture_feishu_delivery
    TicketJournal.on_tool_start = observe_tool_start
    TicketJournal.on_tool_end = capture_tool_end

    try:
        with tempfile.TemporaryDirectory(
            prefix="uda-fastapi-scenarios-",
            dir=PROJECT_ROOT / ".tmp",
        ) as temp_dir:
            ticket_database = Path(temp_dir) / "agent_tickets.db"
            runtime._default_agent_database = lambda: ticket_database

            async with fastapi_server.app.router.lifespan_context(fastapi_server.app):
                transport = httpx.ASGITransport(app=fastapi_server.app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://fastapi.local",
                    timeout=30,
                ) as client:
                    health = await client.get("/health")
                    health.raise_for_status()
                    if health.json() != {
                        "status": "ok",
                        "feishu_connected": True,
                        "workflow_ready": True,
                    }:
                        raise RuntimeError("FastAPI did not report a ready Feishu workflow")

                    selected = [
                        scenario for scenario in SCENARIOS
                        if not selected_names or scenario["name"] in selected_names
                    ]
                    for scenario in selected:
                        response = await client.post("/tickets", json={
                            "user_id": users[scenario["user_index"]],
                            "message": scenario["message"],
                        })
                        if response.status_code != 202:
                            raise RuntimeError(
                                f"{scenario['name']} POST /tickets returned "
                                f"HTTP {response.status_code}"
                            )
                        ticket_id = response.json()["ticket_id"]
                        task = tasks.get(ticket_id)
                        if task is None:
                            raise RuntimeError(f"{scenario['name']} task was not dispatched")
                        state = await asyncio.wait_for(task, timeout=300)
                        if state["ticket"]["status"] != scenario["workflow_status"]:
                            raise RuntimeError(
                                f"{scenario['name']} ended with workflow status "
                                f"{state['ticket']['status']}"
                            )

                        tool_names = observed_tools.get(ticket_id, set())
                        if scenario["tool"] not in tool_names:
                            raise RuntimeError(
                                f"{scenario['name']} did not record tool "
                                f"{scenario['tool']}; observed {sorted(tool_names)}"
                            )
                        if scenario["name"] == "rag" and not any(
                            name.startswith("qdrant_") for name in tool_names
                        ):
                            raise RuntimeError("RAG did not invoke a Qdrant search tool")
                        if scenario["name"] == "account" and not any(
                            name.startswith("account_account_get_user")
                            for name in tool_names
                        ):
                            raise RuntimeError("account agent did not invoke an account MCP tool")
                        tool_result = tool_outputs.get(ticket_id, {}).get(
                            scenario["tool"], [],
                        )
                        if not tool_result:
                            raise RuntimeError(
                                f"{scenario['name']} has no captured tool result"
                            )
                        if scenario["name"] in {"rag", "account"}:
                            parsed = tool_result[-1]
                            if hasattr(parsed, "content"):
                                parsed = parsed.content
                            if isinstance(parsed, str):
                                try:
                                    parsed = json.loads(parsed)
                                except json.JSONDecodeError:
                                    pass
                            if not isinstance(parsed, dict) or parsed.get("status") != "succeeded":
                                raise RuntimeError(
                                    f"{scenario['name']} tool did not report success"
                                )
                        if scenario["name"] == "clarification":
                            parsed = tool_result[-1]
                            if hasattr(parsed, "content"):
                                parsed = parsed.content
                            if isinstance(parsed, str):
                                try:
                                    parsed = json.loads(parsed)
                                except json.JSONDecodeError:
                                    pass
                            final_message = state.get("supervisor", {}).get("message")
                            if (
                                not isinstance(parsed, dict)
                                or parsed.get("status") != "asked"
                                or parsed.get("question") != final_message
                                or not isinstance(final_message, str)
                                or not final_message.strip()
                            ):
                                raise RuntimeError(
                                    "clarification question was not returned as the final message"
                                )
                        run_record = await fastapi_server.app.state.ticket_store.get(ticket_id)
                        if run_record is None or run_record["status"] != "success":
                            raise RuntimeError(f"{scenario['name']} ticket run did not succeed")
                        if scenario["name"] == "handoff" and len(delivered) != 1:
                            raise RuntimeError("handoff did not produce exactly one Feishu message")
                        print(
                            f"{scenario['name']}: HTTP 202; workflow="
                            f"{state['ticket']['status']}; tool={scenario['tool']}"
                        )
                        if scenario["name"] == "clarification":
                            print(f"clarification reply: {final_message}")

                    if "handoff" in selected_names and len(delivered) != 1:
                        raise RuntimeError("unexpected Feishu message count")
                    if delivered:
                        print(f"handoff Feishu message_id: {delivered[0]}")
    finally:
        runtime._default_agent_database = original_database_factory
        fastapi_server.start_ticket = original_start_ticket
        fastapi_server.send_feishu_text = original_send
        TicketJournal.on_tool_start = original_tool_start
        TicketJournal.on_tool_end = original_tool_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", action="append", choices=[item["name"] for item in SCENARIOS],
        help="scenario to run; repeat to select several (default: all three)",
    )
    selected_names = set(parser.parse_args().scenario or [])
    asyncio.run(run())
