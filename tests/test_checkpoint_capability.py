"""Dependency capability check, not acceptance of the V5 business workflow."""

import json
import subprocess
import sys


PROBE = '''
import asyncio
import json
import sys
from typing import TypedDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict, total=False):
    question: str
    reply: str

def clarify(state):
    return {"reply": interrupt(state["question"])}

async def main():
    async with AsyncSqliteSaver.from_conn_string(sys.argv[1]) as saver:
        builder = StateGraph(State)
        builder.add_node("clarify", clarify)
        builder.add_edge(START, "clarify")
        builder.add_edge("clarify", END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "capability-probe"}}
        payload = {"question": "请补充问题"} if sys.argv[2] == "start" else Command(resume="客户补充原文")
        result = await graph.ainvoke(payload, config)
        print(json.dumps({"question": result["question"], "reply": result.get("reply"), "paused": bool(result.get("__interrupt__"))}))

asyncio.run(main())
'''


def test_sqlite_interrupt_resumes_in_an_independent_process(tmp_path):
    database = str(tmp_path / "checkpoints.sqlite")

    def run(phase):
        process = subprocess.run(
            [sys.executable, "-c", PROBE, database, phase],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    assert run("start") == {
        "question": "请补充问题", "reply": None, "paused": True,
    }
    assert run("resume") == {
        "question": "请补充问题", "reply": "客户补充原文", "paused": False,
    }
