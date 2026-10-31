from agentic.orchestration.workflow.factory import build_workflow
from langgraph.checkpoint.memory import MemorySaver


class DiagramAgent:
    async def ainvoke(self, _payload):
        return {"summary": "", "description": ""}

if __name__ == "__main__":
    graph = build_workflow(
        checkpointer=MemorySaver(),
        escalation_agent=DiagramAgent(),
    ).get_graph()

    mermaid = graph.draw_mermaid()

    print(mermaid)
