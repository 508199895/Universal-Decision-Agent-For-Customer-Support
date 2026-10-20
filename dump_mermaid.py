from agentic.workflow import build_workflow
from langgraph.checkpoint.memory import MemorySaver
from agentic.ticket_store import TicketStore


class DiagramAgent:
    async def ainvoke(self, _payload):
        return {"summary": "", "description": ""}

if __name__ == "__main__":
    graph = build_workflow(
        checkpointer=MemorySaver(),
        ticket_store=TicketStore(":memory:"),
        agent=DiagramAgent(),
    ).get_graph()

    mermaid = graph.draw_mermaid()

    print(mermaid)
