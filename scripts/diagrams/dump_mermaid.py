from agentic.orchestration.workflow import build_workflow
from langgraph.checkpoint.memory import MemorySaver
from agentic.data.tickets.ticket_store import TicketEventStore


class DiagramAgent:
    async def ainvoke(self, _payload):
        return {"summary": "", "description": ""}

if __name__ == "__main__":
    graph = build_workflow(
        checkpointer=MemorySaver(),
        ticket_event_store=TicketEventStore(":memory:"),
        agent=DiagramAgent(),
    ).get_graph()

    mermaid = graph.draw_mermaid()

    print(mermaid)
