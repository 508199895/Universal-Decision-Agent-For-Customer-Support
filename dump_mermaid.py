from agentic.workflow import build_workflow

if __name__ == "__main__":
    graph = build_workflow().get_graph()

    mermaid = graph.draw_mermaid()

    print(mermaid)
