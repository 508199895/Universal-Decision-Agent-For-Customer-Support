import asyncio

from agentic.lifespan.runtime import default_workflow_runtime


async def main():
    async with default_workflow_runtime() as (workflow, *_):
        print(workflow.get_graph().draw_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
