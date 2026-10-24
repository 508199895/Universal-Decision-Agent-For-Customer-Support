import asyncio
import json
from agentic.capabilities.tools.knowledge_client import aget_kb_tools

async def main():
    tools = await aget_kb_tools()
    print("Available KB tools:")
    for t in tools:
        print("  -", t.name)

    # Pick the *actual* search tool by name fragment
    kb_search_tool = None
    for t in tools:
        if "kb_search" in t.name:
            kb_search_tool = t
            break

    if kb_search_tool is None:
        raise RuntimeError("No tool with 'kb_search' in its name found!")

    query = "How do I reserve a spot for a CultPass event?"
    print("\nCalling:", kb_search_tool.name)
    results = await kb_search_tool.ainvoke({"query": query, "limit": 5})

    print("RAW RESULT TYPE:", type(results))
    print("RAW RESULT:", results)

    # Normalize the result the same way the resolver does
    if isinstance(results, str):
        s = results.strip()
        if not s:
            print("\n[DEBUG] KB returned an empty string → no results.")
            return
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            print("\n[DEBUG] KB returned a non-JSON string:", s)
            return
    else:
        parsed = results

    print("\nParsed results (len={}):".format(len(parsed)))
    for i, r in enumerate(parsed):
        print(f"--- result {i} ---")
        print(r)

asyncio.run(main())
