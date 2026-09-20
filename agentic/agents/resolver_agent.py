from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agentic.tools.knowledge_client import get_kb_tools
from agentic.tools.account_client import get_account_tools
from agentic.tools.memory_client import get_memory_tools


def build_resolver_agent(*, model: BaseChatModel) -> Any:
    """
    Resolver Agent

    Purpose:
        - Use tools (KB search, account lookup, memory) to resolve tickets when possible.
        - Provide a clear answer referencing KB content.
        - Estimate confidence; if low, request escalation instead of guessing.
        - Optionally write a long-term memory when a resolution is found.

    Tools available:
        - kb_search / kb_get
        - account_get_user / account_get_user_reservations
        - memory_write / memory_search / memory_get_all

    Expected Input (state fragment):
        {
            "ticket": { ... },
            "intake": { ... },
            "classification": { ... }
        }

    Output (merged under 'resolution'):
        {
            "status": "resolved" | "needs_escalation",
            "answer": str,
            "confidence": float,  # 0.0 - 1.0
            "used_kb_articles": [article_id, ...],
            "notes_for_human": str
        }

    Note:
        The graph/workflow will be responsible for interpreting 'status' and possibly
        routing to the EscalationAgent if needed.
    """
    # Gather tools from MCP-backed clients
    kb_tools: List[BaseTool] = get_kb_tools()
    account_tools: List[BaseTool] = get_account_tools()
    memory_tools: List[BaseTool] = get_memory_tools()

    tools: List[BaseTool] = kb_tools + account_tools + memory_tools

    system_prompt = SystemMessage(
        content=(
            "You are the Resolver Agent for UDA-Hub, a decision system that handles CultPass tickets.\n"
            "You have access to tools that can:\n"
            "- Search and read knowledge base articles (kb_* tools)\n"
            "- Look up user accounts and reservations (account_* tools)\n"
            "- Store and search long-term memory about prior resolutions (memory_* tools)\n\n"
            "GENERAL BEHAVIOR:\n"
            "- Always call kb_search first to find relevant articles.\n"
            "- Base your answer strictly on KB content and tool outputs.\n"
            "- If you lack enough information or KB does not cover the issue, do NOT guess.\n"
            "  Instead, produce an answer that suggests escalation.\n\n"
            "OUTPUT FORMAT:\n"
            "At the end of your reasoning, respond with a JSON object only, with keys:\n"
            "{\n"
            '  "status": "resolved" | "needs_escalation",\n'
            '  "answer": str,\n'
            '  "confidence": float,  # between 0.0 and 1.0\n'
            '  "used_kb_articles": list[str],\n'
            '  "notes_for_human": str\n'
            "}\n"
            "Where:\n"
            "- 'answer' is what we would send back to the user.\n"
            "- 'notes_for_human' includes any internal notes for a human agent.\n"
            "- If you are not confident (confidence < 0.6) or KB is clearly insufficient, set\n"
            "  status='needs_escalation' and explain why in 'notes_for_human'.\n"
        )
    )

    # Use a small in-memory checkpointer just for this agent's internal ReAct reasoning
    checkpointer = MemorySaver()

    # We use a ReAct-style agent internally for the resolver.
    # The outer LangGraph workflow will control when to call this agent.
    resolver_agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    return resolver_agent
