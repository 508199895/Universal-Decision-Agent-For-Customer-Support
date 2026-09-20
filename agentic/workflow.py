from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Optional, TypedDict
import asyncio
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0"))

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is missing.")

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig

from agentic.agents.intake_agent import build_intake_agent
from agentic.agents.classifier_agent import build_classifier_agent
from agentic.agents.escalation_agent import build_escalation_agent
from agentic.agents.supervisor_agent import build_supervisor_agent
from agentic.tools.knowledge_client import (
    get_kb_search_tool,  
    aget_kb_tools,
)
from agentic.tools.account_client import (
    get_account_get_user_tool,
    get_account_get_user_reservations_tool,
)
from agentic.tools.memory_client import (
    get_memory_write_tool,
    get_memory_search_tool,
)

from logger import get_logger

log = get_logger()


# State schema

class TicketState(TypedDict, total=False):
    """
    Shared state for the UDA-Hub workflow.
    """

    ticket: Dict[str, Any]
    intake: Dict[str, Any]
    classification: Dict[str, Any]
    resolution: Dict[str, Any]
    escalation: Dict[str, Any]
    supervisor: Dict[str, Any]


# Global models & tools

LLM = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=DEEPSEEK_TEMPERATURE,
)

INTAKE_AGENT = build_intake_agent(model=LLM)
CLASSIFIER_AGENT = build_classifier_agent(model=LLM)
ESCALATION_AGENT = build_escalation_agent(model=LLM)
SUPERVISOR_AGENT = build_supervisor_agent(model=LLM)

#KB_SEARCH_TOOL = get_kb_search_tool()
ACCOUNT_GET_USER_TOOL = get_account_get_user_tool()
ACCOUNT_GET_RESERVATIONS_TOOL = get_account_get_user_reservations_tool()
MEMORY_WRITE_TOOL = get_memory_write_tool()
MEMORY_SEARCH_TOOL = get_memory_search_tool()


# helper to extract IDs for logging
def _ids_for_log(state: TicketState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    ticket_id = ticket.get("ticket_id")
    thread_id = None

    if config is not None:
        cfg = config.get("configurable", {}) or {}
        thread_id = cfg.get("thread_id")

    data: Dict[str, Any] = {}
    if ticket_id:
        data["ticket_id"] = ticket_id
    if thread_id:
        data["thread_id"] = thread_id
    return data


# Node implementations
def intake_node(state: TicketState, config: RunnableConfig) -> TicketState:
    """
    Normalize the ticket and extract a summary, sentiment, etc.
    """
    ids = _ids_for_log(state, config)
    log.info("node_start_intake", extra={"extra_data": ids})

    ticket = state["ticket"]
    result = INTAKE_AGENT.invoke(
        {
            "ticket_content": ticket.get("content", ""),
            "channel": ticket.get("channel", "unknown"),
            "tags": ticket.get("tags", ""),
            "owner_name": ticket.get("owner_name", ""),
        }
    )

    log.info(
        "node_end_intake",
        extra={
            "extra_data": {
                **ids,
                "summary": result.get("summary"),
                "sentiment": result.get("sentiment"),
                "suspected_language": result.get("suspected_language"),
            }
        },
    )
    return {"intake": result}


def classifier_node(state: TicketState, config: RunnableConfig) -> TicketState:
    """
    Classify issue type, urgency, complexity, and whether it likely needs escalation.
    """
    ids = _ids_for_log(state, config)
    log.info("node_start_classifier", extra={"extra_data": ids})

    ticket = state["ticket"]
    intake = state["intake"]

    result = CLASSIFIER_AGENT.invoke(
        {
            "ticket_content": ticket.get("content", ""),
            "normalized_issue": intake.get("normalized_issue", ""),
            "sentiment": intake.get("sentiment", "neutral"),
            "channel": ticket.get("channel", "unknown"),
            "tags": ticket.get("tags", ""),
        }
    )

    log.info(
        "node_end_classifier",
        extra={
            "extra_data": {
                **ids,
                "issue_type": result.get("issue_type"),
                "urgency": result.get("urgency"),
                "complexity": result.get("complexity"),
                "should_escalate_immediately": result.get("should_escalate_immediately"),
            }
        },
    )
    return {"classification": result}

async def resolver_node(state: TicketState, config: RunnableConfig) -> TicketState:
    """
    Resolver node (agent):

    - Uses MCP-backed tools:
        * KB_SEARCH_TOOL: search relevant KB articles
        * ACCOUNT_GET_USER_TOOL / ACCOUNT_GET_RESERVATIONS_TOOL: personalization
        * MEMORY_SEARCH_TOOL: recall prior interactions
        * MEMORY_WRITE_TOOL: store successful resolutions
    - Uses LLM to craft final answer grounded in KB + tools.
    - Sets a confidence score and decides resolved vs needs_escalation.
    """
    # IDs for logging
    ids = _ids_for_log(state, config)
    log.info("node_start_resolver", extra={"extra_data": ids})

    ticket = state["ticket"]
    intake = state.get("intake", {}) or {}
    classification = state.get("classification", {}) or {}

    # Raw user text from the ticket
    raw_text = (ticket.get("content") or "").strip()
    # Normalized summary from Intake (nice for LLM context, but not ideal as a SQL search string)
    normalized_issue = (intake.get("normalized_issue") or "").strip()

    # For the LLM answer, we still want the nice normalized issue if available
    user_issue = normalized_issue or raw_text

    # For KB search, lean on raw user question, optionally enriched with normalized form
    kb_query_parts = [raw_text] if raw_text else []
    if normalized_issue and normalized_issue.lower() not in raw_text.lower():
        kb_query_parts.append(normalized_issue)
    kb_query = " ".join(kb_query_parts).strip()

    external_user_id = ticket.get("owner_id")
    ticket_id = ticket.get("ticket_id")
    thread_id = config.get("configurable", {}).get("thread_id") if config else None

    # Knowledge retrieval via KB tool (async) – with normalization to a list
    used_kb_articles: list[str] = []
    kb_results: list[dict[str, Any]] = []
    top_score = 0.0

    kb_tool = None
    if kb_query:
        try:
            tools = await aget_kb_tools()
            for t in tools:
                if "kb_search" in t.name:
                    kb_tool = t
                    break
        except Exception as e:
            log.error(
                "kb_tool_fetch_error",
                extra={
                    "extra_data": {
                        "ticket_id": ticket_id,
                        "thread_id": thread_id,
                        "error": str(e),
                    }
                },
            )

    if kb_tool is not None and kb_query:
        log.info(
            "tool_call_kb_search_start",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "query": kb_query,
                }
            },
        )

        kb_raw = await kb_tool.ainvoke({"query": kb_query, "limit": 5}) or []

        # Normalize JSON-string → list[dict]
        if isinstance(kb_raw, str):
            kb_raw = kb_raw.strip()
            if kb_raw:
                try:
                    kb_results = json.loads(kb_raw)
                except json.JSONDecodeError:
                    kb_results = []
            else:
                kb_results = []
        elif isinstance(kb_raw, list):
            kb_results = kb_raw
        else:
            kb_results = []

        if kb_results:
            used_kb_articles = [
                a.get("article_id")
                for a in kb_results
                if isinstance(a, dict) and a.get("article_id")
            ]
            first = kb_results[0]
            if isinstance(first, dict):
                # In kb_search, "score" is the count of matching query words
                top_score = float(first.get("score", 0.0) or 0.0)

        log.info(
            "tool_call_kb_search_end",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "result_count": len(kb_results),
                    "top_score": top_score,
                }
            },
        )

    # Confidence based on KB hits + lexical overlap heuristic
    if not kb_results:
        confidence = 0.2
        lexical_overlap = 0.0
    else:
        # Base confidence mapping from raw score
        base_confidence = max(0.5, min(0.95, top_score / 2.0 + 0.3))

        # Compute lexical overlap: score / number of query tokens
        query_tokens = [
            w for w in kb_query.lower().split()
            if w.strip()
        ]
        query_len = len(query_tokens) or 1
        lexical_overlap = float(top_score) / float(query_len)

        # If overlap is very low, clamp confidence down — KB hits are probably spurious
        if lexical_overlap < 0.25:
            confidence = min(base_confidence, 0.4)
        else:
            confidence = base_confidence

    # Additional guard: salient token overlap between user query and KB content
    SALIENT_STOPWORDS = {
        "do", "does", "did", "you", "your", "yours",
        "the", "and", "or", "for", "to", "of", "in", "on", "at",
        "a", "an", "is", "are", "was", "were", "be", "been",
        "with", "about", "this", "that", "it", "my", "our", "we",
        "can", "could", "would", "should"
    }

    # salient tokens only from the *raw* user text
    raw_tokens = [
        t.strip("?,.!").lower()
        for t in raw_text.split()
        if t.strip()
    ]
    salient_tokens = {
        t for t in raw_tokens
        if len(t) >= 4 and t not in SALIENT_STOPWORDS
    }

    kb_text_concat = ""
    if kb_results:
        kb_text_concat = " ".join(
            (art.get("title", "") + " " + art.get("content", ""))
            for art in kb_results[:3]
        ).lower()

    salient_hits = [
        t for t in salient_tokens
        if t in kb_text_concat
    ]
    salient_overlap = (
        len(salient_hits) / float(len(salient_tokens))
        if salient_tokens else 0.0
    )

    # If salient overlap is very low, clamp confidence as well
    if kb_results and salient_tokens and salient_overlap < 0.4:
        confidence = min(confidence, 0.4)

    log.info(
        "resolver_kb_confidence",
        extra={
            "extra_data": {
                "ticket_id": ticket_id,
                "thread_id": thread_id,
                "kb_result_count": len(kb_results),
                "top_score": top_score,
                "lexical_overlap": lexical_overlap,
                "salient_overlap": salient_overlap,
                "salient_tokens": list(salient_tokens),
                "salient_hits": salient_hits,
                "confidence": confidence,
            }
        },
    )

    # Account lookup for personalization (async)
    account_snippet = ""

    if external_user_id and ACCOUNT_GET_USER_TOOL is not None:
        log.info(
            "tool_call_account_get_user_start",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "external_user_id": external_user_id,
                }
            },
        )
        user_info = await ACCOUNT_GET_USER_TOOL.ainvoke(
            {"external_user_id": external_user_id}
        )
        log.info(
            "tool_call_account_get_user_end",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "has_user_info": bool(user_info),
                }
            },
        )
        if user_info:
            account_snippet += f"User profile: {user_info}\n\n"

    if external_user_id and ACCOUNT_GET_RESERVATIONS_TOOL is not None:
        log.info(
            "tool_call_account_get_reservations_start",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "external_user_id": external_user_id,
                }
            },
        )
        reservations = await ACCOUNT_GET_RESERVATIONS_TOOL.ainvoke(
            {"external_user_id": external_user_id}
        )
        log.info(
            "tool_call_account_get_reservations_end",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "reservations_count": len(reservations) if reservations else 0,
                }
            },
        )
        if reservations:
            account_snippet += f"Current reservations: {reservations}\n\n"

    # Long-term memory search (async)
    memory_snippet = ""
    if external_user_id and MEMORY_SEARCH_TOOL is not None:
        log.info(
            "tool_call_memory_search_start",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "external_user_id": external_user_id,
                    "query": user_issue,
                }
            },
        )
        memories = await MEMORY_SEARCH_TOOL.ainvoke(
            {
                "external_user_id": external_user_id,
                "query": user_issue,
                "limit": 5,
            }
        )
        log.info(
            "tool_call_memory_search_end",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "thread_id": thread_id,
                    "memories_count": len(memories) if memories else 0,
                }
            },
        )
        if memories:
            memory_snippet = f"Relevant prior memories: {memories}\n\n"

    # Prepare KB context for the LLM
    kb_snippet = ""
    if kb_results:
        kb_snippet_lines = []
        for art in kb_results[:3]:
            kb_snippet_lines.append(
                f"Title: {art.get('title')}\nContent:\n{art.get('content')}\n"
            )
        kb_snippet = "\n\n---\n\n".join(kb_snippet_lines)

    # Decide if we should attempt resolution or escalate
    # Require: some KB hits, reasonable confidence, and non-trivial overlaps
    likely_resolvable = (
        bool(kb_results)
        and confidence >= 0.5
        and lexical_overlap >= 0.3
        and (not salient_tokens or salient_overlap >= 0.4)
    )

    if not likely_resolvable:
        resolution = {
            "status": "needs_escalation",
            "answer": (
                "I'm not fully confident I can resolve this automatically based on the "
                "available knowledge. A human support agent should review this ticket."
            ),
            "confidence": confidence,
            "used_kb_articles": used_kb_articles,
            "notes_for_human": (
                "KB search returned no clearly related matches for the key terms in the "
                f"user's question (salient_overlap={salient_overlap:.2f}). "
                "Please manually review the issue and consider updating the KB."
            ),
        }
        log.info(
            "resolver_decision_needs_escalation",
            extra={
                "extra_data": {
                    **ids,
                    "confidence": confidence,
                    "kb_result_count": len(kb_results),
                    "lexical_overlap": lexical_overlap,
                    "salient_overlap": salient_overlap,
                }
            },
        )
        log.info(
            "node_end_resolver",
            extra={
                "extra_data": {
                    **ids,
                    "status": "needs_escalation",
                }
            },
        )
        return {"resolution": resolution}

    # Use LLM to craft a grounded answer
    system_prompt = (
        "You are the Resolver Agent for UDA-Hub, helping CultPass users.\n"
        "You MUST base your answer ONLY on the knowledge base articles and data provided.\n"
        "If something is not covered in the knowledge, do not invent a policy.\n\n"
        "Respond in a friendly, concise tone.\n"
    )

    user_prompt = (
        f"User issue (normalized): {user_issue}\n\n"
        f"Issue type: {classification.get('issue_type', 'other')}\n"
        f"Urgency: {classification.get('urgency', 'low')}\n"
        f"Complexity: {classification.get('complexity', 'low')}\n\n"
        f"{account_snippet}"
        f"{memory_snippet}"
        "Relevant knowledge base articles:\n"
        f"{kb_snippet}\n\n"
        "Using ONLY the information above, draft a final answer to the user.\n"
        "Do NOT mention internal tools or KB article IDs. "
        "Just explain what the user should do or what we can do for them."
    )

    llm_resp = await LLM.ainvoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    answer_text = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)

    status = "resolved"
    notes_for_human = (
        "Resolved automatically using KB content. "
        "If the user replies that this didn't help, escalate."
    )

    resolution = {
        "status": status,
        "answer": answer_text,
        "confidence": confidence,
        "used_kb_articles": used_kb_articles,
        "notes_for_human": notes_for_human,
    }

    log.info(
        "resolver_decision_resolved",
        extra={
            "extra_data": {
                **ids,
                "confidence": confidence,
                "kb_result_count": len(kb_results),
                "lexical_overlap": lexical_overlap,
                "salient_overlap": salient_overlap,
            }
        },
    )
    log.info(
        "node_end_resolver",
        extra={"extra_data": {**ids, "status": "resolved"}},
    )

    # Write long-term memory (best-effort)
    if external_user_id and MEMORY_WRITE_TOOL is not None:
        try:
            await MEMORY_WRITE_TOOL.ainvoke(
                {
                    "external_user_id": external_user_id,
                    "ticket_id": ticket_id,
                    "content": f"Resolved issue: {user_issue}\nAnswer: {answer_text}",
                    "metadata": {
                        "issue_type": classification.get("issue_type"),
                        "kb_articles": used_kb_articles,
                        "confidence": confidence,
                    },
                }
            )
        except Exception:
            # Don't break the workflow on memory write failure
            pass

    return {"resolution": resolution}


def escalation_node(state: TicketState, config: RunnableConfig) -> TicketState:
    """
    Prepare a structured escalation summary for a human agent.
    """
    ids = _ids_for_log(state, config)
    log.info("node_start_escalation", extra={"extra_data": ids})

    ticket = state["ticket"]
    intake = state.get("intake", {})
    classification = state.get("classification", {})
    resolution = state.get("resolution", {})

    resolver_notes = resolution.get("notes_for_human") if resolution else None

    result = ESCALATION_AGENT.invoke(
        {
            "ticket_content": ticket.get("content", ""),
            "intake_summary": intake.get("summary", ""),
            "sentiment": intake.get("sentiment", "neutral"),
            "classification": classification,
            "resolver_notes": resolver_notes or "",
        }
    )

    log.info(
        "node_end_escalation",
        extra={
            "extra_data": {
                **ids,
                "recommended_department": result.get("recommended_department"),
            }
        },
    )

    return {"escalation": result}


def supervisor_node(state: TicketState, config: RunnableConfig) -> TicketState:
    """
    Decide the next step: 'resolver', 'escalation', or 'done'.
    """
    ids = _ids_for_log(state, config)
    log.info("node_start_supervisor", extra={"extra_data": ids})

    intake = state.get("intake", {})
    classification = state.get("classification", {})
    resolution = state.get("resolution", {})

    result = SUPERVISOR_AGENT.invoke(
        {
            "summary": intake.get("summary", ""),
            "issue_type": classification.get("issue_type", "other"),
            "urgency": classification.get("urgency", "low"),
            "complexity": classification.get("complexity", "low"),
            "resolver_status": resolution.get("status") if resolution else None,
            "resolver_confidence": resolution.get("confidence") if resolution else None,
        }
    )

    log.info(
        "supervisor_decision",
        extra={
            "extra_data": {
                **ids,
                "next_step": result.get("next_step"),
                "reason": result.get("reason"),
                "issue_type": classification.get("issue_type", "other"),
                "urgency": classification.get("urgency", "low"),
                "complexity": classification.get("complexity", "low"),
            }
        },
    )

    log.info(
        "node_end_supervisor",
        extra={"extra_data": {**ids, "next_step": result.get("next_step")}},
    )

    return {"supervisor": result}


# Routing logic

def route_from_supervisor(state: TicketState) -> str:
    supervisor = state.get("supervisor", {}) or {}
    next_step = supervisor.get("next_step", "done")

    if next_step == "resolver":
        return "resolver"
    if next_step == "escalation":
        return "escalation"
    return END


# Build and compile workflow

def build_workflow():
    graph = StateGraph(TicketState)

    graph.add_node("intake", intake_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", "supervisor")
    graph.add_edge("resolver", "supervisor")
    graph.add_edge("escalation", END)

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "resolver": "resolver",
            "escalation": "escalation",
            END: END,
        },
    )

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app


orchestrator = build_workflow()


def run_ticket(ticket: Dict[str, Any], thread_id: str) -> TicketState:
    """
    Run the LangGraph workflow for a single ticket.
    """
    initial_state: TicketState = {
        "ticket": ticket,
    }

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    log.info(
        "workflow_start",
        extra={
            "extra_data": {
                "ticket_id": ticket.get("ticket_id"),
                "thread_id": thread_id,
            }
        },
    )

    final_state = asyncio.run(orchestrator.ainvoke(initial_state, config=config))

    # Log final resolution object
    resolution = (final_state or {}).get("resolution", {})
    log.info(
        "workflow_completed",
        extra={
            "extra_data": {
                "ticket_id": ticket.get("ticket_id"),
                "thread_id": thread_id,
                "final_status": resolution.get("status"),
                "final_confidence": resolution.get("confidence"),
            }
        },
    )

    return final_state
