"""Build the public-knowledge query and validation expert graph."""

from __future__ import annotations

import html
import json
import re
from typing import Any

from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from agentic.agents.agent import build_agent_from_config
from agentic.agents.public_knowledge.state import (
    PublicKnowledgeResult,
    PublicKnowledgeState,
)
from agentic.capabilities.middleware.token_control import TicketRuntimeContext


TOOL_NAME = "query_public_knowledge"
TOOL_DESCRIPTION = (
    "Search public knowledge, independently validate the answer, and return its evidence."
)
MAX_QUERY_REVISIONS = 3
MAX_QUERY_ATTEMPTS = MAX_QUERY_REVISIONS + 1


def _candidate_answer(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("query Agent returned no messages")
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai" or getattr(message, "tool_calls", None):
            continue
        text = getattr(message, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        break
    raise ValueError("query Agent did not return a plain answer")


def extract_public_evidence(messages: list[Any]) -> list[dict[str, str]]:
    """Extract source-backed evidence from Qdrant tool results."""
    evidence: list[dict[str, str]] = []
    for message in messages:
        name, content = getattr(message, "name", None), getattr(message, "content", None)
        if not isinstance(name, str) or "qdrant-find" not in name:
            continue
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text_value = part if isinstance(part, str) else (
                part.get("text") if isinstance(part, dict) else None
            )
            if not isinstance(text_value, str):
                continue
            try:
                decoded = json.loads(text_value)
            except json.JSONDecodeError:
                decoded = None
            candidates = (
                [value for value in decoded if isinstance(value, str)]
                if isinstance(decoded, list) else [text_value]
            )
            for candidate in candidates:
                pattern = (
                    r"<entry>\s*<content>(.*?)</content>\s*<metadata>(.*?)</metadata>"
                    r"\s*</entry>"
                )
                for match in re.finditer(pattern, candidate, flags=re.DOTALL):
                    try:
                        metadata = json.loads(html.unescape(match.group(2).strip()))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    source_ref = next((
                        metadata.get(key)
                        for key in ("article_id", "source_ref", "uri", "path")
                        if isinstance(metadata.get(key), str) and metadata.get(key).strip()
                    ), None)
                    body = html.unescape(match.group(1)).strip()
                    if source_ref and body:
                        evidence.append({
                            "source_type": "public_knowledge",
                            "source_ref": source_ref,
                            "content": body,
                        })
    return evidence


def build_graph(*, app_config, models, available_tools):
    """Construct the isolated public-knowledge expert graph."""
    query_agent = build_agent_from_config(
        agent_name="query",
        config_filename="query.yaml",
        app_config=app_config,
        models=models,
        available_tools=available_tools,
    )
    validate_agent = build_agent_from_config(
        agent_name="validate",
        config_filename="validate.yaml",
        app_config=app_config,
        models=models,
        available_tools=available_tools,
        response_format=ProviderStrategy(PublicKnowledgeResult),
    )

    def initialize(_state: PublicKnowledgeState) -> dict[str, Any]:
        return {
            "candidate": "",
            "evidence": [],
            "attempt": 0,
            "feedback": "",
            "validation": {},
            "validated": False,
            "failure_summary": "",
            "result": None,
        }

    async def query(
        state: PublicKnowledgeState,
        config: RunnableConfig,
        runtime: Runtime[TicketRuntimeContext],
    ) -> dict[str, Any]:
        task = (
            "回答以下公共知识问题。只依据可检索到的公共知识，提供简洁答案和必要来源。\n"
            + state["task"]
        )
        if state["attempt"]:
            task += (
                "\n\n上一轮校核未通过。请根据以下具体原因修订答案，并重新检索或核实：\n"
                + state["feedback"]
                + "\n原候选答案：\n"
                + state["candidate"]
            )
        messages = list(state.get("messages", []))
        if state["attempt"]:
            messages.append(HumanMessage(content=task))
        result = await query_agent.ainvoke(
            {"messages": messages or [HumanMessage(content=task)]},
            config=config,
            context=runtime.context,
        )
        agent_messages = result.get("messages", []) if isinstance(result, dict) else []
        return {
            "messages": agent_messages,
            "candidate": _candidate_answer(agent_messages),
            "evidence": extract_public_evidence(agent_messages),
            "attempt": state["attempt"] + 1,
            "validation": {},
            "validated": False,
            "failure_summary": "",
            "result": None,
        }

    async def validate(
        state: PublicKnowledgeState,
        config: RunnableConfig,
        runtime: Runtime[TicketRuntimeContext],
    ) -> dict[str, Any]:
        review_input = {
            "question": state["task"],
            "answer": state["candidate"],
            "evidence": state["evidence"],
            "attempts": state["attempt"],
        }
        agent_result = await validate_agent.ainvoke(
            {"messages": [HumanMessage(
                content=(
                    "独立校核以下公共知识候选答案，并按图 State 的 PublicKnowledgeResult 结构化格式返回最终 result。"
                    "status、answer、evidence、attempts、failure_summary 和 validation 都必须完整填写；"
                    "通过时 status 为 succeeded 且 answer 使用候选答案，不通过时 status 为 failed 且 answer 为 null，"
                    "attempts 使用输入值，失败时说明具体原因。\n"
                    + json.dumps(review_input, ensure_ascii=False)
                ),
            )]},
            config=config,
            context=runtime.context,
        )
        structured = (
            agent_result.get("structured_response")
            if isinstance(agent_result, dict) else agent_result
        )
        final_result = (
            structured if isinstance(structured, PublicKnowledgeResult)
            else PublicKnowledgeResult.model_validate(structured)
        )
        validation = final_result.validation
        passed = all((
            validation.context_relevance,
            validation.faithfulness,
            validation.answer_relevance,
        ))
        feedback = "\n".join(filter(None, (
            "失败原因：" + "；".join(validation.failure_reasons)
            if validation.failure_reasons else "",
            "修订指令：" + validation.revision_instruction
            if validation.revision_instruction else "",
            "校核反馈：" + validation.feedback if validation.feedback else "",
        )))
        agent_messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
        return {
            "messages": agent_messages,
            "validation": validation.model_dump(),
            "validated": passed,
            "feedback": feedback,
            "failure_summary": final_result.failure_summary,
            "result": final_result,
        }

    def route(state: PublicKnowledgeState) -> str:
        return "done" if state["validated"] or state["attempt"] >= MAX_QUERY_ATTEMPTS else "retry"

    builder = StateGraph(PublicKnowledgeState, context_schema=TicketRuntimeContext)
    builder.add_node("initialize", initialize)
    builder.add_node("query", query)
    builder.add_node("validate", validate)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "query")
    builder.add_edge("query", "validate")
    builder.add_conditional_edges("validate", route, {"retry": "query", "done": END})
    return builder.compile()
