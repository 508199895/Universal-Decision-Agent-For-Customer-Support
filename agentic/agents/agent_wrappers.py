"""Build isolated specialist subgraphs and expose them as supervisor tools."""

from __future__ import annotations

import asyncio
import html
import json
import re
from typing import Any, TypedDict

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agentic.capabilities.middleware.token_control import TicketRuntimeContext
from agentic.agents.validate.contract import ValidateResult


MAX_QUERY_REVISIONS = 3
MAX_QUERY_ATTEMPTS = MAX_QUERY_REVISIONS + 1


class QueryValidateState(TypedDict):
    """Private state for the query/validate subgraph; no parent fields exist."""

    question: str
    candidate: str
    evidence: list[dict[str, str]]
    attempt: int
    feedback: str
    validation: dict[str, Any]
    validated: bool
    status: str
    failure_summary: str


async def _invoke(agent: Any, *, task: str, config: RunnableConfig,
                  context: TicketRuntimeContext) -> Any:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("specialist task must be a non-empty string")
    # Every specialist receives a newly constructed AgentState, never parent messages/state.
    return await asyncio.wait_for(
        agent.ainvoke({"messages": [HumanMessage(content=task)]},
                      config=config, context=context),
        timeout=60,
    )


def _candidate_answer(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("specialist returned no messages")
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai" or getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        break
    raise ValueError("specialist did not return a plain answer")


def _public_evidence(messages: list[Any]) -> list[dict[str, str]]:
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
            text_values = (
                [value for value in decoded if isinstance(value, str)]
                if isinstance(decoded, list) else [text_value]
            )
            for text_value in text_values:
                pattern = r"<entry>\s*<content>(.*?)</content>\s*<metadata>(.*?)</metadata>\s*</entry>"
                for match in re.finditer(pattern, text_value, flags=re.DOTALL):
                    try:
                        metadata = json.loads(html.unescape(match.group(2).strip()))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    source_ref = next((
                        metadata.get(key) for key in ("article_id", "source_ref", "uri", "path")
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


def build_query_validate_subgraph(*, query_agent: Any, validate_agent: Any):
    """Compile isolated query and validation agents with up to three revisions."""
    if query_agent is None or validate_agent is None:
        raise ValueError("query and validate agents are required")

    async def query(state: QueryValidateState, config: RunnableConfig,
                    runtime) -> dict[str, Any]:
        revision = ""
        if state["attempt"]:
            revision = (
                "\n\n上一轮校核未通过。请根据以下具体原因修订答案，并重新检索/核实：\n"
                + state["feedback"]
                + "\n原候选答案：\n" + state["candidate"]
            )
        result = await _invoke(
            query_agent,
            task="回答以下公共知识问题。只依据可检索到的公共知识，提供简洁答案和必要来源。\n"
            + state["question"] + revision,
            config=config,
            context=runtime.context,
        )
        messages = result.get("messages") if isinstance(result, dict) else None
        return {
            "candidate": _candidate_answer(messages),
            "evidence": _public_evidence(messages or []),
            "attempt": state["attempt"] + 1,
            "validation": {},
            "validated": False,
            "status": "reviewing",
        }

    async def validate(state: QueryValidateState, config: RunnableConfig,
                       runtime) -> dict[str, Any]:
        review_input = {
            "question": state["question"],
            "answer": state["candidate"],
            "evidence": state["evidence"],
        }
        result = await _invoke(
            validate_agent,
            task="独立校核以下公共知识候选答案。未通过时逐项给出具体原因和可执行的修订指令：\n"
            + json.dumps(review_input, ensure_ascii=False),
            config=config,
            context=runtime.context,
        )
        structured = (
            result.get("structured_response") if isinstance(result, dict) else result
        )
        validation = (
            structured if isinstance(structured, ValidateResult)
            else ValidateResult.model_validate(structured)
        )
        validation_dump = validation.model_dump()
        passed = all(validation_dump[key] for key in (
            "context_relevance", "faithfulness", "answer_relevance",
        ))
        revision_feedback = "\n".join(filter(None, (
            "失败原因：" + "；".join(validation.failure_reasons)
            if validation.failure_reasons else "",
            "修订指令：" + validation.revision_instruction
            if validation.revision_instruction else "",
            "校核反馈：" + validation.feedback if validation.feedback else "",
        )))
        return {
            "validation": validation_dump,
            "validated": passed,
            "feedback": revision_feedback,
            "status": "succeeded" if passed else "reviewing",
            "failure_summary": "" if passed else (
                revision_feedback
                or "候选答案未通过独立校核，校核方未提供具体原因。"
            ),
        }

    def route(state: QueryValidateState) -> str:
        if state["validated"] or state["attempt"] >= MAX_QUERY_ATTEMPTS:
            return "done"
        return "retry"

    builder = StateGraph(QueryValidateState, context_schema=TicketRuntimeContext)
    builder.add_node("query", query)
    builder.add_node("validate", validate)
    builder.add_edge(START, "query")
    builder.add_edge("query", "validate")
    builder.add_conditional_edges("validate", route, {"retry": "query", "done": END})
    return builder.compile()


def build_public_knowledge_tool(*, subgraph: Any) -> BaseTool:
    """Wrap an already-built query/validate subgraph as a supervisor tool."""
    if subgraph is None:
        raise ValueError("query/validate subgraph is required")
    @tool("query_public_knowledge")
    async def query_public_knowledge(question: str, runtime: ToolRuntime[TicketRuntimeContext]) -> str:
        """Search public knowledge, validate the answer, and return its status and evidence."""
        try:
            result = await subgraph.ainvoke({
                "question": question,
                "candidate": "",
                "evidence": [],
                "attempt": 0,
                "feedback": "",
                "validation": {},
                "validated": False,
                "status": "pending",
                "failure_summary": "",
            }, config=runtime.config, context=runtime.context)
        except Exception:
            return json.dumps({
                "status": "failed",
                "failure_summary": "公共知识查询或校核服务未能可靠完成。",
                "attempts": 0,
            }, ensure_ascii=False)
        passed = result["validated"] is True
        payload = {
            "status": "succeeded" if passed else "failed",
            "answer": result["candidate"] if passed else None,
            "evidence": result["evidence"] if passed else [],
            "validation": result["validation"],
            "attempts": result["attempt"],
            "failure_summary": "" if passed else result["failure_summary"],
        }
        return json.dumps(payload, ensure_ascii=False)

    return query_public_knowledge


def build_account_agent_tool(*, account_agent: Any) -> BaseTool:
    """Expose account lookup with an explicit task string and isolated child messages."""
    @tool("lookup_current_user_account")
    async def lookup_current_user_account(task: str, runtime: ToolRuntime[TicketRuntimeContext]) -> str:
        """Read the current user's account/profile/status information for this request."""
        try:
            result = await _invoke(
                account_agent, task=task, config=runtime.config, context=runtime.context,
            )
            messages = result.get("messages") if isinstance(result, dict) else None
            answer = _candidate_answer(messages)
            return json.dumps({"status": "succeeded", "answer": answer}, ensure_ascii=False)
        except Exception:
            return json.dumps({
                "status": "failed",
                "failure_summary": "账户资料查询未能可靠完成。",
            }, ensure_ascii=False)

    return lookup_current_user_account


