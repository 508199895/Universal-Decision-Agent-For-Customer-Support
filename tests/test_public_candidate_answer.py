import pytest
from langchain_core.messages import AIMessage

from agentic.agents.public_knowledge.graph import _candidate_answer


def test_candidate_answer_extracts_plain_string_content():
    assert _candidate_answer([AIMessage(content="  候选答案。  ")]) == "候选答案。"


def test_candidate_answer_extracts_text_from_response_content_blocks():
    message = AIMessage(content=[
        {
            "type": "text",
            "text": "候选答案。",
            "id": "message-1",
            "phase": "final",
            "annotations": [],
        },
    ])

    assert _candidate_answer([message]) == "候选答案。"


def test_candidate_answer_skips_tool_call_messages():
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "qdrant_search", "args": {}, "id": "call-1"}],
    )
    answer = AIMessage(content=[{"type": "text", "text": "最终候选答案。"}])

    assert _candidate_answer([answer, tool_call]) == "最终候选答案。"


def test_candidate_answer_rejects_final_message_without_text():
    message = AIMessage(content=[{"type": "reasoning", "reasoning": "内部推理"}])

    with pytest.raises(ValueError, match="plain answer"):
        _candidate_answer([message])
