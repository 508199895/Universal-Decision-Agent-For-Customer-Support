import json

from langchain_core.messages import ToolMessage

from agentic.agents.public_knowledge.graph import extract_public_evidence as _public_evidence


def test_qdrant_evidence_parses_json_encoded_result_list():
    result = json.dumps([
        "Results for the query 'CultPass app crashes'",
        '<entry><content>Force close the app and relaunch it.</content>'
        '<metadata>{"article_id":"00d48632-d803-4c00-a17f-ba3a50e80ecc",'
        '"title":"Troubleshooting App Crashes or Freezes"}</metadata></entry>',
    ])

    evidence = _public_evidence([
        ToolMessage(
            name="qdrant_qdrant-find",
            tool_call_id="call-qdrant-json-result",
            content=result,
        ),
    ])

    assert evidence == [{
        "source_type": "public_knowledge",
        "source_ref": "00d48632-d803-4c00-a17f-ba3a50e80ecc",
        "content": "Force close the app and relaunch it.",
    }]


def test_qdrant_evidence_still_parses_text_content_blocks():
    evidence = _public_evidence([
        ToolMessage(
            name="qdrant_qdrant-find",
            tool_call_id="call-qdrant-text-blocks",
            content=[
                "Results for the query",
                '<entry><content>Cancellation rules.</content>'
                '<metadata>{"article_id":"policy-17"}</metadata></entry>',
            ],
        ),
    ])

    assert evidence == [{
        "source_type": "public_knowledge",
        "source_ref": "policy-17",
        "content": "Cancellation rules.",
    }]
