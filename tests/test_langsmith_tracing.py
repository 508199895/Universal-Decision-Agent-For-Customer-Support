from __future__ import annotations

from pathlib import Path

from langsmith.anonymizer import create_secret_anonymizer

from agentic.observability import langsmith_tracing
from agentic.observability.langsmith_tracing import build_trace_config
from agentic.orchestration.workflow import _invoke_with_journal


def test_secret_anonymizer_redacts_secret_formats_but_preserves_other_trace_data():
    anonymize = create_secret_anonymizer()
    secret = "sk-" + "A" * 36
    payload = {
        "inputs": {"message": f"test key {secret}", "notes": "ordinary diagnostic text"},
        "outputs": {"authorization": f"Bearer {secret}", "token_usage": {"total_tokens": 6}},
        "metadata": {"ticket_id": "ticket-123", "source_ref": "policies/policy.md"},
        "tags": ["customer-support", secret],
    }

    redacted = anonymize(payload)

    assert secret not in repr(redacted)
    assert "[SECRET_DETECTED]" in repr(redacted)
    assert redacted["inputs"]["notes"] == "ordinary diagnostic text"
    assert redacted["metadata"]["source_ref"] == "policies/policy.md"
    assert redacted["outputs"]["token_usage"] == {"total_tokens": 6}


def test_trace_is_disabled_without_explicit_tracing_and_key(monkeypatch):
    for name in (
        "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert build_trace_config("ticket-123") == {}


def test_trace_config_has_ticket_metadata_and_redacting_callbacks(monkeypatch):
    import langsmith

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-langsmith-key-123")
    monkeypatch.setenv("LANGSMITH_PROJECT", "support-tests")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "another-project")

    class ClientStub:
        def __init__(self, **options):
            self.options = options

    monkeypatch.setattr(langsmith, "Client", ClientStub)

    config = build_trace_config("ticket-123")
    tracer = config["callbacks"][0]

    assert config["run_name"] == "customer_support_ticket"
    assert config["metadata"] == {"ticket_id": "ticket-123"}
    assert config["tags"] == ["customer-support", "ticket-123"]
    expected_project = Path(langsmith_tracing.__file__).resolve().parents[2].name
    assert tracer.project_name == expected_project
    anonymizer = tracer.client.options["anonymizer"]
    secret = "sk-" + "B" * 36
    for field in ("inputs", "outputs", "metadata", "error"):
        assert secret not in repr(anonymizer({field: f"credential {secret}"}))
    assert anonymizer({"source_ref": "private.md", "content": "markdown body"}) == {
        "source_ref": "private.md", "content": "markdown body",
    }

    secret_config = build_trace_config(secret)
    assert secret not in repr(secret_config["metadata"])
    assert secret not in repr(secret_config["tags"])


def test_workflow_invocation_attaches_ticket_trace_settings(monkeypatch):
    captured = {}
    trace_config = {
        "run_name": "customer_support_ticket",
        "metadata": {"ticket_id": "ticket-456"},
        "tags": ["customer-support", "ticket-456"],
        "callbacks": ["langsmith-tracer"],
    }
    monkeypatch.setattr(
        "agentic.orchestration.workflow.build_trace_config", lambda ticket_id: trace_config,
    )

    class Workflow:
        async def ainvoke(self, payload, *, config, durability):
            captured.update(config)
            return {"ok": True}

    class Journal:
        ticket_id = "ticket-456"

        async def record_lifecycle(self, *_args, **_kwargs):
            pass

        async def flush(self):
            pass

    import asyncio

    result = asyncio.run(_invoke_with_journal(Workflow(), "user-thread", Journal()))

    assert result == {"ok": True}
    assert captured["metadata"] == {"ticket_id": "ticket-456"}
    assert captured["tags"] == ["customer-support", "ticket-456"]
    assert isinstance(captured["callbacks"][0], Journal)
    assert captured["callbacks"][1:] == ["langsmith-tracer"]
