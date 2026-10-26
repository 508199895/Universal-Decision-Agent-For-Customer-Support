from __future__ import annotations

from agentic.observability.langsmith_tracing import (
    build_trace_config,
    redact_trace_data,
)
from agentic.orchestration.workflow import _invoke_with_journal


def test_redactor_masks_configured_secrets_and_sensitive_fields_recursively():
    secret = "synthetic-langsmith-key-123"
    value = {
        "input": {"content": f"request included {secret}"},
        "metadata": {"api_key": secret},
        "tags": [secret],
    }

    redacted = redact_trace_data(value, secrets=(secret,))

    assert secret not in repr(redacted)
    assert "request included [REDACTED]" in redacted["input"]["content"]
    assert redacted["metadata"]["api_key"] == "[REDACTED]"
    assert redact_trace_data({"api_key": "different-secret"}) == {"api_key": "[REDACTED]"}


def test_redactor_drops_markdown_file_contents_with_source_reference():
    value = {
        "evidence": {
            "source_ref": "policies/cancellation.md",
            "content": "# Cancellation policy\nInternal markdown content",
            "score": 0.91,
        },
        "debug": "tool returned 2 records",
    }

    redacted = redact_trace_data(value)

    assert "Internal markdown content" not in repr(redacted)
    assert redacted["evidence"] == {"redacted": "markdown file"}
    assert redacted["debug"] == "tool returned 2 records"
    assert redact_trace_data({"token_usage": {"total_tokens": 6}}) == {
        "token_usage": {"total_tokens": 6},
    }


def test_redactor_masks_markdown_paths_and_bearer_credentials():
    redacted = redact_trace_data({
        "path": "docs/private.md",
        "authorization": "Bearer synthetic-token",
    })

    assert redacted == {
        "path": "[REDACTED MARKDOWN FILE]",
        "authorization": "[REDACTED]",
    }


def test_redactor_masks_markdown_with_common_metadata_path_aliases():
    for path_key in ("source", "file_path", "file_name", "source_file"):
        redacted = redact_trace_data({
            "metadata": {path_key: "knowledge/policy.md"},
            "page_content": "# Private test policy",
            "status": "retrieved",
        })
        assert redacted == {"redacted": "markdown file"}


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

    class ClientStub:
        def __init__(self, **options):
            self.options = options

    monkeypatch.setattr(langsmith, "Client", ClientStub)

    config = build_trace_config("ticket-123")
    tracer = config["callbacks"][0]

    assert config["run_name"] == "customer_support_ticket"
    assert config["metadata"] == {"ticket_id": "ticket-123"}
    assert config["tags"] == ["customer-support", "ticket-123"]
    assert tracer.project_name == "support-tests"
    assert tracer.client.options["hide_inputs"]({"content": "synthetic-langsmith-key-123"}) == {
        "content": "[REDACTED]",
    }
    assert tracer.client.options["hide_outputs"]({"content": "synthetic-langsmith-key-123"}) == {
        "content": "[REDACTED]",
    }
    assert tracer.client.options["hide_metadata"]({"credential": "synthetic-langsmith-key-123"}) == {
        "credential": "[REDACTED]",
    }
    assert tracer.client.options["anonymizer"]({"error": "synthetic-langsmith-key-123"}) == {
        "error": "[REDACTED]",
    }
    assert tracer.client.options["hide_inputs"]({
        "source_ref": "private.md", "content": "fake source body",
    }) == {
        "redacted": "markdown file",
    }


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
