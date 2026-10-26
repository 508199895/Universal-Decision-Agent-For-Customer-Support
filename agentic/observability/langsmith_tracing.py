"""Environment-driven LangSmith tracing with credential and Markdown redaction."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable


_PATH_KEYS = {
    "path", "file", "filename", "file_name", "filepath", "file_path", "uri",
    "source", "source_ref", "source_path", "source_file",
}
_BEARER_TOKEN = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_COMMON_API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|lsv2_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+|"
    r"xox[baprs]-[A-Za-z0-9-]+)\b"
)


def _is_markdown_path(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().split("?", 1)[0].endswith(".md")


def _is_markdown_source(value: dict[Any, Any]) -> bool:
    has_markdown_path = _contains_markdown_path(value)
    has_document_body = any(
        isinstance(key, str) and key.lower() in {"content", "page_content", "body", "text"}
        for key in value
    )
    return has_markdown_path and has_document_body


def _contains_markdown_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and key.lower() in _PATH_KEYS and _is_markdown_path(item))
            or _contains_markdown_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_markdown_path(item) for item in value)
    return False


def _is_secret_key(value: str) -> bool:
    key = value.lower().replace("-", "_")
    return key in {
        "token", "authorization", "credential", "secret", "password", "api_key",
        "apikey", "access_token", "refresh_token",
    } or key.endswith((
        "_api_key", "apikey", "_access_token", "_refresh_token", "_secret",
        "_password", "_credential",
    ))


def _environment_secrets() -> tuple[str, ...]:
    return tuple(
        value for name, value in os.environ.items()
        if _is_secret_key(name) and len(value) >= 8
    )


def redact_trace_data(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    """Return a copy with configured credentials and Markdown files removed."""
    known_secrets = tuple(dict.fromkeys((*_environment_secrets(), *secrets)))
    if isinstance(value, dict):
        if _is_markdown_source(value):
            return {"redacted": "markdown file"}
        return {
            key: "[REDACTED]" if isinstance(key, str) and _is_secret_key(key)
            else "[REDACTED MARKDOWN FILE]" if isinstance(key, str) and _is_markdown_path(key)
            else redact_trace_data(item, secrets=known_secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_trace_data(item, secrets=known_secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_trace_data(item, secrets=known_secrets) for item in value)
    if isinstance(value, str):
        if _is_markdown_path(value):
            return "[REDACTED MARKDOWN FILE]"
        redacted = value
        for secret in known_secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        redacted = _BEARER_TOKEN.sub("Bearer [REDACTED]", redacted)
        return _COMMON_API_KEY.sub("[REDACTED]", redacted)
    return value


def _tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).lower() in {
        "1", "true", "yes", "on",
    }


def build_trace_config(ticket_id: str) -> dict[str, Any]:
    """Build per-ticket tracing settings, staying offline without opt-in and credentials."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not _tracing_enabled() or not api_key:
        return {}

    # Keep LangSmith optional for offline/test runs and import it only when enabled.
    from langsmith import Client
    from langchain_core.tracers.langchain import LangChainTracer

    secret_values = _environment_secrets()
    client_options: dict[str, Any] = {
        "anonymizer": lambda value: redact_trace_data(value, secrets=secret_values),
        "hide_inputs": lambda value: redact_trace_data(value, secrets=secret_values),
        "hide_outputs": lambda value: redact_trace_data(value, secrets=secret_values),
        "hide_metadata": lambda value: redact_trace_data(value, secrets=secret_values),
    }
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        client_options["api_url"] = endpoint
    client = Client(**client_options)
    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    tracer = LangChainTracer(client=client, project_name=project_name)
    safe_ticket_id = str(redact_trace_data(ticket_id, secrets=secret_values))
    metadata = redact_trace_data({"ticket_id": safe_ticket_id}, secrets=secret_values)
    tags = redact_trace_data(["customer-support", safe_ticket_id], secrets=secret_values)
    return {
        "run_name": "customer_support_ticket",
        "metadata": metadata,
        "tags": tags,
        "callbacks": [tracer],
    }
