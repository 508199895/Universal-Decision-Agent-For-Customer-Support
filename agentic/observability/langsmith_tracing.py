"""Environment-driven LangSmith tracing with SDK secret anonymization."""

from __future__ import annotations

import os
from typing import Any


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
    from langsmith.anonymizer import create_secret_anonymizer
    from langchain_core.tracers.langchain import LangChainTracer

    secret_anonymizer = create_secret_anonymizer()
    client_options: dict[str, Any] = {"anonymizer": secret_anonymizer}
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        client_options["api_url"] = endpoint
    client = Client(**client_options)
    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    tracer = LangChainTracer(client=client, project_name=project_name)
    metadata = secret_anonymizer({"ticket_id": ticket_id})
    tags = secret_anonymizer({"tags": ["customer-support", ticket_id]})["tags"]
    return {
        "run_name": "customer_support_ticket",
        "metadata": metadata,
        "tags": tags,
        "callbacks": [tracer],
    }
