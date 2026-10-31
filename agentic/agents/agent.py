"""Read and validate YAML role configurations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


AGENT_NAMES = ("supervisor", "query", "account", "validate", "escalation")


@lru_cache(maxsize=len(AGENT_NAMES))
def get_agent_soul(agent_name: str) -> dict[str, Any]:
    """Read and cache the complete YAML configuration for one agent role."""
    if (
        not isinstance(agent_name, str)
        or not agent_name
        or Path(agent_name).name != agent_name
        or any(separator in agent_name for separator in ("/", "\\"))
    ):
        raise ValueError("agent_name must be a simple role name")
    path = Path(__file__).parent / agent_name / "config.yaml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read agent YAML config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"agent YAML config {path} must contain a mapping")

    required_fields = {"name", "prompt"}
    optional_fields = {
        "model", "tools", "middleware", "context_schema", "response_format",
    }
    allowed_fields = required_fields | optional_fields
    missing_fields = sorted(required_fields - config.keys())
    unknown_fields = sorted(config.keys() - allowed_fields)
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {', '.join(missing_fields)}")
        if unknown_fields:
            details.append(f"unsupported fields: {', '.join(unknown_fields)}")
        raise ValueError(f"invalid agent YAML config {path}: {'; '.join(details)}")
    if config["name"] != agent_name:
        raise ValueError(f"agent YAML config {path} must have name: {agent_name}")

    prompt = config["prompt"]
    if not isinstance(prompt, dict):
        raise ValueError(f"agent YAML config {path} prompt must be a mapping")
    required_prompt_fields = {"role", "objective", "instructions", "boundaries"}
    missing_prompt_fields = sorted(required_prompt_fields - prompt.keys())
    unknown_prompt_fields = sorted(prompt.keys() - required_prompt_fields)
    if missing_prompt_fields or unknown_prompt_fields:
        details = []
        if missing_prompt_fields:
            details.append(f"missing fields: {', '.join(missing_prompt_fields)}")
        if unknown_prompt_fields:
            details.append(f"unsupported fields: {', '.join(unknown_prompt_fields)}")
        raise ValueError(f"invalid prompt config in {path}: {'; '.join(details)}")
    return config

__all__ = ["AGENT_NAMES", "get_agent_soul"]
