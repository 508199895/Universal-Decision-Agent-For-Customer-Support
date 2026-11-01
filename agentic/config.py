"""Application-wide configuration loading and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


AGENTIC_ROOT = Path(__file__).resolve().parent
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def load_app_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the application-wide model/tool/workflow selection."""
    config_path = Path(path) if path is not None else AGENTIC_ROOT / "config.yml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read application config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"application config {config_path} must contain a mapping")
    if set(config) != {"models", "tools", "workflow"}:
        raise ValueError("application config must contain exactly models, tools, and workflow")

    models = config["models"]
    if not isinstance(models, dict) or not isinstance(models.get("enabled"), dict):
        raise ValueError("application config models.enabled must be a mapping")
    if not models["enabled"]:
        raise ValueError("application config must enable at least one model")
    default_model = models.get("default")
    if not isinstance(default_model, str) or default_model not in models["enabled"]:
        raise ValueError("application config models.default must reference an enabled model")
    for model_name, model_config in models["enabled"].items():
        if not isinstance(model_name, str) or not _NAME_PATTERN.fullmatch(model_name):
            raise ValueError(f"invalid model name: {model_name!r}")
        if not isinstance(model_config, dict) or not isinstance(model_config.get("model"), str):
            raise ValueError(f"models.enabled.{model_name} must declare a model identifier")
        if model_config.get("api_mode", "chat_completions") not in {
            "chat_completions", "responses",
        }:
            raise ValueError(
                f"models.enabled.{model_name}.api_mode must be chat_completions or responses"
            )
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise ValueError(
                f"models.enabled.{model_name}.reasoning_effort must be a non-empty string"
            )

    tools = config["tools"]
    enabled_tools = tools.get("enabled") if isinstance(tools, dict) else None
    if not isinstance(enabled_tools, list) or any(
        not isinstance(name, str) or not name.strip() for name in enabled_tools
    ):
        raise ValueError("application config tools.enabled must be a list of tool names")
    if len(enabled_tools) != len(set(enabled_tools)):
        raise ValueError("application config tools.enabled must not contain duplicates")

    workflow = config["workflow"]
    if not isinstance(workflow, dict) or set(workflow) != {"directory", "name"}:
        raise ValueError("application config workflow must contain directory and name")
    directory, workflow_name = workflow["directory"], workflow["name"]
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("workflow.directory must be a non-empty relative path")
    if Path(directory).is_absolute() or ".." in Path(directory).parts:
        raise ValueError("workflow.directory must stay within the agentic directory")
    if not isinstance(workflow_name, str) or not _NAME_PATTERN.fullmatch(workflow_name):
        raise ValueError("workflow.name must be a simple workflow name")
    resolved_workflow_dir = (AGENTIC_ROOT / directory).resolve()
    try:
        resolved_workflow_dir.relative_to(AGENTIC_ROOT.resolve())
    except ValueError as error:
        raise ValueError("workflow.directory must stay within the agentic directory") from error
    return config


__all__ = ["AGENTIC_ROOT", "load_app_config"]
