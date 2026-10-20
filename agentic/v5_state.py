"""Strict V5 state initialization and authorized patch merging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import uuid


class StateContractError(ValueError):
    """Raised when state data violates the V5 contract or writer boundary."""


GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "ticket": {
        "ticket_id": None,
        "status": "processing",
        "messages": [],
        "input": "",
        "owner_id": None,
        "owner_name": None,
    },
    "supervisor": {
        "intent": "unknown",
        "intent_version": 1,
        "next_node": None,
        "query_instruction": None,
        "message": None,
    },
    "query": {
        "answer": None,
        "description": None,
        "validated": None,
        "evidence": [],
        "next_node": None,
    },
    "clarify": {"prompt": None},
    "confirm": {"prompt": None, "decision": None},
    "account": {"pending_action": None, "status": "pending", "description": None},
    "validate": {
        "context_relevance": None,
        "faithfulness": None,
        "answer_relevance": None,
        "next_node": None,
        "feedback": None,
    },
    "escalation": {"description": None, "summary": None},
    "budget": {
        "outer_nodes_used": 0,
        "query_calls_used": 0,
        "answer_round": 0,
        "business_calls_used": 0,
    },
}

ENUMS = {
    ("ticket", "status"): {"processing", "waiting_reply", "completed", "handed_off"},
    ("supervisor", "intent"): {"query", "account", "escalation", "unknown"},
    ("supervisor", "next_node"): {"query", "account", "clarify", "escalation", "finish", None},
    ("query", "next_node"): {"validate", "supervisor", None},
    ("confirm", "decision"): {"confirm", "cancel", None},
    ("account", "status"): {"pending", "succeeded", "failed", "unresolved", "needs_human"},
    ("validate", "next_node"): {"query", "supervisor", None},
}

WRITERS: dict[str, dict[str, set[str]]] = {
    "entry": {"ticket": {"messages", "input"}},
    "supervisor": {"supervisor": {"intent", "intent_version", "next_node", "query_instruction", "message"}},
    "query": {"query": {"answer", "description", "evidence", "next_node"}},
    "clarify": {"clarify": {"prompt"}},
    "confirm": {"confirm": {"prompt", "decision"}},
    "account": {"account": {"pending_action", "status", "description"}},
    "validate": {"validate": {"context_relevance", "faithfulness", "answer_relevance", "next_node", "feedback"}},
    "escalation": {"escalation": {"description", "summary"}},
    # The execution boundary owns identity, status, routing authorization,
    # validation verdicts, all budgets, and cross-node transitions.
    "boundary": {group: set(fields) for group, fields in {
        "ticket": GROUP_DEFAULTS["ticket"],
        "supervisor": GROUP_DEFAULTS["supervisor"],
        "query": GROUP_DEFAULTS["query"],
        "clarify": GROUP_DEFAULTS["clarify"],
        "confirm": GROUP_DEFAULTS["confirm"],
        "account": GROUP_DEFAULTS["account"],
        "validate": GROUP_DEFAULTS["validate"],
        "escalation": GROUP_DEFAULTS["escalation"],
        "budget": GROUP_DEFAULTS["budget"],
    }.items()},
}


def initial_state(
    *, ticket_id: str, owner_id: str, owner_name: str | None, message: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        raise StateContractError("ticket_id must be a non-empty string")
    if not isinstance(owner_id, str) or not owner_id:
        raise StateContractError("owner_id must be a string")
    state = deepcopy(GROUP_DEFAULTS)
    _ensure_message_id(message)
    state["ticket"].update(
        ticket_id=ticket_id,
        messages=[message],
        input=_message_text(message),
        owner_id=owner_id,
        owner_name=owner_name,
    )
    validate_state(state)
    return state


def merge_patch(
    state: dict[str, Any], patch: dict[str, Any], *, writer: str,
) -> dict[str, Any]:
    """Merge authorized field changes into replacement group objects.

    Groups omitted from ``patch`` are retained. Within a supplied group,
    omitted fields are retained; ``None`` and empty lists explicitly clear
    nullable/list fields. The input state and its nested groups are never
    mutated.
    """
    if writer not in WRITERS:
        raise StateContractError(f"unknown state writer: {writer}")
    validate_state(state)
    if not isinstance(patch, dict):
        raise StateContractError("state patch must be an object")
    result = deepcopy(state)
    permissions = WRITERS[writer]
    for group, updates in patch.items():
        if group not in GROUP_DEFAULTS:
            raise StateContractError(f"unknown state group: {group}")
        if group not in permissions:
            raise StateContractError(f"writer {writer} cannot update group {group}")
        if not isinstance(updates, dict):
            raise StateContractError(f"state group {group} must be an object")
        unknown = set(updates) - set(GROUP_DEFAULTS[group])
        if unknown:
            raise StateContractError(f"unknown fields in {group}: {sorted(unknown)}")
        denied = set(updates) - permissions[group]
        if denied:
            raise StateContractError(f"writer {writer} cannot update {group}.{sorted(denied)[0]}")
        result[group] = {**result[group], **deepcopy(updates)}
    validate_state(result)
    return result


def validate_state(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != set(GROUP_DEFAULTS):
        raise StateContractError("state must contain exactly the nine V5 groups")
    for group, fields in GROUP_DEFAULTS.items():
        value = state[group]
        if not isinstance(value, dict) or set(value) != set(fields):
            raise StateContractError(f"{group} must contain exactly its V5 fields")
        for field, item in value.items():
            _validate_field(group, field, item)
    if state["account"]["status"] != "pending" and not state["account"]["description"]:
        raise StateContractError("account.description is required when account.status is not pending")
    if state["ticket"]["status"] == "handed_off":
        if not state["escalation"]["description"] or not state["escalation"]["summary"]:
            raise StateContractError("handed_off ticket requires complete escalation material")


def _validate_field(group: str, field: str, value: Any) -> None:
    enum_values = ENUMS.get((group, field))
    if (group, field) in ENUMS:
        if not any(type(value) is type(option) and value == option for option in enum_values):
            raise StateContractError(f"invalid {group}.{field}: {value!r}")
        return
    if (group, field) in {
        ("ticket", "ticket_id"), ("ticket", "input"), ("ticket", "owner_id"),
        ("ticket", "owner_name"), ("supervisor", "query_instruction"),
        ("supervisor", "message"), ("query", "answer"), ("query", "description"),
        ("clarify", "prompt"), ("confirm", "prompt"), ("account", "description"),
        ("validate", "feedback"), ("escalation", "description"),
        ("escalation", "summary"),
    }:
        required = (group, field) in {
            ("ticket", "ticket_id"), ("ticket", "input"), ("ticket", "owner_id"),
        }
        if required and (not isinstance(value, str) or not value.strip()):
            raise StateContractError(f"{group}.{field} must be a non-empty string")
        if value is not None and (not isinstance(value, str) or (not value.strip() and field != "input")):
            raise StateContractError(f"{group}.{field} must be a non-empty string or null")
    elif (group, field) == ("ticket", "messages"):
        if not isinstance(value, list) or any(
            not (
                isinstance(message.get("id"), str) and message["id"].strip()
            ) if isinstance(message, dict) else not (
                isinstance(getattr(message, "id", None), str) and getattr(message, "id", "").strip()
            ) for message in value
        ):
            raise StateContractError("ticket.messages must be native messages with stable IDs")
    elif (group, field) == ("query", "validated") or group == "validate" and field in {
        "context_relevance", "faithfulness", "answer_relevance",
    }:
        if value is not None and type(value) is not bool:
            raise StateContractError(f"{group}.{field} must be bool or null")
    elif (group, field) == ("query", "evidence"):
        _validate_evidence(value)
    elif (group, field) == ("account", "pending_action"):
        _validate_pending_action(value)
    elif group == "budget":
        if type(value) is not int or value < 0:
            raise StateContractError(f"budget.{field} must be a non-negative integer")
        limits = {"outer_nodes_used": 12, "query_calls_used": 6, "answer_round": 6, "business_calls_used": 3}
        if value > limits[field]:
            raise StateContractError(f"budget.{field} exceeds {limits[field]}")
    elif (group, field) == ("supervisor", "intent_version"):
        if type(value) is not int or value < 1:
            raise StateContractError("supervisor.intent_version must be an integer >= 1")

def _validate_evidence(value: Any) -> None:
    if not isinstance(value, list):
        raise StateContractError("query.evidence must be a list")
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"source_type", "source_ref", "content"}:
            raise StateContractError("each evidence item must contain exactly source_type/source_ref/content")
        if not isinstance(entry["source_type"], str) or entry["source_type"] not in {
            "public_knowledge", "user_state", "ticket_history",
        }:
            raise StateContractError("invalid evidence source_type")
        if any(not isinstance(entry[key], str) or not entry[key].strip() for key in ("source_ref", "content")):
            raise StateContractError("evidence source_ref and content must be non-empty strings")


def _validate_pending_action(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"tool_name", "tool_args"}:
        raise StateContractError("pending_action must contain exactly tool_name/tool_args")
    if not isinstance(value["tool_name"], str) or value["tool_name"] not in {
        "cancel_subscription", "cancel_reservation",
    }:
        raise StateContractError("pending_action tool_name is not registered")
    if not isinstance(value["tool_args"], dict):
        raise StateContractError("pending_action.tool_args must be an object")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return "" if content is None else str(content)


def _ensure_message_id(message: Any) -> None:
    message_id = message.get("id") if isinstance(message, dict) else getattr(message, "id", None)
    if isinstance(message_id, str) and message_id.strip():
        return
    assigned = str(uuid.uuid4())
    if isinstance(message, dict):
        message["id"] = assigned
        return
    try:
        message.id = assigned
    except (AttributeError, TypeError) as error:
        raise StateContractError("initial native message must accept a stable ID") from error
