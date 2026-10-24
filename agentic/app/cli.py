"""
CLI entry point

Entry point for exercising the current handoff workflow from the command line.

This script:
- Loads environment variables (including OPENAI_API_KEY) from .env
- Opens the asynchronous workflow for the lifetime of the CLI session
- Persists workflow messages and lifecycle events with their customer-support ticket
- Provides:
    * a simple single-ticket demo (`demo` mode)
    * an interactive CLI loop (`chat` mode) where you can type new tickets

How to run:

    # Option 1:
    python -m agentic.app.cli --mode demo

    # Option 2:
    python -m agentic.app.cli --mode chat
"""

from __future__ import annotations

import asyncio
import argparse
import textwrap
import uuid
from typing import Any, Dict

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from agentic.orchestration.handoff import HANDOFF_NOTICE
from agentic.orchestration.workflow import (
    aenter_ticket,
    default_workflow_runtime,
)


# Helper functions


def pretty_print_section(title: str, content: Any) -> None:
    """Print a section with a title and nicely formatted content."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(content, dict):
        for k, v in content.items():
            print(f"{k}: {v}")
    else:
        print(content)


async def _handle_ticket(ticket: Dict[str, Any], workflow, event_store):
    user_id = ticket["user_id"]
    message = HumanMessage(
        content=ticket["content"],
        id=ticket.get("message_id") or f"customer:{user_id}:{uuid.uuid4()}",
    )
    return await aenter_ticket(
        ticket["external_user_id"], message,
        user_id=user_id,
        user_name=ticket.get("user_name"),
        workflow=workflow,
        ticket_event_store=event_store,
        ticket_id=ticket.get("ticket_id"),
        new_ticket=bool(ticket.get("new_ticket", False)),
    )


async def run_single_ticket_demo(workflow, event_store) -> None:
    """
    Run the current handoff-only workflow for a sample ticket.

    This verifies database setup and dialogue-based handoff generation.
    """
    # Sample identity values for the isolated agent ticket database.
    external_user_id = "a4ab87"
    ticket: Dict[str, Any] = {
        "content": "Hi, I can't log in to my CultPass account and I don't get the reset email.",
        "user_id": "demo-user-a4ab87",
        "account_id": "demo-account-cultpass",
        "external_user_id": external_user_id,
        "user_name": "Demo User",
        "ticket_id": f"DEMO-TICKET-{uuid.uuid4().hex[:8]}",
        "new_ticket": True,
    }

    print("\nRunning UDA-Hub workflow for a demo ticket...")
    final_state = await _handle_ticket(ticket, workflow, event_store)

    escalation = final_state.get("escalation") or {}

    pretty_print_section("Ticket", ticket)
    pretty_print_section("交接摘要", escalation.get("summary"))
    pretty_print_section("交接描述", escalation.get("description"))


async def interactive_chat_loop(workflow, event_store) -> None:
    """
    Very simple interactive CLI loop.

    Each user input is treated as a new ticket content.
    We call the workflow once per ticket, print the final resolution, and then
    prompt again until the user types 'exit' or 'quit'.

    This is not a streaming chat; each run exercises one ticket lifecycle.
    """
    print(
        textwrap.dedent(
            """
            UDA-Hub Interactive CLI
            -----------------------
            Type a new CultPass support issue and press Enter.
            Type 'exit' or 'quit' to stop.

            Example:
              I want help cancelling my CultPass subscription.
            """
        )
    )

    # Sample identities for local workflow demonstration.
    default_external_user_id = "a4ab87"  # update if needed
    default_user_id = "demo-user-a4ab87"
    default_account_id = "demo-account-cultpass"

    while True:
        user_text = (await asyncio.to_thread(input, "\nUser ticket> ")).strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting UDA-Hub CLI. Goodbye!")
            break

        ticket_id = f"CLI-{uuid.uuid4().hex[:8]}"
        ticket: Dict[str, Any] = {
            "content": user_text,
            "user_id": default_user_id,
            "account_id": default_account_id,
            "external_user_id": default_external_user_id,
            "user_name": "CLI User",
            "ticket_id": ticket_id,
            "new_ticket": True,
        }

        print(f"\n[Processing ticket {ticket_id}...]")
        final_state = await _handle_ticket(ticket, workflow, event_store)

        escalation = final_state.get("escalation") or {}

        print("\n--- Result ---")
        print(f"Ticket status: {final_state.get('ticket', {}).get('status')}")
        print(HANDOFF_NOTICE)


# CLI

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current UDA-Hub handoff workflow."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "chat"],
        default="demo",
        help=(
            "demo = run a single handoff example\n"
            "chat = enter tickets for handoff testing"
        ),
    )

    args = parser.parse_args()

    async with default_workflow_runtime() as (workflow, event_store):
        if args.mode == "demo":
            await run_single_ticket_demo(workflow, event_store)
        elif args.mode == "chat":
            await interactive_chat_loop(workflow, event_store)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    asyncio.run(main())
