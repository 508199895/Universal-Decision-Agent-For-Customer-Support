"""
03_agentic_app.py

Entry point for exercising the current V5 handoff workflow from the command line.

This script:
- Loads environment variables (including OPENAI_API_KEY) from .env
- Imports the V5 handoff graph via `run_ticket`
- Emits structured JSON logs via logger.get_logger() to stdout and logs/uda_hub.jsonl
- Provides:
    * a simple single-ticket demo (`demo` mode)
    * an interactive CLI loop (`chat` mode) where you can type new tickets

How to run:

    # Option 1:
    python 03_agentic_app.py --mode demo

    # Option 2:
    python 03_agentic_app.py --mode chat
"""

from __future__ import annotations

import argparse
import textwrap
import uuid
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agentic.handoff import HANDOFF_NOTICE
from agentic.workflow import close_default_runtime, run_ticket
from logger import get_logger

log = get_logger()


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


def run_single_ticket_demo() -> None:
    """
    Run the current handoff-only V5 workflow for a sample ticket.

    This verifies database setup and dialogue-based handoff generation.
    """
    # NOTE: owner_id should match an external_user_id in cultpass_users.jsonl
    ticket: Dict[str, Any] = {
        "content": "Hi, I can't log in to my CultPass account and I don't get the reset email.",
        "owner_id": "a4ab87",  # update if needed
        "owner_name": "Demo User",
        "channel": "chat",
        "tags": "login, access",
        "ticket_id": f"DEMO-TICKET-{uuid.uuid4().hex[:8]}",
    }

    # Structured log: ticket received
    log.info(
        "demo_ticket_received",
        extra={
            "extra_data": {
                "ticket_id": ticket["ticket_id"],
                "owner_id": ticket["owner_id"],
                "channel": ticket["channel"],
                "tags": ticket["tags"],
            }
        },
    )

    print("\nRunning UDA-Hub workflow for a demo ticket...")
    final_state = run_ticket(ticket, thread_id=ticket["ticket_id"])

    escalation = final_state.get("escalation") or {}

    # Structured log: supervisor decision + resolution summary
    log.info(
        "demo_workflow_result",
        extra={
            "extra_data": {
                "ticket_id": ticket["ticket_id"],
                "ticket_status": final_state.get("ticket", {}).get("status"),
                "has_handoff_material": bool(escalation),
            }
        },
    )

    pretty_print_section("Ticket", ticket)
    pretty_print_section("交接摘要", escalation.get("summary"))
    pretty_print_section("交接描述", escalation.get("description"))


def interactive_chat_loop() -> None:
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

    # pick a known external_user_id from your cultpass_users.jsonl.
    default_external_user_id = "a4ab87"  # update if needed

    while True:
        user_text = input("\nUser ticket> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting UDA-Hub CLI. Goodbye!")
            break

        ticket_id = f"CLI-{uuid.uuid4().hex[:8]}"
        ticket: Dict[str, Any] = {
            "content": user_text,
            "owner_id": default_external_user_id,
            "owner_name": "CLI User",
            "channel": "chat",
            "tags": "",
            "ticket_id": ticket_id,
        }

        # Structured log: CLI ticket received
        log.info(
            "cli_ticket_received",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "owner_id": ticket["owner_id"],
                    "channel": ticket["channel"],
                }
            },
        )

        print(f"\n[Processing ticket {ticket_id}...]")
        final_state = run_ticket(ticket, thread_id=ticket_id)

        escalation = final_state.get("escalation") or {}

        # Structured log: CLI result
        log.info(
            "cli_workflow_result",
            extra={
                "extra_data": {
                    "ticket_id": ticket_id,
                    "ticket_status": final_state.get("ticket", {}).get("status"),
                    "has_handoff_material": bool(escalation),
                }
            },
        )

        print("\n--- Result ---")
        print(f"Ticket status: {final_state.get('ticket', {}).get('status')}")
        print(HANDOFF_NOTICE)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current UDA-Hub V5 handoff workflow."
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

    try:
        if args.mode == "demo":
            run_single_ticket_demo()
        elif args.mode == "chat":
            interactive_chat_loop()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    finally:
        close_default_runtime()


if __name__ == "__main__":
    main()
