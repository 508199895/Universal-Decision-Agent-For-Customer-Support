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
from types import SimpleNamespace
from typing import Any, Dict

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agentic.data.tickets.TicketEventStore import TicketEventStore
from agentic.data.tickets.TicketStore import TicketStore

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from agentic.orchestration.ticket.manager import TicketManager
from agentic.orchestration.ticket.request import TicketStartBody
from agentic.orchestration.orchestration import start_ticket
from agentic.lifespan.runtime import default_workflow_runtime
from agentic.capabilities.feishu import create_feishu_channel, send_feishu_text


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


async def _handle_ticket(
    ticket: Dict[str, Any], workflow, ticket_manager: TicketManager,
    ticket_store: TicketStore, ticket_event_store: TicketEventStore,
):
    user_id = ticket["user_id"]
    message = HumanMessage(
        content=ticket["content"],
        id=ticket.get("message_id") or f"customer:{user_id}:{uuid.uuid4()}",
    )
    body = TicketStartBody(
        input={"messages": [{
            "role": "user", "content": message.content, "id": message.id,
        }]},
        metadata=ticket.get("metadata") or {},
        config=ticket.get("config") or {},
        context={"model_name": ticket.get("model_name")} if ticket.get("model_name") else {},
        multitask_strategy=ticket.get("multitask_strategy", "reject"),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            ticket_manager=ticket_manager,
            workflow=workflow,
            ticket_store=ticket_store,
            ticket_event_store=ticket_event_store,
        )),
        state=SimpleNamespace(user=SimpleNamespace(
            external_user_id=ticket["external_user_id"],
            user_name=ticket.get("user_name"),
        )),
    )
    record = await start_ticket(body, user_id, request)
    task = record.task
    if task is None:
        raise RuntimeError(f"ticket {record.ticket_id} was not dispatched")
    return await task


async def run_single_ticket_demo(
    workflow, ticket_manager: TicketManager, ticket_store: TicketStore,
    ticket_event_store: TicketEventStore,
) -> None:
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
    }

    print("\nRunning UDA-Hub workflow for a demo ticket...")
    final_state = await _handle_ticket(
        ticket, workflow, ticket_manager, ticket_store, ticket_event_store,
    )

    escalation = final_state.get("escalation") or {}

    pretty_print_section("Ticket", ticket)
    pretty_print_section("交接摘要", escalation.get("summary"))
    pretty_print_section("交接描述", escalation.get("description"))


async def interactive_chat_loop(
    workflow, ticket_manager: TicketManager, ticket_store: TicketStore,
    ticket_event_store: TicketEventStore,
) -> None:
    """
    Very simple interactive CLI loop.

    Existing unfinished tickets are continued automatically; completed tickets
    lead to a fresh ticket on the next message.

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

        ticket: Dict[str, Any] = {
            "content": user_text,
            "user_id": default_user_id,
            "account_id": default_account_id,
            "external_user_id": default_external_user_id,
            "user_name": "CLI User",
        }

        print("\n[Processing ticket...]")
        final_state = await _handle_ticket(
            ticket, workflow, ticket_manager, ticket_store, ticket_event_store,
        )

        print("\n--- Result ---")
        print(f"Ticket status: {final_state.get('ticket', {}).get('status')}")
        messages = final_state.get("messages") or []
        if messages:
            message = messages[-1]
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if content:
                print(content)


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

    channel, chat_id = create_feishu_channel()
    try:
        await channel.connect_until_ready(timeout=30)

        async def send_handoff(text: str) -> str:
            return await send_feishu_text(channel, chat_id, text)

        async with default_workflow_runtime(handoff_sender=send_handoff) as (
            workflow, ticket_manager, ticket_store, ticket_event_store,
        ):
            if args.mode == "demo":
                await run_single_ticket_demo(
                    workflow, ticket_manager, ticket_store, ticket_event_store,
                )
            elif args.mode == "chat":
                await interactive_chat_loop(
                    workflow, ticket_manager, ticket_store, ticket_event_store,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
    finally:
        await channel.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
