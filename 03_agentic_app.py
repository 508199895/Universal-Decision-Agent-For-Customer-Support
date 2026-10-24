"""Backward-compatible launcher for the layered CLI entry point."""

import asyncio

from agentic.app.cli import main


if __name__ == "__main__":
    asyncio.run(main())
