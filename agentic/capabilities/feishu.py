"""Feishu channel construction and message delivery helpers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from lark_channel import FeishuChannel, LogLevel, PolicyConfig


def create_feishu_channel() -> tuple[Any, str]:
    """Build the configured channel and return it with the support group ID."""
    load_dotenv()
    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    chat_id = os.getenv("FEISHU_CHAT_ID")
    if not app_id or not app_secret or not chat_id:
        raise RuntimeError(
            "LARK_APP_ID, LARK_APP_SECRET, and FEISHU_CHAT_ID must be configured."
        )
    channel = FeishuChannel(
        app_id=app_id,
        app_secret=app_secret,
        log_level=LogLevel.WARNING,
        policy=PolicyConfig(
            group_policy="allowlist",
            group_allowlist=[chat_id],
            require_mention=False,
        ),
    )
    return channel, chat_id


async def send_feishu_text(channel: Any, chat_id: str, text: str) -> str:
    """Send a text message through the connected Feishu channel."""
    result = await channel.send(chat_id, {"text": text})
    if not result.success:
        raise RuntimeError("Feishu rejected the handoff message.")
    return result.message_id
