"""FastAPI entry points for the customer-support workflow and Feishu bridge."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from data.models.udahub import User as CoreUser
from agentic.capabilities.feishu import create_feishu_channel, send_feishu_text
from agentic.lifespan.runtime import default_workflow_runtime
from agentic.orchestration.orchestration import start_ticket
from agentic.orchestration.ticket.manager import ConflictError
from agentic.orchestration.ticket.request import TicketStartBody

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TicketMessageBody(BaseModel):
    """Public request fields; account identity details are loaded from the DB."""

    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, description="Existing UDA-Hub user ID")
    message: str = Field(min_length=1, description="Customer message text")


class TicketAcceptedResponse(BaseModel):
    ticket_id: str
    user_id: str
    status: str


class TicketStatusResponse(BaseModel):
    ticket_id: str
    user_id: str
    status: str
    last_ai_message: str | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep workflow, MCP, identity database, and Feishu resources app-scoped."""
    load_dotenv()
    async with AsyncExitStack() as stack:
        channel, chat_id = create_feishu_channel()
        app.state.feishu_channel = channel
        app.state.feishu_chat_id = chat_id
        app.state.feishu_connected = False
        app.state.websocket_clients = set()
        app.state.workflow_ready = False
        app.state.ticket_tasks = set()

        async def forward_feishu_message(message) -> None:
            if message.chat_id != chat_id:
                return

            payload = {
                "type": "feishu_message",
                "chat_id": message.chat_id,
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "text": message.content_text,
            }
            disconnected = []
            for websocket in tuple(app.state.websocket_clients):
                try:
                    await websocket.send_json(payload)
                except Exception:
                    disconnected.append(websocket)
            for websocket in disconnected:
                app.state.websocket_clients.discard(websocket)

        async def mark_reconnecting(_event=None) -> None:
            app.state.feishu_connected = False

        async def mark_reconnected(_event=None) -> None:
            app.state.feishu_connected = True

        async def log_channel_error(error) -> None:
            logger.error("Feishu channel error: %s", error)

        channel.on("message", forward_feishu_message)
        channel.on("reconnecting", mark_reconnecting)
        channel.on("reconnected", mark_reconnected)
        channel.on("error", log_channel_error)
        stack.push_async_callback(channel.disconnect)

        try:
            await channel.connect_until_ready(timeout=30)
            app.state.feishu_connected = True
            logger.info("Connected to Feishu; group chat_id=%s", chat_id)

            async def send_handoff(text: str) -> str:
                if not app.state.feishu_connected:
                    raise RuntimeError("Feishu connection is unavailable")
                return await send_feishu_text(channel, chat_id, text)

            runtime = await stack.enter_async_context(
                default_workflow_runtime(handoff_sender=send_handoff)
            )
            workflow, ticket_manager, ticket_store, ticket_event_store = runtime

            user_database = Path(os.getenv(
                "UDAHUB_DB_PATH",
                str(PROJECT_ROOT / "data" / "core" / "udahub.db"),
            )).expanduser().resolve()
            user_engine = create_async_engine(
                f"sqlite+aiosqlite:///{user_database.as_posix()}"
            )
            stack.push_async_callback(user_engine.dispose)
            app.state.user_session_factory = async_sessionmaker(
                user_engine, expire_on_commit=False,
            )
            app.state.workflow = workflow
            app.state.ticket_manager = ticket_manager
            app.state.ticket_store = ticket_store
            app.state.ticket_event_store = ticket_event_store
            app.state.workflow_ready = True
            yield
        finally:
            app.state.workflow_ready = False
            active_tasks = tuple(app.state.ticket_tasks)
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            app.state.feishu_connected = False


app = FastAPI(title="Customer Support API and Feishu Bridge", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool | str]:
    connected = bool(getattr(app.state, "feishu_connected", False))
    workflow_ready = bool(getattr(app.state, "workflow_ready", False))
    return {
        "status": "ok" if connected and workflow_ready else "starting",
        "feishu_connected": connected,
        "workflow_ready": workflow_ready,
    }


@app.post(
    "/tickets",
    status_code=202,
    response_model=TicketAcceptedResponse,
)
async def create_ticket(body: TicketMessageBody, request: Request) -> TicketAcceptedResponse:
    """Resolve the stored account identity and dispatch a background ticket run."""
    async with request.app.state.user_session_factory() as session:
        user = await session.scalar(
            select(CoreUser).where(CoreUser.user_id == body.user_id)
        )
    if user is None:
        raise HTTPException(status_code=404, detail="User was not found.")

    request.state.user = {
        "external_user_id": user.external_user_id,
        "user_name": user.user_name,
    }
    ticket_body = TicketStartBody(
        input={"messages": [{"role": "user", "content": body.message}]},
    )
    try:
        record = await start_ticket(ticket_body, body.user_id, request)
    except ConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    if record.task is None:
        raise RuntimeError(f"ticket {record.ticket_id} was not dispatched")
    request.app.state.ticket_tasks.add(record.task)
    record.task.add_done_callback(request.app.state.ticket_tasks.discard)
    return TicketAcceptedResponse(
        ticket_id=record.ticket_id,
        user_id=record.user_id,
        status=record.status.value,
    )


@app.get("/tickets/{ticket_id}", response_model=TicketStatusResponse)
async def get_ticket_status(
    ticket_id: str,
    request: Request,
    user_id: str = Query(min_length=1),
) -> TicketStatusResponse:
    """Return the latest workflow result only to the submitted user ID."""
    ticket = await request.app.state.ticket_store.get(ticket_id)
    if ticket is None or ticket["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Ticket was not found.")
    return TicketStatusResponse(
        ticket_id=ticket["ticket_id"],
        user_id=ticket["user_id"],
        status=ticket["status"],
        last_ai_message=ticket["last_ai_message"],
        error=ticket["error"],
    )


@app.websocket("/feishu")
async def user_connection(websocket: WebSocket) -> None:
    """Forward user text to the configured Feishu group and relay group messages."""
    await websocket.accept()
    app.state.websocket_clients.add(websocket)
    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                await websocket.send_json({"type": "error", "message": "text is required"})
                continue
            if not app.state.feishu_connected:
                await websocket.send_json(
                    {"type": "error", "message": "Feishu connection is unavailable"}
                )
                continue

            try:
                result = await app.state.feishu_channel.send(
                    app.state.feishu_chat_id,
                    {"text": text.strip()},
                )
                if not result.success:
                    raise RuntimeError("Feishu rejected the message.")
                await websocket.send_json(
                    {
                        "type": "sent",
                        "text": text.strip(),
                        "message_id": result.message_id,
                    }
                )
            except Exception:
                logger.exception("Failed to send user message to Feishu")
                await websocket.send_json(
                    {"type": "error", "message": "Failed to send message to Feishu"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        app.state.websocket_clients.discard(websocket)
