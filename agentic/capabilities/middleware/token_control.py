"""Shared per-ticket model-call limits and conversation context management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware


TICKET_TOKEN_LIMIT = 20_000_000
CONTEXT_SUMMARIZATION_THRESHOLD = 200_000
DEFAULT_MAX_OUTPUT_TOKENS = 8_192


class TicketTokenLimitExceeded(RuntimeError):
    """Raised before a model call when the ticket has reached its token cap."""


@dataclass(frozen=True)
class TicketRuntimeContext:
    journal: Any


class TicketTokenBudgetMiddleware(AgentMiddleware):
    """Read the current run's shared Journal snapshot before every model call."""

    def wrap_model_call(self, request, handler):
        self._check(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._check(request)
        return await handler(request)

    @staticmethod
    def _check(request) -> None:
        context = request.runtime.context
        TicketTokenBudgetMiddleware.check_context(context)

    @staticmethod
    def check_context(context) -> None:
        journal = getattr(context, "journal", None)
        if journal is None:
            raise RuntimeError("ticket token middleware requires a per-run Journal")
        if journal.total_tokens >= journal.token_limit:
            raise TicketTokenLimitExceeded(
                f"工单累计 Token 已达到上限 {journal.token_limit}，已停止自动回答。"
            )


class TicketSummarizationMiddleware(SummarizationMiddleware):
    """Check the ticket cap before the summarizer makes its own model call."""

    def before_model(self, state, runtime):
        TicketTokenBudgetMiddleware.check_context(runtime.context)
        return super().before_model(state, runtime)

    async def abefore_model(self, state, runtime):
        TicketTokenBudgetMiddleware.check_context(runtime.context)
        return await super().abefore_model(state, runtime)


def build_ticket_middleware(model):
    """Return fresh stateless middleware instances with shared ticket settings."""
    return [
        TicketTokenBudgetMiddleware(),
        TicketSummarizationMiddleware(
            model=model,
            trigger=("tokens", CONTEXT_SUMMARIZATION_THRESHOLD),
            keep=("messages", 20),
        ),
    ]
