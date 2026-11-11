"""Turn AI provider failures into clear HTTP errors instead of a 500 with a traceback."""

import logging

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(openai.APIError)
    async def ai_provider_error(request: Request, exc: openai.APIError) -> JSONResponse:
        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
            # Rate limit, timeout or network: the provider is unavailable for now; retrying later may work.
            status, detail = 503, "The AI provider is unavailable right now. Try again later."
        else:
            # The provider answered with an error (bad model name, bad key...): retrying won't help.
            status, detail = 502, "The AI provider rejected the request."
        logger.warning(
            "AI provider error",
            extra={"status": getattr(exc, "status_code", None), "provider": type(exc).__name__},
        )
        return JSONResponse(status_code=status, content={"detail": detail})
