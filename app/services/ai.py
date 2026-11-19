"""Every call to the model provider goes through here: embeddings and chat."""

import logging
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import ValidationError

from app.config import settings
from app.schemas import Answer, Usage

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parents[1] / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "chat_v1.md").read_text(encoding="utf-8")
STREAM_PROMPT = (PROMPTS_DIR / "chat_stream_v1.md").read_text(encoding="utf-8")


@cache
def client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


async def embed(texts: list[str]) -> list[list[float]]:
    """One vector per text, in the same order."""
    response = await client().embeddings.create(
        model=settings.EMBEDDING_MODEL, input=texts
    )
    logger.info(
        "Embedding tokens",
        extra={"model": response.model, "input_tokens": response.usage.prompt_tokens},
    )
    return [item.embedding for item in response.data]


async def chat(question: str, context: str) -> Answer:
    response = await client().chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=_messages(SYSTEM_PROMPT, question, context),
    )
    usage = Usage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
    )
    logger.info("Chat tokens", extra={"model": response.model, **usage.model_dump()})

    answer = _parse_answer(response.choices[0].message.content, response.model)
    return answer.model_copy(update={"usage": usage})


def _parse_answer(content: str | None, model: str) -> Answer:
    """The JSON object in the reply, ignoring ```json fences or text around it.

    A reply that still isn't a valid Answer becomes "not found" with no sources: better
    no answer than one whose citations we can't trust.
    """
    text = content or ""
    start, end = text.find("{"), text.rfind("}")
    try:
        return Answer.model_validate_json(text[start : end + 1] if start != -1 else text)
    except ValidationError:
        logger.warning("Model reply was not a valid answer", extra={"model": model})
        return Answer(answer="not found")


def _messages(
    system_prompt: str, question: str, context: str
) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(
            role="user", content=f"Context:\n{context}\n\nQuestion: {question}"
        ),
    ]


class ChatStream:
    """An open streaming reply. Read it with `pieces()`; `usage` is filled in at the end."""

    def __init__(self, stream: AsyncStream[ChatCompletionChunk]) -> None:
        self._stream = stream
        self.usage = Usage()

    async def pieces(self) -> AsyncIterator[str]:
        model = ""
        async for chunk in self._stream:
            model = chunk.model or model
            if chunk.usage:  # sent once, in the last chunk, because of include_usage
                self.usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        logger.info("Chat tokens", extra={"model": model, **self.usage.model_dump()})

    async def close(self) -> None:
        """Stop the provider's reply, e.g. when the client went away: unread tokens still cost."""
        await self._stream.close()


async def open_chat_stream(question: str, context: str) -> ChatStream:
    """Start a streaming reply. Errors that happen here still become 502/503 responses,
    because no response has been sent yet."""
    stream = await client().chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        messages=_messages(STREAM_PROMPT, question, context),
    )
    return ChatStream(stream)
