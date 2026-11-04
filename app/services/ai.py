"""Every call to the model provider goes through here: embeddings and chat."""

import json
import logging
from functools import cache
from pathlib import Path

from openai import AsyncOpenAI

from app.config import settings
from app.schemas import Answer, Usage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parents[1] / "prompts" / "chat_v1.md").read_text(
    encoding="utf-8"
)


@cache
def client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


async def embed(texts: list[str]) -> tuple[list[list[float]], str]:
    """One vector per text, in the same order, plus the model that made them."""
    response = await client().embeddings.create(
        model=settings.EMBEDDING_MODEL, input=texts
    )
    logger.info(
        "Embedding tokens",
        extra={"model": response.model, "input_tokens": response.usage.prompt_tokens},
    )
    return [item.embedding for item in response.data], response.model


async def chat(question: str, context: str) -> Answer:
    response = await client().chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    usage = Usage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
    )
    logger.info("Chat tokens", extra={"model": response.model, **usage.model_dump()})

    content = response.choices[0].message.content
    answer = (
        Answer.model_validate(json.loads(content))
        if content
        else Answer(answer="not found")
    )
    return answer.model_copy(update={"usage": usage})
