"""Every call to the model provider goes through here: embeddings and chat."""

import logging
from functools import cache
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import ValidationError

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
        # Log the model only: the reply may contain the user's question.
        logger.warning("Model reply was not a valid answer", extra={"model": model})
        return Answer(answer="not found")
