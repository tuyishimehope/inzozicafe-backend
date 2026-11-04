import logging
import json
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.settings import settings
from app.models.model import Chunk, Document
from app.service.chunking import (
    ChunkData,
    chunk_branches,
    chunk_delivery,
    chunk_leave_policy,
    chunk_menu,
    chunk_refund_policy,
)
from app.service.crud import get_document, save_document, save_embedding
from app.service.schema import Answer, Usage

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

PARSERS = {
    "menu.xml": chunk_menu,
    "branches.xml": chunk_branches,
    "delivery.xml": chunk_delivery,
    "leave-policy.xml": chunk_leave_policy,
    "refund-policy.xml": chunk_refund_policy,
}


def initiate_client() -> AsyncOpenAI:
    # Not async: nothing is awaited here. Built once, then shared by all files.
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
    )


async def index_file(db_session: AsyncSession, client: AsyncOpenAI, file_name: str):
    # 1. Already indexed? Stop before paying for an API call.
    if await get_document(file_name, db_session) is not None:
        return

    # 2. Parse: ONE file gives ONE list of chunks.
    chunks = PARSERS[file_name](DATA_DIR / file_name)

    # 3. Embed: all the texts of this file in one call.
    response = await create_embedding(client=client, chunks=chunks)

    # 4. Save: document first (flush only), then its chunks, then one commit.
    document = await save_document(file_name=file_name, db_session=db_session)
    await save_embedding(
        chunk_data=chunks,
        embeddings=response.data,
        model_name=response.model,
        document_id=document.id,
        db_session=db_session,
    )


async def index_all(db_session: AsyncSession):
    client = initiate_client()
    for file_name in PARSERS:  # the dict already knows the file names
        await index_file(db_session=db_session, client=client, file_name=file_name)


async def create_embedding(client: AsyncOpenAI, chunks: list[ChunkData]):
    # No try/except for now: if the call fails, the error should be loud, not hidden.
    logger.info("Start embedding")
    response =  await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=[c.text for c in chunks],
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
    )   
    logger.info("Embedding tokens", extra={"model": response.model, "input_tokens": response.usage.prompt_tokens})
    return response


TOP_K = 3


async def embed_question(client: AsyncOpenAI, question: str) -> list[float]:
    # Same model as the chunks, or the coordinates live on different maps.
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=question,                       # one string → one vector
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
    )
    logger.info("Embedding tokens", extra={"model": response.model, "input_tokens": response.usage.prompt_tokens})
    return response.data[0].embedding


async def search_chunks(
    question: str,
    db_session: AsyncSession,
    client: AsyncOpenAI,
    k: int = TOP_K,
):
    vector = await embed_question(client, question)
    distance = Chunk.embedding.cosine_distance(vector).label("distance")

    statement = (
        select(Chunk, Document.file_name, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)                   # smallest distance = closest
        .limit(k)
    )
    result = await db_session.execute(statement)
    return [
        {
            "file": file_name,
            "element_id": chunk.element_id,
            "distance": round(dist, 4),
            "text": chunk.text,
        }
        for chunk, file_name, dist in result.all()
    ]




PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "chat_v1.md").read_text(encoding="utf-8")


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[file={c['file']} element_id={c['element_id']}]\n{c['text']}" for c in chunks
    )


def build_full_context() -> str:
    # The baseline: no retrieval, send every file whole.
    return "\n\n".join(
        f"[file={name}]\n{(DATA_DIR / name).read_text(encoding='utf-8')}" for name in PARSERS
    )


async def generate(client: AsyncOpenAI, question: str, context: str) -> tuple[Answer, Usage]:
    """One chat call. Same prompt shape for RAG and baseline, so only the context differs."""
    response = await client.chat.completions.create(
        model="openrouter/free",
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
    logger.info(
        "Chat tokens",
        extra={"model": response.model, "input_tokens": usage.input_tokens,
               "output_tokens": usage.output_tokens},
    )

    content = response.choices[0].message.content
    if not content:
        # Before, `raw` was never set here and the next line crashed with UnboundLocalError.
        return Answer(answer="not found", sources=[]), usage
    return Answer.model_validate(json.loads(content)), usage


async def answer_question_semantic(question: str, db_session: AsyncSession, mode: str = "rag") -> Answer:
    client = initiate_client()
    if mode == "full":
        chunks: list[dict] = []
        context = build_full_context()
    else:
        chunks = await search_chunks(question, db_session, client)
        context = build_context(chunks)

    raw, usage = await generate(client, question, context)

    # The citation check: keep only sources we actually sent (skipped in full mode: all were sent).
    if mode == "full":
        return Answer(answer=raw.answer, sources=raw.sources, usage=usage)
    sent = {(c["file"], c["element_id"]) for c in chunks}
    kept = [s for s in raw.sources if (s.file, s.element_id) in sent]
    dropped = [s for s in raw.sources if (s.file, s.element_id) not in sent]
    return Answer(answer=raw.answer, sources=kept, dropped_sources=dropped, usage=usage)

