"""Find the chunks that answer a question.

semantic: nearest embeddings only.
hybrid:   exact id match first, then keyword and vector results merged by rank (RRF).
"""

import re

from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.schemas import Hit, SearchStrategy
from app.services.ai import embed

CANDIDATES = 10  # how deep each list goes before merging
RRF_K = 60  # damping constant from the RRF paper
ID_PATTERN = re.compile(r"\b[A-Z]{1,3}-[A-Z0-9]+(?:\.[0-9]+)?\b")  # RF-3, D-02, L-1.2, B-KIM


async def search(
    question: str, db: AsyncSession, strategy: SearchStrategy = "hybrid", k: int = settings.TOP_K
) -> list[Hit]:
    [vector], _ = await embed([question])
    if strategy == "semantic":
        return await _hits(db, await _vector_ids(db, vector, k))

    exact = await _id_ids(db, question)
    merged = rrf_merge(await _keyword_ids(db, question), await _vector_ids(db, vector))
    return await _hits(db, (exact + [i for i in merged if i not in exact])[:k])


def rrf_merge(*ranked_lists: list[int], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + position)
    return sorted(scores, key=scores.__getitem__, reverse=True)


async def _id_ids(db: AsyncSession, question: str) -> list[int]:
    ids = ID_PATTERN.findall(question)
    if not ids:
        return []
    return list(await db.scalars(select(Chunk.id).where(Chunk.element_id.in_(ids))))


async def _keyword_ids(db: AsyncSession, question: str, limit: int = CANDIDATES) -> list[int]:
    # plainto_tsquery joins words with AND; swap to OR so one matching word is enough.
    query = func.to_tsquery(
        "english", func.replace(func.plainto_tsquery("english", question).cast(String), "&", "|")
    )
    document = func.to_tsvector("english", Chunk.text)
    statement = (
        select(Chunk.id)
        .where(document.op("@@")(query))
        .order_by(func.ts_rank_cd(document, query).desc())
        .limit(limit)
    )
    return list(await db.scalars(statement))


async def _vector_ids(db: AsyncSession, vector: list[float], limit: int = CANDIDATES) -> list[int]:
    statement = select(Chunk.id).order_by(Chunk.embedding.cosine_distance(vector)).limit(limit)
    return list(await db.scalars(statement))


async def _hits(db: AsyncSession, ids: list[int]) -> list[Hit]:
    """Load the chunks, keeping the order of `ids` (SQL IN does not)."""
    rows = await db.execute(
        select(Chunk, Document.file_name)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(ids))
    )
    by_id = {
        chunk.id: Hit(file=file_name, element_id=chunk.element_id, text=chunk.text)
        for chunk, file_name in rows.all()
    }
    return [by_id[i] for i in ids]
