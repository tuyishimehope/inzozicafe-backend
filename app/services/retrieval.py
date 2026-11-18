"""Find the chunks that answer a question.

semantic: the nearest embeddings only.
hybrid:   three candidate lists, then one ranking:
            1. exact id    - the question names an id like "RF-3"
            2. keyword     - Postgres full-text search
            3. vector      - nearest embeddings
          Exact-id matches go first; keyword and vector are merged by rank (RRF).
"""

import re

from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.schemas import Hit, SearchStrategy
from app.services.ai import embed

CANDIDATES = 10  # how deep each candidate list goes before ranking
RRF_K = 60  # damping constant from the RRF paper
ID_PATTERN = re.compile(r"\b[A-Z]{1,3}-[A-Z0-9]+(?:\.[0-9]+)?\b")  # RF-3, D-02, L-1.2, B-KIM


async def search(
    question: str,
    db: AsyncSession,
    strategy: SearchStrategy = "hybrid",
    k: int = settings.TOP_K,
) -> list[Hit]:
    [vector] = await embed([question])

    if strategy == "semantic":
        ids = await _ids_by_vector(db, vector, k)
    else:
        ids = rank(
            exact=await _ids_by_exact_id(db, question),
            keyword=await _ids_by_keyword(db, question),
            vector=await _ids_by_vector(db, vector),
        )[:k]

    return await _load_hits(db, ids)


# ---------- ranking (pure: no database, easy to test) ----------


def rank(exact: list[int], keyword: list[int], vector: list[int]) -> list[int]:
    """Exact-id matches first, then the rest by RRF."""
    merged = rrf_merge(keyword, vector)
    return exact + [i for i in merged if i not in exact]


def rrf_merge(*ranked_lists: list[int], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + position) to a chunk's score."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + position)
    return sorted(scores, key=scores.__getitem__, reverse=True)


# ---------- candidate lists (each returns chunk ids, best first) ----------


async def _ids_by_exact_id(db: AsyncSession, question: str) -> list[int]:
    element_ids = ID_PATTERN.findall(question)
    if not element_ids:
        return []
    statement = select(Chunk.id).where(Chunk.element_id.in_(element_ids))
    return list(await db.scalars(statement))


async def _ids_by_keyword(
    db: AsyncSession, question: str, limit: int = CANDIDATES
) -> list[int]:
    # plainto_tsquery joins words with AND; swap to OR so one matching word is enough.
    words = func.plainto_tsquery("english", question).cast(String)
    query = func.to_tsquery("english", func.replace(words, "&", "|"))
    document = func.to_tsvector("english", Chunk.text)
    statement = (
        select(Chunk.id)
        .where(document.op("@@")(query))
        .order_by(func.ts_rank_cd(document, query).desc())
        .limit(limit)
    )
    return list(await db.scalars(statement))


async def _ids_by_vector(
    db: AsyncSession, vector: list[float], limit: int = CANDIDATES
) -> list[int]:
    statement = (
        select(Chunk.id).order_by(Chunk.embedding.cosine_distance(vector)).limit(limit)
    )
    return list(await db.scalars(statement))


# ---------- loading ----------


async def _load_hits(db: AsyncSession, ids: list[int]) -> list[Hit]:
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
