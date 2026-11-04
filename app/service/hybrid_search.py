from openai import AsyncOpenAI
from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.model import Chunk, Document

from app.core.config.settings import settings
from app.service.llm import build_context, build_full_context, embed_question, generate, initiate_client
from app.service.schema import Answer

RRF_K = 60          # damping constant from the RRF paper; rarely worth tuning
CANDIDATES = 10     # how deep each list goes before merging

import re

# Shapes from your data: RF-3, D-02, F-04, L-1.2, B-KIM, DL-B
ID_PATTERN = re.compile(r"\b[A-Z]{1,3}-[A-Z0-9]+(?:\.[0-9]+)?\b")


async def id_lookup(query: str, db_session: AsyncSession) -> list[int]:
    ids = ID_PATTERN.findall(query)
    if not ids:
        return []                                   # no id in the question → no effect
    statement = select(Chunk.id).where(Chunk.element_id.in_(ids))
    return list((await db_session.execute(statement)).scalars())

async def keyword_search(query: str, db_session: AsyncSession, limit: int = CANDIDATES):
    # AND → OR: "rule & rf-3" becomes "rule | rf-3"
    ts_query = func.to_tsquery(
        "english",
        func.replace(func.plainto_tsquery("english", query).cast(String), "&", "|"),
    )
    ts_vector = func.to_tsvector("english", Chunk.text)
    rank = func.ts_rank_cd(ts_vector, ts_query).label("rank")

    statement = (
        select(Chunk.id)
        .where(ts_vector.op("@@")(ts_query))      # only chunks that match at least one word
        .order_by(rank.desc())
        .limit(limit)
    )
    return list((await db_session.execute(statement)).scalars())


async def vector_search(vector: list[float], db_session: AsyncSession, limit: int = CANDIDATES):
    distance = Chunk.embedding.cosine_distance(vector)
    statement = select(Chunk.id).order_by(distance).limit(limit)
    return list((await db_session.execute(statement)).scalars())


def rrf_merge(*ranked_lists: list[int], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + position)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


async def hybrid_search(
    question: str, db_session: AsyncSession, k: int = settings.TOP_K
) -> list[dict]:
    client = initiate_client()
    exact_ids = await id_lookup(query=question, db_session=db_session)
    vector = await embed_question(client, question)

    keyword_ids = await keyword_search(question, db_session)
    vector_ids = await vector_search(vector, db_session)
    # top_ids = rrf_merge(keyword_ids, vector_ids)[:k]
    
    merged = rrf_merge(keyword_ids, vector_ids)
    top_ids = (exact_ids + [i for i in merged if i not in exact_ids])[:k]

    # Fetch the winners, then put them back in RRF order (SQL IN doesn't keep order).
    rows = await db_session.execute(
        select(Chunk, Document.file_name)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(top_ids))
    )
    by_id = {chunk.id: (chunk, file_name) for chunk, file_name in rows.all()}
    return [
        {
            "file": by_id[i][1],
            "element_id": by_id[i][0].element_id,
            "text": by_id[i][0].text,
            "in_keyword": i in keyword_ids,   # handy for debugging misses
            "in_vector": i in vector_ids,
        }
        for i in top_ids
    ]
    
async def answer_question_hybrid_search(question: str, db_session: AsyncSession, mode: str = "rag") -> Answer:
    client = initiate_client()
    
    if mode == "full":
        chunks: list[dict] = []
        context = build_full_context()
    else:
        chunks = await hybrid_search(question=question, db_session=db_session )
        context = build_context(chunks)

    raw, usage = await generate(client, question, context)

    # The citation check: keep only sources we actually sent (skipped in full mode: all were sent).
    if mode == "full":
        return Answer(answer=raw.answer, sources=raw.sources, usage=usage)
    sent = {(c["file"], c["element_id"]) for c in chunks}
    kept = [s for s in raw.sources if (s.file, s.element_id) in sent]
    dropped = [s for s in raw.sources if (s.file, s.element_id) not in sent]
    return Answer(answer=raw.answer, sources=kept, dropped_sources=dropped, usage=usage)