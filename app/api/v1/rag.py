from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.db.session import DbSession
from app.schemas import MAX_QUESTION_CHARS, Answer, Hit, Question, SearchStrategy
from app.services.answering import answer, stream_answer
from app.services.retrieval import search

router = APIRouter(tags=["rag"])


@router.get("/search", response_model=list[Hit])
async def search_chunks(
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUESTION_CHARS)],
    db: DbSession,
    strategy: SearchStrategy = "hybrid",
):
    return await search(q, db, strategy)


@router.post("/answers", response_model=Answer)
async def create_answer(body: Question, db: DbSession):
    return await answer(body.question, db, body.strategy)


@router.post(
    "/answers/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def create_answer_stream(body: Question, db: DbSession) -> StreamingResponse:
    """The answer as Server-Sent Events: `delta` {text} pieces, then `done` {sources,
    dropped_sources, usage}, or `error` {detail} if the provider fails mid-answer.

    Sources are checked only after the whole reply is in, so they arrive last. If every
    source was dropped, show the answer as unverified.
    """
    events = await stream_answer(body.question, db, body.strategy)
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
