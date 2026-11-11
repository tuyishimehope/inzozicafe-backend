from typing import Annotated

from fastapi import APIRouter, Query

from app.db.session import DbSession
from app.schemas import MAX_QUESTION_CHARS, Answer, Hit, Question, SearchStrategy
from app.services.answering import answer
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
