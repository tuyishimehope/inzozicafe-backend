from fastapi import APIRouter

from app.db.session import DbSession
from app.schemas import Answer, Hit, Question, SearchStrategy
from app.services.answering import answer
from app.services.retrieval import search

router = APIRouter(tags=["rag"])


@router.get("/search", response_model=list[Hit])
async def search_chunks(q: str, db: DbSession, strategy: SearchStrategy = "hybrid"):
    return await search(q, db, strategy)


@router.post("/answers", response_model=Answer)
async def create_answer(body: Question, db: DbSession):
    return await answer(body.question, db, body.strategy)
