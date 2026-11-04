from typing import Annotated

from fastapi import APIRouter, Depends

from app.db.dependencies import get_db_session
from app.service.hybrid_search import hybrid_search
from app.service.crud import get_documents
from app.service.hybrid_search import answer_question_hybrid_search
from app.service.llm import  answer_question_semantic, index_all
from sqlalchemy.ext.asyncio import AsyncSession
from app.service.llm import  index_all, initiate_client, search_chunks


DOCUMENT_ROUTER = APIRouter(prefix="/api/v1/document", tags=["document"])


@DOCUMENT_ROUTER.post("")
async def process_document(db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    await index_all(db_session)
    
@DOCUMENT_ROUTER.get("")
async def get_all_documents(db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    return await get_documents(db_session)


@DOCUMENT_ROUTER.get("/search")
async def search(
    question: str,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
):
    return await search_chunks(question, db_session, initiate_client())


@DOCUMENT_ROUTER.post("/ask/semantic")
async def ask(question: str, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    return await answer_question_semantic(question, db_session)

@DOCUMENT_ROUTER.post("/hybrid-search")
async def hybrid_search_endpoint(question: str, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    response = await hybrid_search(question=question, db_session=db_session)
    return response

@DOCUMENT_ROUTER.post("/ask/hybrid-search")
async def hybrid_search_query(question: str, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    response = await answer_question_hybrid_search(question=question, db_session=db_session)
    return response