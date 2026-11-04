from fastapi import APIRouter

from app.db.session import DbSession
from app.schemas import DocumentOut
from app.services import documents

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: DbSession):
    return await documents.list_documents(db)


@router.post("/index")
async def index_documents(db: DbSession) -> dict[str, list[str]]:
    return {"indexed": await documents.index_all(db)}
