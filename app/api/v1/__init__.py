from fastapi import APIRouter

from app.api.v1 import documents, rag

router = APIRouter(prefix="/api/v1")
router.include_router(documents.router)
router.include_router(rag.router)
