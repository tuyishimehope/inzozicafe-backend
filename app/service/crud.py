from openai.types import Embedding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model import Chunk, Document
from app.service.chunking import ChunkData


async def get_document(file_name: str, db_session: AsyncSession) -> Document | None:
    statement = select(Document).where(Document.file_name == file_name)
    result = await db_session.execute(statement)
    return result.scalar_one_or_none()  # the Document itself, not just its id

async def get_documents(db_session: AsyncSession):
    statement = select(Document)
    result = await db_session.execute(statement)
    return result.scalars().all()

async def save_document(file_name: str, db_session: AsyncSession) -> Document:
    document = Document(file_name=file_name)
    db_session.add(document)
    await db_session.flush()  # fills document.id, but nothing is final until commit
    return document

async def save_embedding(
    chunk_data: list[ChunkData],
    embeddings: list[Embedding],
    model_name: str,
    document_id: int,
    db_session: AsyncSession,
):
    # strict=True: if the API returned a different number of vectors, fail here.
    rows = [
        Chunk(
            element_id=chunk.element_id,
            text=chunk.text,
            embedding=item.embedding,  # THIS chunk's own vector
            model_name=model_name,
            document_id=document_id,
        )
        for chunk, item in zip(chunk_data, embeddings, strict=True)
    ]
    db_session.add_all(rows)
    await db_session.commit()  # document + all chunks together, or nothing
    return rows
