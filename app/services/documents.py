from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.ai import embed
from app.services.chunking import DATA_DIR, PARSERS


async def list_documents(db: AsyncSession) -> Sequence[Document]:
    return (await db.scalars(select(Document).order_by(Document.id))).all()


async def index_all(db: AsyncSession) -> list[str]:
    """Parse, embed and store every file not stored yet. Returns the newly indexed names."""
    indexed = []
    for file_name, parse in PARSERS.items():
        if await db.scalar(select(Document.id).where(Document.file_name == file_name)):
            continue  # already indexed: don't pay for the embeddings again

        chunks = parse(DATA_DIR / file_name)
        vectors, model = await embed([c.text for c in chunks])
        db.add(
            Document(
                file_name=file_name,
                chunks=[
                    Chunk(element_id=c.element_id, text=c.text, embedding=v, model_name=model)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        )
        await db.commit()  # one file = one transaction: the document with all its chunks, or nothing
        indexed.append(file_name)
    return indexed
