import hashlib
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.services.ai import embed
from app.services.chunking import DATA_DIR, PARSERS, chunk_file


async def list_documents(db: AsyncSession) -> Sequence[Document]:
    return (await db.scalars(select(Document).order_by(Document.id))).all()


def content_hash(path: Path) -> str:
    """Fingerprint of the file's bytes: same content, same hash, every time."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def index_all(db: AsyncSession) -> dict[str, list[str]]:
    """Bring the database in line with the files: add new, re-index changed, drop removed."""
    report: dict[str, list[str]] = {
        "added": [], "updated": [], "unchanged": [], "removed": [], "missing": []
    }
    stored = {d.file_name: d for d in await db.scalars(select(Document))}
    # The model each document's vectors came from. Vectors from two models can't be compared.
    by_document = select(Chunk.document_id, func.min(Chunk.model_name)).group_by(Chunk.document_id)
    stored_models = dict((await db.execute(by_document)).all())
    model = settings.EMBEDDING_MODEL

    for file_name in PARSERS:
        path = DATA_DIR / file_name
        document = stored.pop(file_name, None)
        if not path.is_file():
            # Keep what's stored: a file that is missing for a moment shouldn't wipe its chunks.
            report["missing"].append(file_name)
            continue
        new_hash = content_hash(path)

        if (
            document is not None
            and document.content_hash == new_hash
            and stored_models.get(document.id) == model
        ):
            report["unchanged"].append(file_name)  # same bytes, same model: nothing to redo
            continue

        # Embed before touching the database, so no transaction waits on the network.
        chunks = chunk_file(path)
        vectors = await embed([c.text for c in chunks])
        rows = [
            Chunk(element_id=c.element_id, text=c.text, embedding=v, model_name=model)
            for c, v in zip(chunks, vectors, strict=True)
        ]

        if document is None:
            db.add(Document(file_name=file_name, content_hash=new_hash, chunks=rows))
            report["added"].append(file_name)
        else:
            # Replace, don't merge: old chunks may have ids or text that no longer exist.
            await db.execute(delete(Chunk).where(Chunk.document_id == document.id))
            for row in rows:
                row.document_id = document.id
            db.add_all(rows)
            document.content_hash = new_hash
            report["updated"].append(file_name)

        await db.commit()  # one file = one transaction: old chunks out and new chunks in, or neither

    for document in stored.values():  # stored, but no longer in PARSERS
        await db.delete(document)  # the foreign key's ON DELETE CASCADE removes its chunks
        report["removed"].append(document.file_name)
    await db.commit()

    return report
