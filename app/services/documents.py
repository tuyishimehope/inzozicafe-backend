import hashlib
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.ai import embed
from app.services.chunking import DATA_DIR, PARSERS


async def list_documents(db: AsyncSession) -> Sequence[Document]:
    return (await db.scalars(select(Document).order_by(Document.id))).all()


def content_hash(path: Path) -> str:
    """Fingerprint of the file's bytes: same content, same hash, every time."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def index_all(db: AsyncSession) -> dict[str, list[str]]:
    """Bring the database in line with the files: add new, re-index changed, drop removed."""
    report: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "removed": []}
    stored = {d.file_name: d for d in await db.scalars(select(Document))}

    for file_name, parse in PARSERS.items():
        path = DATA_DIR / file_name
        new_hash = content_hash(path)
        document = stored.pop(file_name, None)

        if document is not None and document.content_hash == new_hash:
            report["unchanged"].append(file_name)  # same bytes: don't pay for embeddings again
            continue

        # Embed before touching the database, so no transaction waits on the network.
        chunks = parse(path)
        vectors, model = await embed([c.text for c in chunks])
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
