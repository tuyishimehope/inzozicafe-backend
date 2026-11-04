from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import Answer, AnswerStrategy, Hit
from app.services.ai import chat
from app.services.chunking import DATA_DIR, PARSERS
from app.services.retrieval import search


async def answer(
    question: str, db: AsyncSession, strategy: AnswerStrategy = "hybrid"
) -> Answer:
    if strategy == "full":
        return await chat(question, _full_context())

    hits = await search(question, db, strategy)
    result = await chat(question, _context(hits))

    # Citation check: a source the model names must be one of the chunks we sent.
    sent = {(h.file, h.element_id) for h in hits}
    kept = [s for s in result.sources if (s.file, s.element_id) in sent]
    dropped = [s for s in result.sources if (s.file, s.element_id) not in sent]
    return result.model_copy(update={"sources": kept, "dropped_sources": dropped})


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[file={h.file} element_id={h.element_id}]\n{h.text}" for h in hits
    )


def _full_context() -> str:
    """The baseline: every file, whole, no retrieval."""
    return "\n\n".join(
        f"[file={name}]\n{(DATA_DIR / name).read_text(encoding='utf-8')}"
        for name in PARSERS
    )
