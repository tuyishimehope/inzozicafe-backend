"""Answer a question from the retrieved chunks, all at once or streamed.

Both paths share the same steps: build the context, ask the model, then keep only the
sources that were really sent (the citation check). Streaming changes one thing: the text
reaches the client before the sources are checked, so the checked sources come last, in
the `done` event.
"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator

import openai
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import Answer, AnswerStrategy, Hit, Source
from app.services.ai import ChatStream, chat, open_chat_stream
from app.services.chunking import DATA_DIR, PARSERS
from app.services.retrieval import search

logger = logging.getLogger(__name__)

SOURCES_MARKER = "SOURCES:"
SOURCE_PATTERN = re.compile(r"([\w.-]+\.xml)#([\w.-]+)")  # menu.xml#D-02


# ---------- shared by both paths ----------


async def _prepare(
    question: str, db: AsyncSession, strategy: AnswerStrategy
) -> tuple[list[Hit] | None, str]:
    """The chunks sent to the model (None for `full`: whole files, nothing to check) and the context."""
    if strategy == "full":
        return None, _full_context()
    hits = await search(question, db, strategy)
    return hits, _context(hits)


def check_sources(
    sources: list[Source], hits: list[Hit] | None
) -> tuple[list[Source], list[Source]]:
    """Split the model's sources into (kept, dropped): a kept source is one of the chunks we sent."""
    if hits is None:
        return sources, []
    sent = {(h.file, h.element_id) for h in hits}
    kept = [s for s in sources if (s.file, s.element_id) in sent]
    dropped = [s for s in sources if (s.file, s.element_id) not in sent]
    return kept, dropped


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


# ---------- all at once ----------


async def answer(
    question: str, db: AsyncSession, strategy: AnswerStrategy = "hybrid"
) -> Answer:
    hits, context = await _prepare(question, db, strategy)
    result = await chat(question, context)
    kept, dropped = check_sources(result.sources, hits)
    return result.model_copy(update={"sources": kept, "dropped_sources": dropped})


# ---------- streamed ----------


async def stream_answer(
    question: str, db: AsyncSession, strategy: AnswerStrategy = "hybrid"
) -> AsyncIterator[str]:
    """Do everything that can fail with a clean HTTP error *now*, then hand back the events.

    Retrieval and opening the model's stream happen before the response starts, so
    provider errors here still become 502/503. After that the status is already 200.
    """
    hits, context = await _prepare(question, db, strategy)
    # Retrieval is done: give the connection back while the model talks.
    await db.close()
    chat_stream = await open_chat_stream(question, context)
    return _events(chat_stream, hits)


async def _events(
    chat_stream: ChatStream, hits: list[Hit] | None
) -> AsyncIterator[str]:
    """Server-Sent Events: `delta` for each piece of text, then `done` (or `error`)."""
    started = time.perf_counter()
    first_piece_ms = None
    splitter = AnswerSplitter()
    try:
        async for piece in chat_stream.pieces():
            if first_piece_ms is None:
                first_piece_ms = round((time.perf_counter() - started) * 1000)
            if text := splitter.feed(piece):
                yield _event("delta", {"text": text})

        text, sources_line = splitter.finish()
        if text:
            yield _event("delta", {"text": text})

        kept, dropped = check_sources(parse_sources(sources_line), hits)
        logger.info(
            "Answer streamed",
            extra={
                "first_piece_ms": first_piece_ms,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "citations_kept": len(kept),
                "citations_dropped": len(dropped),
            },
        )
        yield _event(
            "done",
            {
                "sources": [s.model_dump() for s in kept],
                "dropped_sources": [s.model_dump() for s in dropped],
                "usage": chat_stream.usage.model_dump(),
            },
        )
    except openai.APIError as exc:
        # Too late for a 502/503: the client already has a 200, so say it in the stream.
        logger.warning(
            "AI provider error mid-stream", extra={"provider": type(exc).__name__}
        )
        yield _event(
            "error", {"detail": "The AI provider stopped while answering. Try again."}
        )
    finally:
        await chat_stream.close()  # also runs when the client disconnects


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class AnswerSplitter:
    """Separate the answer text from the final `SOURCES:` line while the reply streams in.

    The marker can arrive split across pieces ("SOUR" + "CES:"), so the last few
    characters are held back until we know they are not the start of it.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._found = False

    def feed(self, piece: str) -> str:
        """Add a piece; return the text that is safe to show now (may be empty)."""
        self._buffer += piece
        if self._found:
            return ""  # everything after the marker is the sources line
        at = self._buffer.upper().find(SOURCES_MARKER)
        if at != -1:
            text = self._buffer[:at]
            self._buffer = self._buffer[at + len(SOURCES_MARKER) :]
            self._found = True
            return text
        safe = len(self._buffer) - (len(SOURCES_MARKER) - 1)
        if safe <= 0:
            return ""
        text, self._buffer = self._buffer[:safe], self._buffer[safe:]
        return text

    def finish(self) -> tuple[str, str]:
        """At the end of the stream: (text still held back, sources line)."""
        if self._found:
            return "", self._buffer
        return self._buffer, ""


def parse_sources(line: str) -> list[Source]:
    """'menu.xml#D-02, refund-policy.xml#RF-3' -> sources. 'none' or junk -> []."""
    return [
        Source(file=file, element_id=element_id)
        for file, element_id in SOURCE_PATTERN.findall(line)
    ]
