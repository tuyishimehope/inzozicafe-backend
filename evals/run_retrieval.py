"""All evals in one place.

    python -m evals.run_retrieval                  # retrieval (semantic vs hybrid) + answers
    python -m evals.run_retrieval retrieval        # retrieval only: embedding calls, cheap
    python -m evals.run_retrieval answers          # answers only: chat calls, uses your limit
    python -m evals.run_retrieval answers 10 12    # only cases 10 and 12

Run inside the container: docker compose exec api python -m evals.run_retrieval
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from app.db.dependencies import get_db_session
from app.service.hybrid_search import answer_question_hybrid_search, hybrid_search
from app.service.llm import initiate_client, search_chunks

EVAL_FILE = Path("evals/document.json")
CASES = json.loads(EVAL_FILE.read_text())["cases"]


def normalise(text: str) -> str:
    return text.lower().replace(",", "")  # "2,500" -> "2500"


def pick(case_ids: set[int]) -> list[dict]:
    return [c for c in CASES if not case_ids or c["id"] in case_ids]


# ---------- 1. Retrieval: is the right chunk in the top 3? ----------

async def eval_retrieval(cases: list[dict]) -> None:
    client = initiate_client()
    searches = {
        "semantic": lambda q, db: search_chunks(q, db, client),
        "hybrid": lambda q, db: hybrid_search(q, db),
    }
    hits = {name: 0 for name in searches}
    scored = 0

    async with asynccontextmanager(get_db_session)() as db:
        print(f"{'#':>3}  {'semantic':<9} {'hybrid':<9} question")
        for case in cases:
            if not case["expected"]:  # unanswerable: checked by eval_answers
                print(f"{case['id']:>3}  {'skip':<9} {'skip':<9} {case['question']}")
                continue
            expected = {(e["file"], e["element_id"].strip()) for e in case["expected"]}
            marks = []
            for name, search in searches.items():
                got = {(r["file"], r["element_id"]) for r in await search(case["question"], db)}
                hit = expected <= got  # HIT only if EVERY expected chunk is in the top 3
                hits[name] += hit
                marks.append("HIT" if hit else "MISS")
            scored += 1
            print(f"{case['id']:>3}  {marks[0]:<9} {marks[1]:<9} {case['question']}")

    print()
    for name, n in hits.items():
        print(f"Hit rate @3 {name:<9} {n}/{scored} = {n / scored:.0%}")


# ---------- 2. Answers: does the answer contain the right words? ----------

async def eval_answers(cases: list[dict]) -> None:
    passed = 0
    async with asynccontextmanager(get_db_session)() as db:
        for case in cases:
            result = await answer_question_hybrid_search(case["question"], db)
            answer = normalise(result.answer)
            ok = all(normalise(word) in answer for word in case["must_contain"])
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] #{case['id']} {case['question']}")
            print(f"        answer={result.answer!r}")
            print(f"        sources={[(s.file, s.element_id) for s in result.sources]}"
                  f"  dropped={[(s.file, s.element_id) for s in result.dropped_sources]}")
    print(f"\nAnswer pass rate: {passed}/{len(cases)}")


async def main() -> None:
    args = sys.argv[1:]
    mode = args.pop(0) if args and not args[0].isdigit() else "all"
    cases = pick({int(a) for a in args})

    if mode in ("all", "retrieval"):
        print("=== Retrieval ===")
        await eval_retrieval(cases)
    if mode in ("all", "answers"):
        print("\n=== Answers ===")
        await eval_answers(cases)


if __name__ == "__main__":
    asyncio.run(main())
