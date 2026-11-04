"""Retrieval and answer evals.

    python -m evals.run_retrieval                  # retrieval (semantic vs hybrid) + answers
    python -m evals.run_retrieval retrieval        # retrieval only: embedding calls, cheap
    python -m evals.run_retrieval answers 10 12    # answers for cases 10 and 12

Run inside the container: docker compose exec api python -m evals.run_retrieval
"""

import asyncio
import sys

from app.db.session import SessionLocal
from app.services.answering import answer
from app.services.retrieval import search
from evals.cases import normalise, selected

STRATEGIES = ("semantic", "hybrid")


async def eval_retrieval(cases: list[dict]) -> None:
    hits = dict.fromkeys(STRATEGIES, 0)
    scored = 0
    async with SessionLocal() as db:
        print(f"{'#':>3}  {'semantic':<9} {'hybrid':<9} question")
        for case in cases:
            if not case["expected"]:  # unanswerable: checked by eval_answers
                print(f"{case['id']:>3}  {'skip':<9} {'skip':<9} {case['question']}")
                continue
            expected = {(e["file"], e["element_id"]) for e in case["expected"]}
            marks = []
            for strategy in STRATEGIES:
                got = {
                    (h.file, h.element_id)
                    for h in await search(case["question"], db, strategy)
                }
                hit = expected <= got  # every expected chunk is in the top k
                hits[strategy] += hit
                marks.append("HIT" if hit else "MISS")
            scored += 1
            print(f"{case['id']:>3}  {marks[0]:<9} {marks[1]:<9} {case['question']}")

    print()
    for strategy, n in hits.items():
        print(f"Hit rate @k {strategy:<9} {n}/{scored} = {n / scored:.0%}")


async def eval_answers(cases: list[dict]) -> None:
    passed = 0
    async with SessionLocal() as db:
        for case in cases:
            result = await answer(case["question"], db)
            ok = all(
                normalise(w) in normalise(result.answer) for w in case["must_contain"]
            )
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] #{case['id']} {case['question']}")
            print(f"        answer={result.answer!r}")
            print(
                f"        sources={[(s.file, s.element_id) for s in result.sources]}"
                f"  dropped={[(s.file, s.element_id) for s in result.dropped_sources]}"
            )
    print(f"\nAnswer pass rate: {passed}/{len(cases)}")


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "all"
    cases = selected()
    if mode in ("all", "retrieval"):
        print("=== Retrieval ===")
        await eval_retrieval(cases)
    if mode in ("all", "answers"):
        print("\n=== Answers ===")
        await eval_answers(cases)


if __name__ == "__main__":
    asyncio.run(main())
