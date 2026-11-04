"""How many tokens does RAG save compared with sending all five files?

    python -m evals.run_tokens          # every case
    python -m evals.run_tokens 1 5      # only cases 1 and 5

Each case makes 2 chat calls (hybrid + full) and 1 embedding call. Mind the free-tier limit.
"""

import asyncio

from app.db.session import SessionLocal
from app.services.answering import answer
from evals.cases import selected

STRATEGIES = ("hybrid", "full")


async def main() -> None:
    totals = dict.fromkeys(STRATEGIES, 0)
    async with SessionLocal() as db:
        print(f"{'#':>3}  {'rag in':>7} {'rag out':>8}  {'full in':>8} {'full out':>9}")
        for case in selected():
            row = []
            for strategy in STRATEGIES:
                usage = (await answer(case["question"], db, strategy)).usage
                totals[strategy] += usage.input_tokens + usage.output_tokens
                row += [usage.input_tokens, usage.output_tokens]
            print(f"{case['id']:>3}  {row[0]:>7} {row[1]:>8}  {row[2]:>8} {row[3]:>9}")

    rag, full = totals["hybrid"], totals["full"]
    print(f"\nTotal  rag={rag}  full={full}")
    if full:
        print(f"RAG used {1 - rag / full:.0%} fewer tokens")


if __name__ == "__main__":
    asyncio.run(main())
