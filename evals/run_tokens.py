"""Step 7: how many tokens does RAG save compared with sending all five files?

    docker compose exec api python -m evals.run_tokens          # every case
    docker compose exec api python -m evals.run_tokens 1 5      # only cases 1 and 5

Each case makes 2 chat calls (rag + full) and 1 embedding call. Mind the free-tier limit.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from app.db.dependencies import get_db_session
from app.service.llm import answer_question_semantic

cases = json.loads(Path("evals/document.json").read_text())["cases"]
only = {int(a) for a in sys.argv[1:]}


async def main():
    totals = {"rag": [0, 0], "full": [0, 0]} 

    async with asynccontextmanager(get_db_session)() as db:
        print(f"{'#':>3}  {'rag in':>7} {'rag out':>8}  {'full in':>8} {'full out':>9}")
        for case in cases:
            if only and case["id"] not in only:
                continue
            row = []
            for mode in ("rag", "full"):
                usage = (await answer_question_semantic(case["question"], db, mode=mode)).usage
                totals[mode][0] += usage.input_tokens
                totals[mode][1] += usage.output_tokens
                row += [usage.input_tokens, usage.output_tokens]
            print(f"{case['id']:>3}  {row[0]:>7} {row[1]:>8}  {row[2]:>8} {row[3]:>9}")

    rag, full = sum(totals["rag"]), sum(totals["full"])
    print(f"\nTotal  rag={rag}  full={full}")
    if full:
        print(f"RAG used {1 - rag / full:.0%} fewer tokens")


if __name__ == "__main__":
    asyncio.run(main())
