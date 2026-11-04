import json
import sys
from pathlib import Path

CASES = json.loads((Path(__file__).parent / "document.json").read_text())["cases"]


def selected(args: list[str] = sys.argv[1:]) -> list[dict]:
    """Cases whose id is in `args`, or all of them. Non-numeric args are ignored."""
    ids = {int(a) for a in args if a.isdigit()}
    return [c for c in CASES if not ids or c["id"] in ids]


def normalise(text: str) -> str:
    return text.lower().replace(",", "")  # "2,500" -> "2500"
