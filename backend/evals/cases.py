"""Eval cases: routing, grounded retrieval, and must-refuse questions.

Retrieval cases come from tests/fixtures/retrieval_eval.json and the client
brief's pilot questions in smoke_assistance.py, so evals and smoke runs agree.
"""

import json
from pathlib import Path

from smoke_assistance import SMOKE_CASES

_FIXTURE = Path(__file__).parents[1] / "tests/fixtures/retrieval_eval.json"


def _case(question: str, **expected) -> dict:
    return {"inputs": {"question": question}, "outputs": expected}


ROUTING_CASES = [
    _case("hi", route="instant"),
    _case("What can you help me do?", route="direct"),
    _case("How do citations work in this app?", route="direct"),
]

RETRIEVAL_CASES = [
    _case(item["query"], route="rag", ticker=item["ticker"], keywords=item["keywords"])
    for item in json.loads(_FIXTURE.read_text(encoding="utf-8"))
]

BRIEF_CASES = [
    _case(
        case.question,
        route="rag",
        tickers=list(case.filters.tickers),
        requires_refusal=case.requires_refusal,
    )
    for case in SMOKE_CASES
]

CASES = ROUTING_CASES + RETRIEVAL_CASES + BRIEF_CASES
