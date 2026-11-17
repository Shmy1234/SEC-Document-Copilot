"""Deterministic evaluators. Each takes (outputs, expected) and returns a LangSmith
feedback dict; score None means the check does not apply to that case."""

from collections import defaultdict

REFUSAL_PHRASES = ("not enough evidence", "insufficient evidence", "does not contain enough")
_RAG = {"quick_rag", "deep_rag"}


def route_correct(outputs: dict, expected: dict) -> dict:
    want, got = expected.get("route"), outputs.get("route")
    ok = got in _RAG if want == "rag" else got == want
    return {"key": "route_correct", "score": int(ok), "comment": f"route={got}"}


def completed(outputs: dict, expected: dict) -> dict:
    error = outputs.get("error")
    return {"key": "completed", "score": int(error is None), "comment": error or "ok"}


def grounded(outputs: dict, expected: dict) -> dict:
    """A RAG answer either cites evidence or explicitly refuses; never neither."""
    if expected.get("route") != "rag" or outputs.get("error"):
        return {"key": "grounded", "score": None}
    cited = bool(outputs.get("citations"))
    refused = _refused(outputs)
    return {"key": "grounded", "score": int(cited or refused)}


def cites_expected_company(outputs: dict, expected: dict) -> dict:
    tickers = expected.get("tickers") or ([expected["ticker"]] if expected.get("ticker") else [])
    citations = outputs.get("citations") or []
    if not tickers or not citations or expected.get("requires_refusal"):
        return {"key": "cites_expected_company", "score": None}
    labels = [c["citation_label"] for c in citations]
    hit = any(label.split()[0] in tickers for label in labels)
    return {"key": "cites_expected_company", "score": int(hit), "comment": ", ".join(labels)}


def keyword_coverage(outputs: dict, expected: dict) -> dict:
    keywords = expected.get("keywords")
    if not keywords or outputs.get("error"):
        return {"key": "keyword_coverage", "score": None}
    text = " ".join(
        [outputs.get("answer") or ""] + [c["quoted_text"] for c in outputs.get("citations") or []]
    ).casefold()
    found = sum(keyword.casefold() in text for keyword in keywords)
    return {"key": "keyword_coverage", "score": found / len(keywords)}


def refusal_correct(outputs: dict, expected: dict) -> dict:
    if not expected.get("requires_refusal"):
        return {"key": "refusal_correct", "score": None}
    ok = _refused(outputs) and not outputs.get("citations")
    return {"key": "refusal_correct", "score": int(ok)}


def latency(outputs: dict, expected: dict) -> dict:
    return {"key": "latency_ms", "score": outputs.get("latency_ms")}


def _refused(outputs: dict) -> bool:
    return any(phrase in (outputs.get("answer") or "").casefold() for phrase in REFUSAL_PHRASES)


EVALUATORS = [
    route_correct,
    completed,
    grounded,
    cites_expected_company,
    keyword_coverage,
    refusal_correct,
    latency,
]


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of each applicable score, plus how many cases it applied to."""
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for score in row["scores"]:
            if score.get("score") is not None:
                totals[score["key"]].append(float(score["score"]))
    return {
        key: {"mean": round(sum(values) / len(values), 3), "cases": len(values)}
        for key, values in totals.items()
    }
