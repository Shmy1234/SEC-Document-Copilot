from evals.cases import CASES
from evals.scoring import (
    cites_expected_company,
    grounded,
    keyword_coverage,
    refusal_correct,
    route_correct,
    summarize,
)

CITED = {
    "route": "quick_rag",
    "answer": "Services revenue increased 14%.",
    "citations": [
        {"citation_label": "AAPL 10-K 2025", "quoted_text": "Services revenue increased"}
    ],
}
REFUSED = {
    "route": "deep_rag",
    "answer": "The corpus does not contain enough evidence.",
    "citations": [],
}


def test_cases_cover_routing_retrieval_and_refusal():
    routes = {case["outputs"]["route"] for case in CASES}
    assert routes == {"instant", "direct", "rag"}
    assert any(case["outputs"].get("requires_refusal") for case in CASES)
    assert len(CASES) == 16


def test_rag_expectation_accepts_either_rag_route():
    assert route_correct(CITED, {"route": "rag"})["score"] == 1
    assert route_correct({"route": "direct"}, {"route": "rag"})["score"] == 0


def test_grounded_requires_citations_or_explicit_refusal():
    assert grounded(CITED, {"route": "rag"})["score"] == 1
    assert grounded(REFUSED, {"route": "rag"})["score"] == 1
    assert (
        grounded(
            {"route": "quick_rag", "answer": "Apple did well.", "citations": []}, {"route": "rag"}
        )["score"]
        == 0
    )
    assert grounded(CITED, {"route": "direct"})["score"] is None


def test_company_and_keyword_checks():
    assert cites_expected_company(CITED, {"ticker": "AAPL"})["score"] == 1
    assert cites_expected_company(CITED, {"tickers": ["MSFT"]})["score"] == 0
    assert keyword_coverage(CITED, {"keywords": ["services", "margin"]})["score"] == 0.5


def test_refusal_case_must_not_cite():
    assert refusal_correct(REFUSED, {"requires_refusal": True})["score"] == 1
    assert refusal_correct(CITED, {"requires_refusal": True})["score"] == 0


def test_summary_averages_only_applicable_scores():
    rows = [
        {"scores": [{"key": "grounded", "score": 1}, {"key": "x", "score": None}]},
        {"scores": [{"key": "grounded", "score": 0}]},
    ]
    assert summarize(rows) == {"grounded": {"mean": 0.5, "cases": 2}}
