"""Run the client brief's pilot questions against the configured live services."""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from uuid import uuid4

from openai import AsyncOpenAI

from app.assistant.agent import agent_usage_limits, document_agent
from app.assistant.deps import DocumentAgentDeps
from app.chat.orchestrator import AgentRunner
from app.config import settings
from app.grounding.validator import GroundingValidator
from app.retrieval.embeddings import OpenAIQueryEmbedder
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.schemas import RetrievalFilters
from ingest.repository import create_sessionmaker


@dataclass(frozen=True)
class SmokeCase:
    key: str
    question: str
    filters: RetrievalFilters
    requires_refusal: bool = False


FISCAL_YEARS = (2021, 2022, 2023, 2024, 2025)
TEN_K = ("10-K",)

SMOKE_CASES = (
    SmokeCase(
        "apple_revenue_mix",
        "Across Apple's 2021–2025 10-Ks, how did the revenue mix between iPhone, "
        "Services, Mac, iPad, and Wearables change, and which category appears to have "
        "contributed most to any mix shift?",
        RetrievalFilters(tickers=("AAPL",), filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "amazon_segment_profitability",
        "For Amazon, compare AWS operating income and margin against North America and "
        "International from 2021–2025. In which years did AWS appear to fund losses or "
        "weaker profitability elsewhere?",
        RetrievalFilters(tickers=("AMZN",), filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "nvidia_data_center",
        "How did NVIDIA describe demand drivers, customer concentration, and supply "
        "constraints for its Data Center business from fiscal 2021 through fiscal 2025?",
        RetrievalFilters(tickers=("NVDA",), filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "microsoft_cloud_capacity",
        "Across Microsoft's 2021–2025 filings, what changed in the way the company describes "
        "Azure, AI infrastructure, and cloud capacity constraints?",
        RetrievalFilters(tickers=("MSFT",), filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "alphabet_revenue_trends",
        "For Alphabet, how did Google Search, YouTube ads, Google Network, "
        "subscriptions/platforms/devices, and Google Cloud revenue trends differ across the "
        "available 10-Ks?",
        RetrievalFilters(tickers=("GOOGL",), filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "cross_company_risks",
        "Which of the five companies added, removed, or materially changed risk-factor "
        "language related to AI, cloud infrastructure, export controls, supply chain "
        "concentration, or regulation between 2021 and 2025?",
        RetrievalFilters(filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
    ),
    SmokeCase(
        "supplier_concentration",
        "For Apple and NVIDIA, what do the filings say about supplier concentration or "
        "dependence on third-party manufacturing, and did the wording become more or less "
        "urgent over time?",
        RetrievalFilters(
            tickers=("AAPL", "NVDA"), filing_types=TEN_K, fiscal_years=FISCAL_YEARS
        ),
    ),
    SmokeCase(
        "ai_cloud_investment",
        "Compare capital expenditures and purchase commitments for Microsoft, Alphabet, "
        "Amazon, and NVIDIA. What do the filings imply about the scale and timing of AI/cloud "
        "infrastructure investment?",
        RetrievalFilters(
            tickers=("MSFT", "GOOGL", "AMZN", "NVDA"),
            filing_types=TEN_K,
            fiscal_years=FISCAL_YEARS,
        ),
    ),
    SmokeCase(
        "geographic_exposure",
        "For each company, summarize the most important geographic revenue exposures "
        "disclosed in the latest 10-K, then identify any year-over-year changes that could "
        "matter to an analyst.",
        RetrievalFilters(filing_types=TEN_K, fiscal_years=(2024, 2025)),
    ),
    SmokeCase(
        "generative_ai_margins",
        "If an analyst asks whether the filings prove that generative AI improved margins for "
        "any of these companies, what evidence exists in the corpus, and where should the bot "
        "refuse to infer beyond the filings?",
        RetrievalFilters(filing_types=TEN_K, fiscal_years=FISCAL_YEARS),
        requires_refusal=True,
    ),
)


@dataclass(frozen=True)
class SmokeResult:
    key: str
    question: str
    requires_refusal: bool
    model: str
    elapsed_seconds: float
    first_stage_seconds: float | None
    answer: str
    citation_count: int
    citations: list[dict[str, object]]


def format_answer(answer) -> str:
    return json.dumps(
        {
            "answer": answer.answer,
            "citations": [citation.model_dump(mode="json") for citation in answer.citations],
        },
        indent=2,
    )


async def run_query(
    case: SmokeCase = SMOKE_CASES[0],
    *,
    agent_runner: AgentRunner | None = None,
    retriever: DocumentRetriever | None = None,
) -> SmokeResult:
    runner = agent_runner or document_agent
    active_retriever = retriever or DocumentRetriever(
        session_factory=create_sessionmaker(),
        embedder=OpenAIQueryEmbedder(AsyncOpenAI(api_key=settings.OPENAI_API_KEY)),
    )
    validator = GroundingValidator()
    started = perf_counter()
    first_stage_seconds: float | None = None

    async def record_stage(_stage: str) -> None:
        nonlocal first_stage_seconds
        if first_stage_seconds is None:
            first_stage_seconds = perf_counter() - started

    deps = DocumentAgentDeps(
        user_id=uuid4(),
        thread_id=uuid4(),
        retriever=active_retriever,
        grounding_validator=validator,
        retrieval_filters=case.filters,
        on_stage=record_stage,
    )
    result = await runner.run(
        case.question,
        deps=deps,
        usage_limits=agent_usage_limits(),
    )
    answer = validator.validate(
        result.output,
        deps.retrieved_passages,
        evidence_candidates=deps.evidence_candidates,
    )
    citations = [citation.model_dump(mode="json") for citation in answer.citations]
    return SmokeResult(
        key=case.key,
        question=case.question,
        requires_refusal=case.requires_refusal,
        model=settings.OPENAI_CHAT_MODEL,
        elapsed_seconds=round(perf_counter() - started, 3),
        first_stage_seconds=(
            round(first_stage_seconds, 3) if first_stage_seconds is not None else None
        ),
        answer=answer.answer,
        citation_count=len(citations),
        citations=citations,
    )


async def run_cases(cases: tuple[SmokeCase, ...]) -> list[SmokeResult]:
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.key}", flush=True)
        results.append(await run_query(case))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        choices=[case.key for case in SMOKE_CASES],
        help="Run one case instead of the complete ten-question brief.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path. The report is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = tuple(case for case in SMOKE_CASES if args.question in {None, case.key})
    report = {
        "model": settings.OPENAI_CHAT_MODEL,
        "question_count": len(cases),
        "results": [asdict(result) for result in asyncio.run(run_cases(cases))],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
