import asyncio
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import AgentAnswer, CitationDraft
from app.assistant.tools import (
    NO_MATCHES,
    RESEARCH_TOOL_SPECS,
    ReadChunkArgs,
    ReadSurroundingChunksArgs,
    SearchFilingsArgs,
    read_chunk,
    read_surrounding_chunks,
    search_filings,
)
from app.chat.stages import RetrievalError
from app.grounding.evidence import EvidenceCandidate
from app.grounding.validator import GroundingValidator
from app.retrieval.schemas import ChunkReference, RetrievalFilters, SourcePassage


def chunk() -> ChunkReference:
    return ChunkReference(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=uuid4(),
        chunk_index=1,
        text="Services revenue increased.",
        page_number=42,
        section="Results",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="0000320193-25-000079",
        source_url="https://www.sec.gov/example",
    )


class FakeRetriever:
    def __init__(self, passages=None):
        self.filters = None
        self.retrieve_calls = 0
        self.passages = (
            passages
            if passages is not None
            else [SourcePassage(center={"chunk": chunk(), "rrf_score": 1.0})]
        )

    async def retrieve(self, query: str, *, top_k: int = 5, filters=None, **kwargs):
        assert query
        assert top_k == 5
        self.retrieve_calls += 1
        self.filters = filters
        return self.passages

    async def read_chunk(self, chunk_id):
        return chunk()

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        return {"previous_chunks": [], "next_chunks": []}


def test_agent_instructions_keep_structured_citations_out_of_answer_text():
    instructions = (Path(__file__).parents[2] / "app/assistant/instructions.md").read_text()

    assert "Do not put chunk ids or citation notation in the answer text" in instructions
    assert "Cite only evidence ids returned by the tools" in instructions
    assert "Do not cite chunk ids" in instructions
    assert "Do not reread a chunk that already has an evidence candidate" in instructions
    assert "multiple distinct searches" in instructions
    assert "comparisons, multi-part synthesis, or missing context" in instructions


def test_search_filings_caches_identical_searches():
    retriever = FakeRetriever()
    deps = agent_deps(retriever)

    first = asyncio.run(search_filings(deps, SearchFilingsArgs(query="Services revenue")))
    second = asyncio.run(search_filings(deps, SearchFilingsArgs(query="  services REVENUE  ")))

    assert first == second
    assert retriever.retrieve_calls == 1
    assert len(deps.retrieved_passages) == 1
    assert first[0].evidence_id in deps.evidence_candidates
    assert first[0].exact_quote == chunk().text
    assert retriever.filters == RetrievalFilters(tickers=("AAPL",))


def test_search_filings_allows_distinct_follow_up_searches():
    retriever = FakeRetriever()
    deps = agent_deps(retriever)

    asyncio.run(search_filings(deps, SearchFilingsArgs(query="Services revenue")))
    asyncio.run(search_filings(deps, SearchFilingsArgs(query="Product revenue")))

    assert retriever.retrieve_calls == 2


def test_surrounding_chunks_remain_readable_after_search():
    deps = agent_deps(FakeRetriever())
    asyncio.run(search_filings(deps, SearchFilingsArgs(query="Services revenue")))

    result = asyncio.run(
        read_surrounding_chunks(deps, ReadSurroundingChunksArgs(chunk_id=chunk().chunk_id))
    )

    assert result


def test_read_chunk_reuses_evidence_id_registered_by_search():
    deps = agent_deps(FakeRetriever())
    searched = asyncio.run(search_filings(deps, SearchFilingsArgs(query="Services revenue")))

    reread = asyncio.run(read_chunk(deps, ReadChunkArgs(chunk_id=chunk().chunk_id)))

    assert reread[0].evidence_id == searched[0].evidence_id
    assert len(deps.evidence_candidates) == 1


def test_search_filings_returns_terminal_message_when_no_passages_match():
    deps = agent_deps(FakeRetriever(passages=[]))

    result = asyncio.run(search_filings(deps, SearchFilingsArgs(query="No match")))

    assert (
        result
        == "No matching filing passages were found. Stop searching and report insufficient evidence."
    )


def test_read_chunk_registers_chunk_as_trusted_evidence():
    deps = agent_deps(FakeRetriever())
    deps.evidence_query = "Services revenue"
    retrieved = asyncio.run(read_chunk(deps, ReadChunkArgs(chunk_id=chunk().chunk_id)))
    answer = AgentAnswer(
        answer="Services revenue increased.",
        citations=[CitationDraft(evidence_id=retrieved[0].evidence_id)],
    )

    validated = GroundingValidator().validate(
        answer,
        deps.retrieved_passages,
        evidence_candidates=deps.evidence_candidates,
    )

    assert retrieved[0].chunk_id == chunk().chunk_id
    assert validated.citations[0].citation_label == "AAPL 10-K 2025"


def test_read_surrounding_chunks_registers_neighbors_as_trusted_evidence():
    deps = agent_deps(FakeRetriever())
    neighbor = chunk().model_copy(
        update={
            "chunk_id": UUID("00000000-0000-0000-0000-000000000002"),
            "text": "Neighbor evidence.",
        }
    )

    async def surrounding(chunk_id, *, window=1):
        return {"previous_chunks": [neighbor], "next_chunks": []}

    deps.retriever.read_surrounding_chunks = surrounding
    deps.evidence_query = "Neighbor evidence"
    result = asyncio.run(
        read_surrounding_chunks(
            deps, ReadSurroundingChunksArgs(chunk_id=chunk().chunk_id, window=1)
        )
    )
    answer = AgentAnswer(
        answer="Neighbor evidence.",
        citations=[
            CitationDraft(
                evidence_id=next(
                    candidate.evidence_id
                    for candidate in result
                    if candidate.chunk_id == neighbor.chunk_id
                )
            )
        ],
    )

    validated = GroundingValidator().validate(
        answer,
        deps.retrieved_passages,
        evidence_candidates=deps.evidence_candidates,
    )

    assert any(candidate.chunk_id == neighbor.chunk_id for candidate in result)
    assert validated.citations[0].chunk_id == neighbor.chunk_id


def agent_deps(retriever: FakeRetriever) -> DocumentAgentDeps:
    return DocumentAgentDeps(
        retriever=retriever,
        grounding_validator=GroundingValidator(),
        retrieval_filters=RetrievalFilters(tickers=("AAPL",)),
    )


def candidate(value: ChunkReference) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=value.chunk_id,
        exact_quote=value.text,
        citation_label=value.citation_label,
        location_label=value.location_label,
    )


def test_tool_specs_expose_the_four_research_tools():
    names = [spec["function"]["name"] for spec in RESEARCH_TOOL_SPECS]

    assert names == ["search_filings", "read_chunk", "read_surrounding_chunks", "submit_answer"]
    submit = RESEARCH_TOOL_SPECS[-1]["function"]["parameters"]
    assert set(submit["properties"]) == {"answer", "citations"}


def test_search_top_k_is_capped_at_ten():
    seen = {}

    class Recording(FakeRetriever):
        async def retrieve(self, query, *, top_k=5, filters=None, **kwargs):
            seen["top_k"] = top_k
            return self.passages

    asyncio.run(search_filings(agent_deps(Recording()), SearchFilingsArgs(query="x", top_k=50)))

    assert seen["top_k"] == 10


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "x", "top_k": 0}])
def test_search_rejects_invalid_arguments(args):
    with pytest.raises(ValidationError):
        SearchFilingsArgs.model_validate(args)


@pytest.mark.parametrize("window", [-1, 4])
def test_surrounding_window_is_bounded(window):
    with pytest.raises(ValidationError):
        ReadSurroundingChunksArgs(chunk_id=uuid4(), window=window)


def test_search_maps_retriever_failure_to_retrieval_error():
    class Broken(FakeRetriever):
        async def retrieve(self, query, **kwargs):
            raise RuntimeError("embedding service unavailable")

    with pytest.raises(RetrievalError):
        asyncio.run(search_filings(agent_deps(Broken()), SearchFilingsArgs(query="x")))


def test_prefetched_passages_are_used_without_calling_the_retriever():
    retriever = FakeRetriever()
    deps = agent_deps(retriever)
    prefetched = FakeRetriever().passages

    result = asyncio.run(
        search_filings(deps, SearchFilingsArgs(query="Services revenue"), prefetched=prefetched)
    )

    assert retriever.retrieve_calls == 0
    assert result[0].chunk_id == chunk().chunk_id
    again = asyncio.run(search_filings(deps, SearchFilingsArgs(query="services revenue")))
    assert again == result
    assert retriever.retrieve_calls == 0


def test_search_reports_analyzing_stage():
    stages = []

    async def report(stage):
        stages.append(stage)

    deps = agent_deps(FakeRetriever())
    deps.on_stage = report
    asyncio.run(search_filings(deps, SearchFilingsArgs(query="Services revenue")))

    assert stages == ["analyzing"]


def test_no_match_message_is_terminal():
    assert "Stop searching" in NO_MATCHES
