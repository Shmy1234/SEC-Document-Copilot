import asyncio
import json
from datetime import date
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.assistant.agent import (
    INSTRUCTIONS,
    PREFETCH_CALL_ID,
    ResearchLimitExceeded,
    ResearchLimits,
    research,
)
from app.assistant.deps import DocumentAgentDeps
from app.chat.stages import RetrievalError
from app.grounding.validator import GroundingError, GroundingValidator
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

FACT = "Services revenue increased 14% in fiscal 2025."


def chunk() -> ChunkReference:
    return ChunkReference(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=uuid4(),
        chunk_index=1,
        text=FACT,
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


def passages() -> list[SourcePassage]:
    return [SourcePassage(center=FusedChunk(chunk=chunk(), rrf_score=1.0))]


class FakeRetriever:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = passages() if result is None else result
        self.error = error
        self.queries = []

    async def retrieve(self, query, **kwargs):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result

    async def read_chunk(self, chunk_id):
        return chunk()

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        return {"previous_chunks": [], "next_chunks": []}


class ScriptedModel:
    """Each step is a function of the conversation so far that returns the reply."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.steps.pop(0)(messages)


def call(name, args, call_id=None):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id or str(uuid4())}]
    )


def search(query="Services revenue"):
    return lambda _messages: call("search_filings", {"query": query})


def evidence_ids(messages) -> list[str]:
    ids = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.content.startswith("["):
            ids += [item["evidence_id"] for item in json.loads(message.content)]
    return ids


def submit(answer=FACT, cite=True):
    def step(messages):
        citations = [{"evidence_id": evidence_ids(messages)[0]}] if cite else []
        return call("submit_answer", {"answer": answer, "citations": citations})

    return step


def run(model, retriever=None, *, prefetched=None, limits=None, stages=None):
    async def report(stage):
        if stages is not None:
            stages.append(stage)

    deps = DocumentAgentDeps(
        retriever=retriever or FakeRetriever(),
        grounding_validator=GroundingValidator(),
        on_stage=report,
    )
    return asyncio.run(
        research(
            "What happened to Services revenue?",
            model=model,
            deps=deps,
            prefetched=prefetched,
            limits=limits,
        )
    )


def test_prefetched_search_lets_the_model_answer_in_one_request():
    retriever = FakeRetriever()
    model = ScriptedModel(submit())

    answer = run(model, retriever, prefetched=passages())

    assert len(model.calls) == 1
    assert retriever.queries == []
    seeded = model.calls[0]
    assert isinstance(seeded[0], SystemMessage) and seeded[0].content == INSTRUCTIONS
    assert seeded[2].tool_calls[0]["id"] == PREFETCH_CALL_ID
    assert seeded[3].tool_call_id == PREFETCH_CALL_ID
    assert answer.answer == FACT
    assert answer.citations[0].chunk_id == chunk().chunk_id
    assert answer.citations[0].citation_label == "AAPL 10-K 2025"


def test_model_searches_then_submits_grounded_answer():
    retriever = FakeRetriever()
    model = ScriptedModel(search(), submit())
    stages = []

    answer = run(model, retriever, stages=stages)

    assert retriever.queries == ["Services revenue"]
    assert len(model.calls) == 2
    assert answer.citations[0].quoted_text == FACT
    assert stages == ["analyzing", "validating"]


def test_rejected_answer_is_returned_to_the_model_once_to_fix():
    model = ScriptedModel(
        search(),
        submit(answer="Services revenue increased 41% in fiscal 2025."),
        submit(),
    )

    answer = run(model)

    feedback = model.calls[2][-1]
    assert isinstance(feedback, ToolMessage)
    assert feedback.content.startswith("Answer rejected")
    assert answer.answer == FACT


def test_second_grounding_failure_fails_the_turn():
    bad = "Services revenue increased 41% in fiscal 2025."
    model = ScriptedModel(search(), submit(answer=bad), submit(answer=bad))

    with pytest.raises(GroundingError):
        run(model)


def test_uncited_factual_answer_is_rejected():
    model = ScriptedModel(search(), submit(cite=False), submit(cite=False))

    with pytest.raises(GroundingError):
        run(model)


def test_insufficient_evidence_answer_needs_no_citation():
    model = ScriptedModel(
        search(),
        submit(answer="The corpus does not contain enough evidence.", cite=False),
    )

    answer = run(model)

    assert answer.citations == []


def test_request_limit_stops_a_model_that_never_submits():
    model = ScriptedModel(*(search(f"query {n}") for n in range(10)))

    with pytest.raises(ResearchLimitExceeded):
        run(model, limits=ResearchLimits(request_limit=3, tool_calls_limit=20))

    assert len(model.calls) == 3


def test_tool_call_limit_counts_every_evidence_call():
    many = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_filings", "args": {"query": f"q{n}"}, "id": f"c{n}"}
            for n in range(5)
        ],
    )
    model = ScriptedModel(lambda _m: many)

    with pytest.raises(ResearchLimitExceeded):
        run(model, limits=ResearchLimits(tool_calls_limit=4))


def test_text_reply_without_tool_call_fails_closed():
    model = ScriptedModel(lambda _m: AIMessage(content="Services revenue went up."))

    with pytest.raises(ResearchLimitExceeded):
        run(model)


def test_invalid_tool_arguments_are_reported_back_not_raised():
    model = ScriptedModel(
        lambda _m: call("search_filings", {"query": ""}),
        search(),
        submit(),
    )

    answer = run(model)

    error = model.calls[1][-1]
    assert isinstance(error, ToolMessage) and error.content.startswith("Invalid arguments")
    assert answer.answer == FACT


def test_unknown_tool_is_reported_back():
    model = ScriptedModel(lambda _m: call("delete_everything", {}), search(), submit())

    run(model)

    assert model.calls[1][-1].content == "Unknown tool delete_everything."


def test_parallel_tool_calls_each_get_a_tool_message():
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_filings", "args": {"query": "Services"}, "id": "a"},
            {"name": "read_chunk", "args": {"chunk_id": str(chunk().chunk_id)}, "id": "b"},
        ],
    )
    model = ScriptedModel(lambda _m: both, submit())

    run(model)

    replies = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in replies} == {"a", "b"}


def test_retrieval_failure_inside_the_loop_is_typed():
    model = ScriptedModel(search())

    with pytest.raises(RetrievalError):
        run(model, FakeRetriever(error=RuntimeError("pgvector down")))


def test_reused_reply_object_still_hits_the_request_limit():
    """add_messages merges replies that share an id; the loop must still terminate."""
    same = call("search_filings", {"query": "x"}, call_id="same")
    model = ScriptedModel(*([lambda _m: same] * 10))

    with pytest.raises(ResearchLimitExceeded):
        run(model, limits=ResearchLimits(request_limit=4, tool_calls_limit=20))


def test_model_can_only_cite_evidence_registered_this_run():
    forged = lambda _m: call(
        "submit_answer", {"answer": FACT, "citations": [{"evidence_id": str(uuid4())}]}
    )
    model = ScriptedModel(search(), forged, forged)

    with pytest.raises(GroundingError):
        run(model)
