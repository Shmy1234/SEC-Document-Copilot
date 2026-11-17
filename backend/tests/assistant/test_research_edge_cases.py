"""Research graph boundary cases with real tools and GroundingValidator."""

import asyncio
import json
from datetime import date
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.assistant.agent import ResearchLimitExceeded, ResearchLimits, research
from app.assistant.deps import DocumentAgentDeps
from app.assistant.tools import NO_CHUNK, NO_MATCHES
from app.chat.stages import RetrievalError
from app.grounding.validator import GroundingError, GroundingValidator
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

QUESTION = "Services revenue"
FACT = "Services revenue increased 14%."
INSUFFICIENT = "The corpus does not contain enough evidence."


def chunk(index=1, text=FACT):
    return ChunkReference(
        chunk_id=UUID(int=index),
        document_id=UUID(int=1000 + index),
        chunk_index=index,
        text=text,
        page_number=index,
        section="Results",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="test-filing",
        source_url="https://example.invalid/filing",
    )


def passage(item=None, **kwargs):
    return SourcePassage(center=FusedChunk(chunk=item or chunk(), rrf_score=1.0), **kwargs)


class Retriever:
    def __init__(self, passages=None, chunks=None, surrounding=None):
        self.passages = [passage()] if passages is None else passages
        self.chunks = {chunk().chunk_id: chunk()} if chunks is None else chunks
        self.surrounding = surrounding
        self.calls = []

    async def retrieve(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return self.passages

    async def read_chunk(self, chunk_id):
        self.calls.append(("read", chunk_id))
        return self.chunks.get(chunk_id)

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        self.calls.append(("surrounding", chunk_id, window))
        return self.surrounding


class Script:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []
        self.runnable = RunnableLambda(self.invoke)

    async def invoke(self, messages):
        self.calls.append(list(messages))
        assert self.steps, "Unexpected extra model request"
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def tool(name, args, call_id="call"):
    return {"name": name, "args": args, "id": call_id}


def reply(*calls, message_id=None):
    return AIMessage(content="", tool_calls=list(calls), id=message_id)


def search(query=QUESTION, call_id="search"):
    return tool("search_filings", {"query": query}, call_id)


def records(messages):
    return [
        item
        for message in messages
        if isinstance(message, ToolMessage) and message.content.startswith("[")
        for item in json.loads(message.content)
    ]


def submit_call(messages, *, answer=FACT, ids=None, call_id="submit"):
    if ids is None:
        ids = [records(messages)[0]["evidence_id"]]
    return tool(
        "submit_answer",
        {
            "answer": answer,
            "citations": [{"evidence_id": str(item)} for item in ids],
        },
        call_id,
    )


def submit(answer=FACT, ids=None):
    return lambda messages: reply(submit_call(messages, answer=answer, ids=ids))


def make_deps(retriever=None, stages=None):
    async def report(stage):
        if stages is not None:
            stages.append(stage)

    return DocumentAgentDeps(
        retriever=retriever or Retriever(),
        grounding_validator=GroundingValidator(),
        on_stage=report,
    )


def run(model, *, deps=None, prefetched=None, limits=None):
    return asyncio.run(
        research(
            QUESTION,
            model=model.runnable,
            deps=deps or make_deps(),
            prefetched=prefetched,
            limits=limits,
        )
    )


@pytest.mark.parametrize("limit", [0, 1, 2, 6])
def test_request_budget_stops_before_request_limit_plus_one(limit):
    model = Script(*[reply(search(f"Services {i}")) for i in range(limit + 1)])
    retriever = Retriever()
    with pytest.raises(ResearchLimitExceeded, match="model request limit"):
        run(model, deps=make_deps(retriever), limits=ResearchLimits(request_limit=limit))

    assert len(model.calls) == len(retriever.calls) == limit


@pytest.mark.parametrize("limit", [1, 2, 6])
def test_answer_on_exact_last_request_is_accepted(limit):
    model = Script(*[reply(search(f"Services {i}")) for i in range(limit - 1)], submit())
    answer = run(model, prefetched=[passage()], limits=ResearchLimits(request_limit=limit))
    assert answer.answer == FACT
    assert len(model.calls) == limit


@pytest.mark.parametrize(
    ("count", "limit", "accepted"),
    [
        (0, 0, True),
        (1, 0, False),
        (1, 1, True),
        (3, 2, False),
        (3, 3, True),
        (8, 8, True),
        (9, 8, False),
    ],
)
def test_parallel_tool_calls_respect_exact_budget_before_any_batch_side_effects(
    count,
    limit,
    accepted,
):
    calls = [search(f"Services {i}", f"search-{i}") for i in range(count)]
    model = Script(lambda messages: reply(*calls, submit_call(messages)))
    retriever = Retriever()
    deps = make_deps(retriever)
    if accepted:
        answer = run(
            model, deps=deps, prefetched=[passage()], limits=ResearchLimits(tool_calls_limit=limit)
        )
        assert answer.answer == FACT
        assert len(retriever.calls) == count
    else:
        with pytest.raises(ResearchLimitExceeded, match="tool call limit"):
            run(
                model,
                deps=deps,
                prefetched=[passage()],
                limits=ResearchLimits(tool_calls_limit=limit),
            )
        assert retriever.calls == []
    # The synthetic prefetch and submit_answer do not consume the evidence-tool budget.
    assert len(model.calls) == 1


@pytest.mark.parametrize("limit", [2, 3])
def test_tool_budget_is_cumulative_across_parallel_batches_including_cached_searches(limit):
    model = Script(
        reply(search(call_id="a"), search("  SERVICES  REVENUE ", "b")),
        reply(search(call_id="c")),
        submit(),
    )
    retriever = Retriever()
    if limit == 2:
        with pytest.raises(ResearchLimitExceeded, match="tool call limit"):
            run(
                model,
                deps=make_deps(retriever),
                prefetched=[passage()],
                limits=ResearchLimits(tool_calls_limit=limit),
            )
        assert len(model.calls) == 2
    else:
        answer = run(
            model,
            deps=make_deps(retriever),
            prefetched=[passage()],
            limits=ResearchLimits(tool_calls_limit=limit),
        )
        assert answer.answer == FACT
        assert len(model.calls) == 3
    assert retriever.calls == []


@pytest.mark.parametrize("bad_count", [1, 2])
def test_grounding_gets_exactly_one_retry_and_emits_validation_for_each_attempt(bad_count):
    stages = []
    bad = submit("Services revenue increased 99%.")
    model = Script(*([bad] * bad_count), submit())
    deps = make_deps(stages=stages)
    if bad_count == 1:
        answer = run(model, deps=deps, prefetched=[passage()])
        assert answer.answer == FACT
        assert model.calls[1][-1].content.startswith("Answer rejected:")
    else:
        with pytest.raises(GroundingError, match="financial number"):
            run(model, deps=deps, prefetched=[passage()])
    assert len(model.calls) == 2
    assert stages == ["analyzing", "validating", "validating"]


def test_zero_grounding_retries_rejects_the_first_bad_answer():
    model = Script(submit("Services revenue increased 99%."), submit())
    with pytest.raises(GroundingError):
        run(model, prefetched=[passage()], limits=ResearchLimits(grounding_retries=0))
    assert len(model.calls) == 1


@pytest.mark.parametrize("submit_first", [False, True])
def test_submit_in_same_step_waits_for_evidence_tools_even_when_listed_first(submit_first):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class GatedRetriever(Retriever):
            async def read_chunk(self, chunk_id):
                started.set()
                await release.wait()
                return await super().read_chunk(chunk_id)

        stages = []
        deps = make_deps(GatedRetriever(), stages)

        def batch(messages):
            # A model can only cite ids it has already seen. Validation must
            # still wait for this batch's outstanding evidence tools.
            read = tool("read_chunk", {"chunk_id": str(chunk().chunk_id)}, "read")
            calls = [read, submit_call(messages)]
            return reply(*(list(reversed(calls)) if submit_first else calls))

        seeded = [passage()]
        model = Script(batch)
        task = asyncio.create_task(
            research(QUESTION, model=model.runnable, deps=deps, prefetched=seeded)
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert not task.done()
            assert stages == ["analyzing"]
            release.set()
            answer = await asyncio.wait_for(task, 2)
            assert answer.answer == FACT
            assert stages == ["analyzing", "validating"]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_submit_in_same_step_cannot_hide_an_evidence_tool_failure():
    class BrokenRetriever(Retriever):
        async def retrieve(self, query, **kwargs):
            raise RuntimeError("search failed")

    model = Script(lambda messages: reply(submit_call(messages), search("Services followup")))
    stages = []
    with pytest.raises(RetrievalError):
        run(model, deps=make_deps(BrokenRetriever(), stages), prefetched=[passage()])
    assert stages == ["analyzing"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("search_filings", {}),
        ("search_filings", {"query": ""}),
        ("search_filings", {"query": "Services", "top_k": 0}),
        ("read_chunk", {"chunk_id": "not-a-uuid"}),
        ("read_surrounding_chunks", {"chunk_id": str(UUID(int=1)), "window": -1}),
        ("read_surrounding_chunks", {"chunk_id": str(UUID(int=1)), "window": 4}),
    ],
)
def test_malformed_evidence_args_return_correlated_error_without_touching_retriever(name, args):
    model = Script(reply(tool(name, args, "invalid")), submit())
    retriever = Retriever()
    answer = run(model, deps=make_deps(retriever), prefetched=[passage()])
    feedback = model.calls[1][-1]
    assert isinstance(feedback, ToolMessage)
    assert feedback.tool_call_id == "invalid" and feedback.name == name
    assert feedback.content.startswith("Invalid arguments:")
    assert retriever.calls == []
    assert answer.answer == FACT


@pytest.mark.parametrize(
    "args", [{}, {"answer": 7}, {"answer": FACT, "citations": [{"evidence_id": "bad"}]}]
)
def test_malformed_submit_gets_feedback_then_can_recover(args):
    model = Script(reply(tool("submit_answer", args, "malformed")), submit())
    stages = []
    answer = run(model, deps=make_deps(stages=stages), prefetched=[passage()])
    feedback = model.calls[1][-1]
    assert feedback.tool_call_id == "malformed"
    assert feedback.content.startswith("Answer rejected: Invalid answer:")
    assert stages == ["analyzing", "validating"]
    assert answer.answer == FACT


def test_duplicate_tool_call_ids_do_not_merge_results_or_corrupt_evidence():
    model = Script(
        reply(
            search(call_id="duplicate"),
            tool(
                "read_chunk",
                {"chunk_id": str(UUID(int=99))},
                "duplicate",
            ),
        ),
        submit(),
    )
    deps = make_deps()
    answer = run(model, deps=deps, prefetched=[passage()])
    results = [
        m for m in model.calls[1] if isinstance(m, ToolMessage) and m.tool_call_id == "duplicate"
    ]
    assert len(results) == 2
    assert [m.name for m in results] == ["search_filings", "read_chunk"]
    assert results[1].content == NO_CHUNK
    assert len(deps.evidence_candidates) == 1
    assert answer.citations[0].chunk_id == chunk().chunk_id


@pytest.mark.parametrize("result", [[], [passage(chunk(text="Unrelated inventory disclosure."))]])
def test_empty_or_no_match_search_has_no_registered_evidence_and_can_refuse(result):
    retriever = Retriever(passages=result)
    deps = make_deps(retriever)
    model = Script(reply(search()), submit(INSUFFICIENT, ids=[]))
    answer = run(model, deps=deps)
    assert model.calls[1][-1].content == NO_MATCHES
    assert deps.evidence_candidates == {}
    assert answer.citations == []
    assert answer.answer == INSUFFICIENT


@pytest.mark.parametrize(
    ("surrounding", "chunks", "expected"),
    [
        (None, {}, NO_CHUNK),
        ({"previous_chunks": [], "next_chunks": []}, {}, "did not contain evidence"),
        ({"previous_chunks": [chunk(2)], "next_chunks": []}, {}, None),
        ({"previous_chunks": [], "next_chunks": []}, {UUID(int=1): chunk()}, None),
    ],
)
def test_surrounding_read_handles_missing_center_neighbors_and_result(
    surrounding, chunks, expected
):
    retriever = Retriever(passages=[], chunks=chunks, surrounding=surrounding)
    deps = make_deps(retriever)
    first = reply(tool("read_surrounding_chunks", {"chunk_id": str(UUID(int=1))}, "around"))
    model = Script(first, submit(INSUFFICIENT, ids=[]) if expected else submit())
    answer = run(model, deps=deps, prefetched=[])
    response = model.calls[1][-1]
    if expected:
        assert expected in response.content
        assert answer.citations == []
        assert deps.evidence_candidates == {}
    else:
        registered = json.loads(response.content)
        assert len(registered) == 1
        expected_id = UUID(int=2) if not chunks else UUID(int=1)
        assert answer.citations[0].chunk_id == expected_id
    assert len(retriever.calls) == (1 if surrounding is None else 2)


def test_evidence_dedupes_across_search_read_surrounding_and_parallel_tools():
    retriever = Retriever(surrounding={"previous_chunks": [chunk()], "next_chunks": [chunk()]})
    deps = make_deps(retriever)
    model = Script(
        reply(
            search("Services followup", "search"),
            tool("read_chunk", {"chunk_id": str(UUID(int=1))}, "read"),
            tool("read_surrounding_chunks", {"chunk_id": str(UUID(int=1))}, "around"),
        ),
        lambda messages: reply(
            submit_call(messages, ids=[r["evidence_id"] for r in records(messages)])
        ),
    )
    answer = run(model, deps=deps, prefetched=[passage()])
    assert len({item["evidence_id"] for item in records(model.calls[1])}) == 1
    assert len(deps.evidence_candidates) == len(deps.retrieved_passages) == 1
    assert len(answer.citations) == 1
    assert answer.citations[0].citation_index == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_citation_indices_are_contiguous_in_selected_evidence_order(reverse):
    first = chunk(1, "Services revenue increased 14%.")
    second = chunk(2, "Services revenue increased 27%.")
    items = [first, second] if not reverse else [second, first]
    text = " ".join(item.text for item in items)
    model = Script(
        lambda messages: reply(
            submit_call(
                messages,
                answer=text,
                ids=[r["evidence_id"] for r in records(messages)],
            )
        )
    )
    answer = run(model, prefetched=[passage(item) for item in items])
    assert [c.chunk_id for c in answer.citations] == [item.chunk_id for item in items]
    assert [c.citation_index for c in answer.citations] == [0, 1]
    assert [c.quoted_text for c in answer.citations] == [item.text for item in items]


@pytest.mark.parametrize("forged", [UUID(int=1), UUID(int=99999)])
def test_chunk_ids_and_fabricated_evidence_ids_fail_even_alongside_a_valid_id(forged):
    def attack(messages):
        return reply(submit_call(messages, ids=[records(messages)[0]["evidence_id"], forged]))

    model = Script(attack, attack, submit())
    with pytest.raises(GroundingError, match="not registered"):
        run(model, prefetched=[passage()])
    assert len(model.calls) == 2


def test_evidence_id_from_a_previous_run_cannot_be_cited_in_a_new_run():
    old_deps = make_deps()
    run(Script(submit()), deps=old_deps, prefetched=[passage()])
    old_id = next(iter(old_deps.evidence_candidates))
    new_deps = make_deps()
    model = Script(submit(ids=[old_id]), submit(ids=[old_id]))
    with pytest.raises(GroundingError, match="not registered"):
        run(model, deps=new_deps, prefetched=[passage()])
    assert old_id not in new_deps.evidence_candidates


def test_parallel_bad_submits_exhaust_the_single_retry_in_the_same_step():
    def two_bad(messages):
        return reply(
            *[
                submit_call(messages, answer="Services revenue increased 99%.", call_id=f"bad-{i}")
                for i in range(2)
            ]
        )

    model = Script(two_bad, submit())
    with pytest.raises(GroundingError):
        run(model, prefetched=[passage()])


def test_malformed_submits_cannot_get_more_than_one_retry():
    malformed = reply(tool("submit_answer", {}, "bad"))
    model = Script(malformed, malformed, submit())
    # Schema failures must have the same bounded repair opportunity as invalid
    # citations, rather than accepting an answer after arbitrarily many repairs.
    with pytest.raises((GroundingError, ResearchLimitExceeded, ValidationError)):
        run(model, prefetched=[passage()])
    assert len(model.calls) == 2


def test_thirteen_allowed_requests_can_finish_at_the_configured_boundary():
    model = Script(*[reply(search(f"Services {i}")) for i in range(12)], submit())
    answer = run(
        model, prefetched=[passage()], limits=ResearchLimits(request_limit=13, tool_calls_limit=12)
    )
    assert answer.answer == FACT
    assert len(model.calls) == 13


def test_reused_nonlatest_message_id_still_executes_the_newest_model_reply():
    model = Script(
        reply(search("Services first"), message_id="reused"),
        reply(search("Services second"), message_id="other"),
        lambda messages: reply(submit_call(messages), message_id="reused"),
        submit(),
    )
    retriever = Retriever()
    answer = run(model, deps=make_deps(retriever))
    assert answer.answer == FACT
    assert len(model.calls) == 3


def test_failed_parallel_tool_cancels_and_drains_its_blocked_sibling():
    async def scenario():
        baseline = asyncio.all_tasks()
        started = asyncio.Event()

        class BrokenRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                await started.wait()
                raise RuntimeError("retrieval unavailable")

            async def read_chunk(self, chunk_id):
                started.set()
                await asyncio.Event().wait()

        model = Script(
            reply(search(), tool("read_chunk", {"chunk_id": str(UUID(int=1))}, "blocked"))
        )
        try:
            with pytest.raises(RetrievalError):
                await asyncio.wait_for(
                    research(
                        QUESTION,
                        model=model.runnable,
                        deps=make_deps(BrokenRetriever()),
                    ),
                    2,
                )
            assert asyncio.all_tasks() == baseline
        finally:
            leaked = asyncio.all_tasks() - baseline
            for task in leaked:
                task.cancel()
            await asyncio.gather(*leaked, return_exceptions=True)

    asyncio.run(scenario())
