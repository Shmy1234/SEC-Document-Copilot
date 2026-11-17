"""Adversarial turns through the production graph, tools and grounding validator.

Events, rather than elapsed time, determine all prefetch interleavings. Task
snapshots are taken inside the running loop, before asyncio.run can clean up.
"""

import asyncio
import json
from datetime import date
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.assistant.models import ChatModels
from app.assistant.outputs import AgentAnswer, CitationDraft
from app.chat.graph import TurnContext
from app.chat.quick_rag import INSUFFICIENT_EVIDENCE_ANSWER
from app.chat.routing import ModelRouteDecision
from app.chat.stages import RetrievalError
from app.chat.turn import AnswerEvent, ChatTurn, RouteEvent, StageEvent
from app.grounding.validator import GroundingError
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

QUESTION = "What was Services revenue in 2025?"
FACT = "Services revenue increased 14%."


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


def passages(index=1, text=FACT):
    return [SourcePassage(center=FusedChunk(chunk=chunk(index, text), rrf_score=1.0))]


class Model:
    def __init__(self, reply):
        self.calls = []
        self.reply = reply
        self.runnable = RunnableLambda(self.invoke)

    async def invoke(self, messages):
        self.calls.append(list(messages))
        return self.reply(messages)


class Store:
    def __init__(self):
        self.messages = []
        self.answers = []

    async def append_message(self, thread_id, role, content, message_data):
        row = (thread_id, role, content, message_data)
        self.messages.append(row)
        return {"content": content}

    async def append_grounded_answer(self, thread_id, content, message_data, citations):
        self.answers.append((thread_id, content, message_data, citations))
        return {"content": content}


class Retriever:
    def __init__(self, result=None):
        self.result = passages() if result is None else result
        self.calls = []

    async def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.result

    async def read_chunk(self, chunk_id):
        return None

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        return None


def evidence(messages):
    result = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.content.startswith("["):
            result.extend(json.loads(message.content))
        elif isinstance(message, HumanMessage) and message.content.startswith('{"question"'):
            result.extend(json.loads(message.content)["evidence"])
    return result


def quick_reply(messages):
    item = evidence(messages)[0]
    return AgentAnswer(
        answer=item["exact_quote"], citations=[CitationDraft(evidence_id=item["evidence_id"])]
    )


def research_reply(messages):
    draft = quick_reply(messages)
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "submit_answer",
                "args": draft.model_dump(mode="json"),
                "id": "answer",
            }
        ],
    )


def make_turn(route, *, retriever=None, store=None, route_model=None, quick=None, deep=None):
    route_model = (
        route_model
        or Model(
            lambda _: ModelRouteDecision(
                route=route,
                answer="Use a company and period." if route == "direct" else None,
            )
        ).runnable
    )
    quick = quick or Model(quick_reply)
    deep = deep or Model(research_reply)
    store = store or Store()
    turn = ChatTurn(
        store=store,
        context=TurnContext(
            models=ChatModels(
                route=route_model, quick_answer=quick.runnable, research=deep.runnable
            ),
            retriever=retriever or Retriever(),
        ),
    )
    return turn, store, quick, deep


DEFAULT_THREAD = UUID(int=500)


async def collect(turn, question=QUESTION, thread_id=DEFAULT_THREAD, events=None):
    events = events if events is not None else []
    async for event in turn.run(thread_id=thread_id, question=question):
        events.append(event)
    return events


def event_names(events):
    return [
        e.stage
        if isinstance(e, StageEvent)
        else f"route:{e.route}"
        if isinstance(e, RouteEvent)
        else "answer"
        for e in events
    ]


def final_answer(events):
    answers = [e.answer for e in events if isinstance(e, AnswerEvent)]
    assert len(answers) == 1
    return answers[0]


@pytest.mark.parametrize(
    ("question", "selected", "actual"),
    [
        ("  HELLO!!!  ", "deep_rag", "instant"),
        ("What documents do you have?", "quick_rag", "instant"),
        ("How do citations work in this workspace?", "direct", "direct"),
        (QUESTION, "quick_rag", "quick_rag"),
        (QUESTION, "deep_rag", "deep_rag"),
        (QUESTION, "direct", "quick_rag"),
    ],
)
def test_every_route_transition_has_ordered_events_and_only_uses_its_models(
    question,
    selected,
    actual,
):
    router = Model(
        lambda _: ModelRouteDecision(
            route=selected,
            answer="Unsafe uncited company claim." if selected == "direct" else None,
        )
    )
    retriever = Retriever()
    turn, store, quick, deep = make_turn(selected, route_model=router.runnable, retriever=retriever)

    events = asyncio.run(collect(turn, question))

    execution = ["searching", "analyzing", "validating"] if actual.endswith("rag") else []
    tail = ["answer", "saving"] if actual == "instant" else ["saving", "answer"]
    assert event_names(events) == ["routing", f"route:{actual}", *execution, *tail]
    assert len(router.calls) == (0 if actual == "instant" else 1)
    assert len(quick.calls) == (1 if actual == "quick_rag" else 0)
    assert len(deep.calls) == (1 if actual == "deep_rag" else 0)
    assert store.answers[0][2]["route"] == actual
    answer = final_answer(events)
    if actual.endswith("rag"):
        assert answer.answer == FACT
        assert answer.citations[0].chunk_id == chunk().chunk_id
    else:
        assert answer.citations == []
    if actual == "instant":
        assert retriever.calls == []


@pytest.mark.parametrize("route", ["quick_rag", "deep_rag"])
@pytest.mark.parametrize("first", ["prefetch", "routing"])
def test_prefetch_before_and_after_routing_is_used_once_and_leaves_no_tasks(route, first):
    async def scenario():
        baseline = asyncio.all_tasks()
        started, finished, routed, release = (asyncio.Event() for _ in range(4))
        order = []

        class GatedRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                started.set()
                if first == "routing":
                    await release.wait()
                result = await super().retrieve(query, **kwargs)
                order.append("prefetch")
                finished.set()
                return result

        async def router(_messages):
            await started.wait()
            if first == "prefetch":
                await finished.wait()
            order.append("routing")
            routed.set()
            return ModelRouteDecision(route=route)

        retriever = GatedRetriever()
        turn, _, _, _ = make_turn(route, retriever=retriever, route_model=RunnableLambda(router))
        events = []
        task = asyncio.create_task(collect(turn, events=events))
        try:
            await asyncio.wait_for(routed.wait(), 2)
            if first == "routing":
                assert not task.done()
                assert not any(isinstance(e, AnswerEvent) for e in events)
                release.set()
            await asyncio.wait_for(task, 2)
            assert order == (
                ["prefetch", "routing"] if first == "prefetch" else ["routing", "prefetch"]
            )
            assert len(retriever.calls) == 1
            assert final_answer(events).citations[0].quoted_text == FACT
            assert asyncio.all_tasks() == baseline
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["direct", "error", "cancel"])
def test_abandoned_prefetch_is_cancelled_and_drained_before_turn_exits(outcome):
    async def scenario():
        baseline = asyncio.all_tasks()
        started, routing, cleaned = (asyncio.Event() for _ in range(3))
        search_tasks = []

        class BlockedRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                search_tasks.append(asyncio.current_task())
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    cleaned.set()

        async def router(_messages):
            await started.wait()
            routing.set()
            if outcome == "error":
                raise ValueError("routing failed")
            if outcome == "cancel":
                await asyncio.Event().wait()
            return ModelRouteDecision(route="direct", answer="Choose a company and period.")

        turn, store, _, _ = make_turn(
            "direct",
            retriever=BlockedRetriever(),
            route_model=RunnableLambda(router),
        )
        task = asyncio.create_task(collect(turn, "How do I use this workspace?"))
        try:
            await asyncio.wait_for(routing.wait(), 2)
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif outcome == "error":
                with pytest.raises(ValueError, match="routing failed"):
                    await task
                assert store.messages[-1][3]["route"] is None
            else:
                assert final_answer(await task).citations == []
            assert cleaned.is_set()
            assert search_tasks[0].done() and search_tasks[0].cancelled()
            assert asyncio.all_tasks() == baseline
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["quick_rag", "deep_rag"])
def test_cancelling_rag_with_pending_prefetch_leaves_no_search_task(route):
    async def scenario():
        baseline = asyncio.all_tasks()
        started, routed, cleaned = (asyncio.Event() for _ in range(3))

        class BlockedRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

        async def router(_messages):
            await started.wait()
            routed.set()
            return ModelRouteDecision(route=route)

        turn, store, _, _ = make_turn(
            route,
            retriever=BlockedRetriever(),
            route_model=RunnableLambda(router),
        )
        task = asyncio.create_task(collect(turn))
        try:
            await asyncio.wait_for(routed.wait(), 2)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleaned.is_set()
            assert store.answers == []
            assert asyncio.all_tasks() == baseline
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_direct_route_discards_an_already_completed_successful_prefetch():
    async def scenario():
        baseline = asyncio.all_tasks()
        finished = asyncio.Event()

        class FinishedRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                result = await super().retrieve(query, **kwargs)
                finished.set()
                return result

        async def router(_messages):
            await finished.wait()
            return ModelRouteDecision(route="direct", answer="Use a company and period.")

        retriever = FinishedRetriever()
        turn, store, quick, deep = make_turn(
            "direct",
            retriever=retriever,
            route_model=RunnableLambda(router),
        )
        events = await collect(turn, "How do I use this workspace?")
        assert final_answer(events).citations == []
        assert store.answers[0][3] == []
        assert len(retriever.calls) == 1
        assert quick.calls == deep.calls == []
        assert asyncio.all_tasks() == baseline

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "quick_rag", "deep_rag"])
@pytest.mark.parametrize("first", ["prefetch", "routing"])
def test_failed_prefetch_is_discarded_fails_quick_or_allows_deep_recovery(route, first):
    async def scenario():
        baseline = asyncio.all_tasks()
        started, failed, routed = (asyncio.Event() for _ in range(3))

        class FailsOnce(Retriever):
            async def retrieve(self, query, **kwargs):
                self.calls.append((query, kwargs))
                if len(self.calls) == 1:
                    started.set()
                    if first == "routing":
                        await routed.wait()
                    failed.set()
                    raise RuntimeError("prefetch unavailable")
                return self.result

        async def router(_messages):
            await started.wait()
            if first == "prefetch":
                await failed.wait()
            routed.set()
            return ModelRouteDecision(
                route=route, answer="Use this workspace." if route == "direct" else None
            )

        def recover(messages):
            if not evidence(messages):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_filings",
                            "args": {"query": QUESTION},
                            "id": "recover",
                        }
                    ],
                )
            return research_reply(messages)

        retriever = FailsOnce()
        turn, store, _, _ = make_turn(
            route,
            retriever=retriever,
            route_model=RunnableLambda(router),
            deep=Model(recover),
        )
        events = []
        question = "How do I use this workspace?" if route == "direct" else QUESTION
        if route == "quick_rag":
            with pytest.raises(RetrievalError):
                await collect(turn, question, events=events)
            assert store.answers == []
            assert store.messages[-1][3]["error_code"] == "retrieval_failed"
            assert event_names(events) == ["routing", "route:quick_rag", "searching"]
        else:
            await collect(turn, question, events=events)
            if route == "deep_rag":
                assert final_answer(events).citations
            else:
                assert final_answer(events).citations == []
        assert len(retriever.calls) == (2 if route == "deep_rag" else 1)
        assert asyncio.all_tasks() == baseline

    asyncio.run(scenario())


@pytest.mark.parametrize("result", [[], passages(text="Unrelated inventory disclosure.")])
def test_empty_and_nonmatching_prefetch_skip_quick_model_and_validation(result):
    turn, store, quick, deep = make_turn("quick_rag", retriever=Retriever(result))
    events = asyncio.run(collect(turn))

    assert final_answer(events).answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert final_answer(events).citations == []
    assert event_names(events) == ["routing", "route:quick_rag", "searching", "saving", "answer"]
    assert quick.calls == deep.calls == []
    assert store.answers[0][3] == []


@pytest.mark.parametrize("route", ["quick_rag", "deep_rag"])
def test_forged_evidence_never_reaches_answer_events_or_saved_citations(route):
    forged = str(UUID(int=999999))
    quick = Model(lambda _: AgentAnswer(answer=FACT, citations=[CitationDraft(evidence_id=forged)]))
    deep = Model(
        lambda _: AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_answer",
                    "args": {"answer": FACT, "citations": [{"evidence_id": forged}]},
                    "id": "forged",
                }
            ],
        )
    )
    turn, store, _, _ = make_turn(route, quick=quick, deep=deep)
    events = []
    with pytest.raises(GroundingError, match="not registered"):
        asyncio.run(collect(turn, events=events))

    assert not any(isinstance(e, AnswerEvent) for e in events)
    assert store.answers == []
    assert store.messages[-1][3]["error_code"] == "grounding_failed"
    assert len(quick.calls) == (1 if route == "quick_rag" else 0)
    assert len(deep.calls) == (2 if route == "deep_rag" else 0)


def test_twenty_concurrent_turns_share_graph_models_store_and_retriever_without_state_bleed():
    async def scenario():
        baseline = asyncio.all_tasks()
        all_started = asyncio.Event()
        queries = []

        class SharedRetriever(Retriever):
            async def retrieve(self, query, **kwargs):
                queries.append(query)
                if len(queries) == 20:
                    all_started.set()
                await all_started.wait()
                index = int(query.split()[1])
                return passages(index, f"Services revenue increased {index}%.")

        async def router(messages):
            index = int(messages[-1].content.split()[1])
            return ModelRouteDecision(route="quick_rag" if index % 2 else "deep_rag")

        def deep_reply(messages):
            # A repeated search must hit only this turn's prefetched cache.
            replies = [m for m in messages if isinstance(m, ToolMessage)]
            if len(replies) == 1:
                question = next(m.content for m in messages if isinstance(m, HumanMessage))
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_filings",
                            "args": {"query": question},
                            "id": "search",
                        }
                    ],
                )
            assert len({e["evidence_id"] for e in evidence(messages)}) == 1
            return research_reply(messages)

        turn, store, quick, deep = make_turn(
            "quick_rag",
            retriever=SharedRetriever(),
            route_model=RunnableLambda(router),
            deep=Model(deep_reply),
        )
        results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    collect(turn, f"Services {i} revenue in 2025?", UUID(int=100 + i))
                    for i in range(1, 21)
                ]
            ),
            3,
        )

        assert len(queries) == len(store.answers) == len(store.messages) == 20
        assert len(quick.calls) == 10 and len(deep.calls) == 20
        saved = {row[0]: row for row in store.answers}
        cited_chunks = set()
        for i, events in enumerate(results, 1):
            answer = final_answer(events)
            assert answer.answer == f"Services revenue increased {i}%."
            assert [c.chunk_id for c in answer.citations] == [UUID(int=i)]
            assert [c.citation_index for c in answer.citations] == [0]
            assert [p.center.chunk.chunk_id for p in answer.cited_passages] == [UUID(int=i)]
            cited_chunks.add(answer.citations[0].chunk_id)
            row = saved[UUID(int=100 + i)]
            assert row[1] == answer.answer
            assert row[3] == [c.model_dump(mode="json") for c in answer.citations]
            expected = "quick_rag" if i % 2 else "deep_rag"
            assert event_names(events) == [
                "routing",
                f"route:{expected}",
                "searching",
                "analyzing",
                *([] if i % 2 else ["analyzing"]),
                "validating",
                "saving",
                "answer",
            ]
        assert len(cited_chunks) == 20
        assert asyncio.all_tasks() == baseline

    asyncio.run(scenario())
