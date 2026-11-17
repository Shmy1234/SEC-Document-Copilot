"""The chat turn through its real LangGraph graph, with fakes only at the model,
retriever and store seams. The grounding validator is the production one."""

import asyncio
import json
from datetime import date
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.assistant.agent import ResearchLimitExceeded
from app.assistant.models import ChatModels
from app.assistant.outputs import AgentAnswer, CitationDraft
from app.chat.graph import TurnContext
from app.chat.routing import ModelRouteDecision
from app.chat.stages import RetrievalError
from app.chat.turn import AnswerEvent, ChatTurn, RouteEvent, StageEvent
from app.grounding.validator import GroundingError
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

QUESTION = "What happened to Services revenue?"
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


class FakeStore:
    def __init__(self):
        self.messages = []
        self.grounded_calls = []

    async def append_message(self, thread_id, role, content, message_data):
        message = {"role": role, "content": content, "message_data": message_data}
        self.messages.append(message)
        return message

    async def append_grounded_answer(self, thread_id, content, message_data, citations):
        self.grounded_calls.append((thread_id, content, message_data, citations))
        self.messages.append({"role": "assistant", "content": content, "message_data": message_data})
        return self.messages[-1]


class FakeRetriever:
    def __init__(self, failures: int = 0):
        self.calls = []
        self.failures = failures

    async def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("pgvector down")
        return passages()

    async def read_chunk(self, chunk_id):
        return chunk()

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        return {"previous_chunks": [], "next_chunks": []}


class Recorder:
    """A model seam: returns the next scripted reply and records its input."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def evidence_ids(messages) -> list[str]:
    ids = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.content.startswith("["):
            ids += [item["evidence_id"] for item in json.loads(message.content)]
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.startswith('{"question"'):
            ids += [item["evidence_id"] for item in json.loads(content)["evidence"]]
    return ids


def quick_answer(answer=FACT):
    return lambda messages: AgentAnswer(
        answer=answer, citations=[CitationDraft(evidence_id=evidence_ids(messages)[0])]
    )


def submit(answer=FACT):
    return lambda messages: AIMessage(
        content="",
        tool_calls=[
            {
                "name": "submit_answer",
                "args": {"answer": answer, "citations": [{"evidence_id": evidence_ids(messages)[0]}]},
                "id": str(uuid4()),
            }
        ],
    )


def models(route: str | None = "quick_rag", reply: str | None = None, *, quick=(), research=()):
    decision = ModelRouteDecision(route=route, answer=reply) if route else None
    return ChatModels(
        route=Recorder(*([decision] if decision else [])),
        quick_answer=Recorder(*quick),
        research=Recorder(*research),
    )


def clock():
    values = iter([1.0, 1.01, 1.03])
    return lambda: next(values)


def run(chat_models, *, store=None, retriever=None, question=QUESTION):
    store = store or FakeStore()
    retriever = retriever or FakeRetriever()
    turn = ChatTurn(
        store=store,
        context=TurnContext(models=chat_models, retriever=retriever),
        clock=clock(),
    )

    async def collect():
        return [event async for event in turn.run(thread_id=uuid4(), question=question)]

    events = asyncio.run(collect())
    return events, store, retriever


def stages(events):
    return [event.stage for event in events if isinstance(event, StageEvent)]


def answer_of(events):
    return next(event.answer for event in events if isinstance(event, AnswerEvent))


# --- routes -----------------------------------------------------------------


def test_instant_route_skips_models_and_retrieval():
    chat_models = models(route=None)

    events, store, retriever = run(chat_models, question="hi")

    assert answer_of(events).answer == "Hi! How can I help with your filing research?"
    assert chat_models.route.calls == []
    assert retriever.calls == []
    assert stages(events) == ["routing", "saving"]
    assert [m["role"] for m in store.messages] == ["user", "assistant"]


def test_direct_route_answers_from_the_routing_call():
    chat_models = models("direct", "I can explain how this workspace works.")

    events, _, _ = run(chat_models, question="How do citations work in this app?")

    assert answer_of(events).answer == "I can explain how this workspace works."
    assert answer_of(events).citations == []
    assert chat_models.quick_answer.calls == []
    assert chat_models.research.calls == []
    assert stages(events) == ["routing", "saving"]


def test_unsafe_direct_decision_is_escalated_to_quick_rag():
    chat_models = models("direct", "Apple made a lot of money.", quick=[quick_answer()])

    events, _, _ = run(chat_models, question="What was Apple Services revenue in fiscal 2025?")

    assert next(e.route for e in events if isinstance(e, RouteEvent)) == "quick_rag"
    assert answer_of(events).citations


def test_quick_rag_uses_the_prefetched_search_and_one_model_call():
    chat_models = models("quick_rag", quick=[quick_answer()])

    events, store, retriever = run(chat_models)

    assert retriever.calls == [(QUESTION, {"top_k": 5, "candidate_k": 50})]
    assert len(chat_models.quick_answer.calls) == 1
    assert chat_models.research.calls == []
    assert answer_of(events).citations[0].quoted_text == FACT
    assert stages(events) == ["routing", "searching", "analyzing", "validating", "saving"]
    assert store.grounded_calls[0][3][0]["citation_index"] == 0


def test_deep_rag_starts_from_the_prefetched_search():
    chat_models = models("deep_rag", research=[submit()])

    events, store, retriever = run(chat_models)

    assert len(retriever.calls) == 1
    assert len(chat_models.research.calls) == 1
    assert answer_of(events).citations[0].chunk_id == chunk().chunk_id
    assert stages(events) == ["routing", "searching", "analyzing", "validating", "saving"]
    assert len(store.grounded_calls[0][3]) == 1


def test_deep_rag_searches_itself_when_the_prefetch_failed():
    search = AIMessage(
        content="",
        tool_calls=[{"name": "search_filings", "args": {"query": "Services"}, "id": "s"}],
    )
    chat_models = models("deep_rag", research=[search, submit()])

    events, _, retriever = run(chat_models, retriever=FakeRetriever(failures=1))

    assert len(retriever.calls) == 2
    assert answer_of(events).citations


def test_direct_route_abandons_a_prefetch_that_failed():
    chat_models = models("direct", "I explain how this workspace works.")

    events, _, _ = run(
        chat_models,
        retriever=FakeRetriever(failures=1),
        question="How do citations work in this app?",
    )

    assert answer_of(events).answer == "I explain how this workspace works."


def test_quick_rag_prefetch_failure_is_a_typed_retrieval_failure():
    store = FakeStore()

    with pytest.raises(RetrievalError):
        run(models("quick_rag"), store=store, retriever=FakeRetriever(failures=1))

    assert store.messages[-1]["message_data"]["error_code"] == "retrieval_failed"
    assert store.messages[-1]["message_data"]["route"] == "quick_rag"


def test_routing_and_prefetch_run_concurrently():
    release = asyncio.Event()

    class WaitsForSearch:
        async def ainvoke(self, messages):
            await asyncio.wait_for(release.wait(), timeout=1)
            return ModelRouteDecision(route="quick_rag")

    class ReleasingRetriever(FakeRetriever):
        async def retrieve(self, query, **kwargs):
            release.set()
            return await super().retrieve(query, **kwargs)

    chat_models = ChatModels(
        route=WaitsForSearch(), quick_answer=Recorder(quick_answer()), research=Recorder()
    )

    events, _, _ = run(chat_models, retriever=ReleasingRetriever())

    assert answer_of(events).citations


def test_routing_failure_cancels_the_prefetch_and_records_no_route():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BrokenRoute:
        async def ainvoke(self, messages):
            await started.wait()
            raise RuntimeError("router down")

    class SlowRetriever(FakeRetriever):
        async def retrieve(self, query, **kwargs):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    store = FakeStore()
    chat_models = ChatModels(route=BrokenRoute(), quick_answer=Recorder(), research=Recorder())

    with pytest.raises(RuntimeError):
        run(chat_models, store=store, retriever=SlowRetriever())

    assert cancelled.is_set()
    assert store.messages[-1]["message_data"] == {
        "phase": 8,
        "route": None,
        "error_code": "processing_failed",
    }


# --- persistence and events ---------------------------------------------------


@pytest.mark.parametrize("route", ["direct", "quick_rag", "deep_rag"])
def test_every_route_persists_route_and_timing(route):
    chat_models = models(
        route,
        "Hello." if route == "direct" else None,
        quick=[quick_answer()],
        research=[submit()],
    )

    _, store, _ = run(chat_models, question="How does this app work?" if route == "direct" else QUESTION)

    message_data = store.grounded_calls[0][2]
    assert message_data["route"] == route
    assert message_data["routing_ms"] == 10
    assert message_data["execution_ms"] == 20
    assert message_data["citations"] == store.grounded_calls[0][3]


def test_question_is_persisted_before_the_graph_runs():
    order = []

    class OrderedStore(FakeStore):
        async def append_message(self, thread_id, role, content, message_data):
            order.append(role)
            return await super().append_message(thread_id, role, content, message_data)

    class Route(Recorder):
        async def ainvoke(self, messages):
            order.append("route")
            return ModelRouteDecision(route="direct", answer="Hello.")

    chat_models = ChatModels(route=Route(), quick_answer=Recorder(), research=Recorder())
    run(chat_models, store=OrderedStore(), question="How does this app work?")

    assert order[:2] == ["user", "route"]


def test_grounding_failure_persists_a_typed_failure_not_an_answer():
    store = FakeStore()
    chat_models = models("quick_rag", quick=[quick_answer("Services revenue fell 41%.")])

    with pytest.raises(GroundingError):
        run(chat_models, store=store)

    assert store.grounded_calls == []
    assert [m["role"] for m in store.messages] == ["user", "assistant"]
    failure = store.messages[1]
    assert failure["content"] == ""
    assert failure["message_data"]["error_code"] == "grounding_failed"


def test_research_limit_is_a_processing_failure():
    store = FakeStore()
    def loop(_messages):
        return AIMessage(
            content="",
            tool_calls=[{"name": "search_filings", "args": {"query": "x"}, "id": str(uuid4())}],
        )

    chat_models = models("deep_rag", research=[loop] * 10)

    with pytest.raises(ResearchLimitExceeded):
        run(chat_models, store=store)

    assert store.messages[-1]["message_data"]["error_code"] == "processing_failed"


def test_failure_row_write_error_does_not_mask_the_original_failure():
    class UnwritableStore(FakeStore):
        async def append_message(self, thread_id, role, content, message_data):
            if role == "assistant":
                raise RuntimeError("insert failed")
            return await super().append_message(thread_id, role, content, message_data)

    with pytest.raises(GroundingError):
        run(models("quick_rag", quick=[quick_answer("Unsupported.")]), store=UnwritableStore())


def test_route_event_precedes_execution_stages():
    events, _, _ = run(models("quick_rag", quick=[quick_answer()]))

    kinds = [
        ("route" if isinstance(e, RouteEvent) else getattr(e, "stage", "answer")) for e in events
    ]
    assert kinds.index("route") < kinds.index("searching")


def test_instant_answer_is_published_before_saving_and_rag_answers_after():
    instant, _, _ = run(models(route=None), question="thanks")
    grounded, _, _ = run(models("quick_rag", quick=[quick_answer()]))

    def order(events):
        return [
            "answer" if isinstance(e, AnswerEvent) else getattr(e, "stage", "route")
            for e in events
        ]

    assert order(instant).index("answer") < order(instant).index("saving")
    assert order(grounded)[-1] == "answer"


def test_successful_turn_logs_route_and_timings_without_the_question(monkeypatch):
    logged = []

    class FakeLogger:
        def info(self, event, **context):
            logged.append((event, context))

    monkeypatch.setattr("app.chat.turn.logger", FakeLogger())

    run(models("direct", "Hello."), question="How does this app work?")

    assert logged == [
        ("chat_turn_completed", {"route": "direct", "routing_ms": 10, "execution_ms": 20})
    ]


def test_tracing_callbacks_receive_the_graph_run():
    from langchain_core.callbacks import AsyncCallbackHandler

    class Spy(AsyncCallbackHandler):
        def __init__(self):
            self.names = []

        async def on_chain_start(self, serialized, inputs, *, name=None, **kwargs):
            self.names.append(name)

    spy = Spy()
    turn = ChatTurn(
        store=FakeStore(),
        context=TurnContext(models=models("direct", "Hello."), retriever=FakeRetriever()),
        callbacks=[spy],
        clock=clock(),
    )

    async def drain():
        return [e async for e in turn.run(thread_id=uuid4(), question="How does this app work?")]

    asyncio.run(drain())

    assert "chat_turn" in spy.names
    assert "choose_route" in spy.names
