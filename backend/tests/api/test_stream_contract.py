"""HTTP contracts over the real turn/research graphs, tools and grounding validator."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from structlog.testing import capture_logs

from app.api.chat import get_chat_store, get_document_retriever, get_resources
from app.assistant.models import ChatModels
from app.assistant.outputs import AgentAnswer, CitationDraft
from app.auth.dependencies import AuthenticatedUser, get_current_user
from app.chat.routing import ModelRouteDecision
from app.chat.streaming import _running_turns
from app.database.chats import ForbiddenThreadError, ThreadNotFoundError
from app.main import app
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

THREAD = UUID("00000000-0000-0000-0000-000000000010")
USER = AuthenticatedUser(id=UUID("00000000-0000-0000-0000-000000000020"))
CHUNK = UUID("00000000-0000-0000-0000-000000000030")
QUESTION = "What happened to Services revenue?"
FACT = "Services revenue increased 14% in fiscal 2025."
SECRET = "private upstream exception: password=do-not-stream"
DIRECT = "I can help you explore this workspace. " * 5 + "Use a company and period. café 📚\n"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def passage():
    return SourcePassage(
        center=FusedChunk(
            chunk=ChunkReference(
                chunk_id=CHUNK,
                document_id=UUID("00000000-0000-0000-0000-000000000040"),
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
            ),
            rrf_score=1.0,
        )
    )


class Store:
    def __init__(self):
        self.rows = []
        self.access_error = None
        self.write_error = False
        self.save_attempts = 0
        self.save_gate = None
        self.saved = asyncio.Event()

    async def get_thread(self, user_id, thread_id):
        assert (user_id, thread_id) == (USER.id, THREAD)
        if self.access_error:
            raise self.access_error
        return {"id": str(THREAD), "user_id": str(USER.id)}

    async def append_message(self, thread_id, role, content, message_data):
        assert thread_id == THREAD
        row = {"role": role, "content": content, "message_data": message_data}
        self.rows.append(row)
        return row

    async def append_grounded_answer(self, thread_id, content, message_data, citations):
        self.save_attempts += 1
        if self.save_gate is not None:
            await self.save_gate.wait()
        if self.write_error:
            raise RuntimeError(SECRET)
        assert citations == message_data["citations"]
        row = await self.append_message(thread_id, "assistant", content, message_data)
        self.saved.set()
        return row


class Retriever:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.empty = False

    async def retrieve(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        if self.fail == "search":
            raise RuntimeError(SECRET)
        return [] if self.empty else [passage()]

    async def read_chunk(self, chunk_id):
        self.calls.append(("read_chunk", chunk_id))
        if self.fail == "read_chunk":
            raise RuntimeError(SECRET)
        return passage().center.chunk

    async def read_surrounding_chunks(self, chunk_id, *, window=1):
        self.calls.append(("read_surrounding_chunks", chunk_id))
        if self.fail == "read_surrounding_chunks":
            raise RuntimeError(SECRET)
        return {"previous_chunks": [], "next_chunks": []}


def evidence_id(messages):
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.content.startswith("["):
            return json.loads(message.content)[0]["evidence_id"]
        if isinstance(message.content, str) and message.content.startswith('{"question"'):
            return json.loads(message.content)["evidence"][0]["evidence_id"]
    raise AssertionError("model was not supplied real registered evidence")


def draft(messages):
    return AgentAnswer(answer=FACT, citations=[CitationDraft(evidence_id=evidence_id(messages))])


def tool(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


def submit(messages):
    return tool("submit_answer", draft(messages).model_dump(mode="json"))


class Harness:
    def __init__(self):
        self.store = Store()
        self.retriever = Retriever()
        self.route = "quick_rag"
        self.route_reply = None
        self.quick_reply = draft
        self.research_reply = submit
        self.model_calls = {"route": [], "quick": [], "research": []}
        self.models = ChatModels(
            route=self.runnable("route"),
            quick_answer=self.runnable("quick"),
            research=self.runnable("research"),
        )

    def runnable(self, role):
        async def invoke(messages):
            self.model_calls[role].append(messages)
            if role == "route":
                if self.route_reply:
                    return await self.route_reply(messages)
                return ModelRouteDecision(
                    route=self.route, answer=DIRECT if self.route == "direct" else None
                )
            return (self.quick_reply if role == "quick" else self.research_reply)(messages)

        return RunnableLambda(invoke)

    async def post(self, question=QUESTION, **body):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/chat/stream", json=payload(question, **body))


@pytest.fixture
def harness():
    h = Harness()
    previous = app.dependency_overrides.copy()

    async def current_user():
        return USER

    async def resources():
        return SimpleNamespace(models=h.models, callbacks=[])

    async def store():
        return h.store

    async def retriever():
        return h.retriever

    app.dependency_overrides.update(
        {
            get_current_user: current_user,
            get_resources: resources,
            get_chat_store: store,
            get_document_retriever: retriever,
        }
    )
    try:
        # Rendering nested graph tracebacks dominates failure-test runtime.
        with capture_logs():
            yield h
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def payload(question=QUESTION, **body):
    return {
        "threadId": str(THREAD),
        "messages": [{"role": "user", "content": question}],
        **body,
    }


def parse_sse(wire):
    """Require complete SSE frames and decode each payload, including escaped text."""
    assert wire.endswith("\n\n")
    frames = wire[:-2].split("\n\n")
    events = []
    for frame in frames:
        assert frame.startswith("data: ")
        assert "\n" not in frame
        event = json.loads(frame[6:])
        assert isinstance(event, dict) and "type" in event
        events.append(event)
    return events


def status(stage):
    return {"type": "data-status", "data": {"stage": stage}}


def answer_events(text, citations):
    return [
        {"type": "text-start", "id": "text-1"},
        *[
            {"type": "text-delta", "id": "text-1", "delta": text[i : i + 80]}
            for i in range(0, len(text), 80)
        ],
        {"type": "text-end", "id": "text-1"},
        {"type": "data-citations", "data": citations},
    ]


def citations():
    return [
        {
            "chunk_id": str(CHUNK),
            "citation_index": 0,
            "quoted_text": FACT,
            "citation_label": "AAPL 10-K 2025",
            "location_label": "page 42, Results",
        }
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["instant", "direct", "quick_rag", "deep_rag"])
async def test_full_wire_contract_for_every_route(harness, route):
    h = harness
    h.route = "direct" if route == "instant" else route
    question = (
        "hi"
        if route == "instant"
        else ("How do I use this workspace?" if route == "direct" else QUESTION)
    )
    response = await h.post(question)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    text = {"instant": "Hi! How can I help with your filing research?", "direct": DIRECT}.get(
        route, FACT
    )
    expected_citations = citations() if route.endswith("rag") else []
    expected = [
        {"type": "start", "messageId": "assistant-message"},
        status("routing"),
        {"type": "data-route", "data": {"route": route}},
    ]
    if route.endswith("rag"):
        expected += [status(s) for s in ("searching", "analyzing", "validating")]
    content = answer_events(text, expected_citations)
    expected += content + [status("saving")] if route == "instant" else [status("saving")] + content
    expected.append({"type": "finish", "finishReason": "stop"})
    assert parse_sse(response.text) == expected
    assert [row["role"] for row in h.store.rows] == ["user", "assistant"]
    assert h.store.rows[0]["content"] == question
    saved = h.store.rows[-1]
    assert saved["content"] == text
    assert saved["message_data"]["route"] == route
    assert saved["message_data"]["citations"] == expected_citations
    assert all(
        isinstance(saved["message_data"][key], int) for key in ("routing_ms", "execution_ms")
    )
    assert {key: len(value) for key, value in h.model_calls.items()} == {
        "route": int(route != "instant"),
        "quick": int(route == "quick_rag"),
        "research": int(route == "deep_rag"),
    }
    if route == "instant":
        assert h.retriever.calls == []
    if route.endswith("rag"):
        assert len(h.retriever.calls) == 1  # Deep tools consume the actual prefetched evidence.


@pytest.mark.anyio
@pytest.mark.parametrize("error,code", [(ForbiddenThreadError, 403), (ThreadNotFoundError, 404)])
async def test_ownership_fails_as_http_before_models_or_persistence(harness, error, code):
    harness.store.access_error = error()
    response = await harness.post()
    assert response.status_code == code
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Thread access denied" if code == 403 else "Thread not found"
    }
    assert harness.store.rows == []
    assert harness.retriever.calls == []
    assert not any(harness.model_calls.values())


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "Old answer"}]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": "user", "content": " \n\t "}]},
        {"messages": [{"role": "user", "parts": [{"type": "text", "text": "\t "}]}]},
        {"messages": [{"role": "user", "parts": [{"type": "image"}]}]},
        {"threadId": "invalid-uuid"},
        {"messages": None},
    ],
)
async def test_invalid_user_input_is_422_without_persistence(harness, body):
    response = await harness.post(**body)
    assert response.status_code == 422
    assert harness.store.rows == []
    assert harness.store.save_attempts == 0
    assert harness.retriever.calls == []
    assert not any(harness.model_calls.values())


@pytest.mark.anyio
async def test_missing_messages_field_is_422(harness):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat/stream", json={"threadId": str(THREAD)})
    assert response.status_code == 422
    assert harness.store.rows == []
    assert not any(harness.model_calls.values())


@pytest.mark.anyio
@pytest.mark.parametrize("latest", [{"role": "user", "content": " \t "}, {"role": "user"}])
async def test_blank_latest_user_message_does_not_replay_history(harness, latest):
    response = await harness.post(
        messages=[
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "Old answer"},
            latest,
        ]
    )
    assert response.status_code == 422
    assert harness.store.rows == []
    assert not any(harness.model_calls.values())


@pytest.mark.anyio
async def test_text_parts_choose_latest_user_and_ignore_client_assistant_history(harness):
    response = await harness.post(
        messages=[
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Untrusted answer"},
            {
                "role": "user",
                "parts": [{"type": "text", "text": "h"}, {"type": "text", "text": "i"}],
            },
        ]
    )
    assert response.status_code == 200
    assert harness.store.rows[0]["content"] == "hi"
    assert not any(harness.model_calls.values())
    assert {"type": "data-route", "data": {"route": "instant"}} in parse_sse(response.text)


def explode(_messages):
    raise RuntimeError(SECRET)


def assert_failure(response, h, code, *, early_text=False):
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0] == {"type": "start", "messageId": "assistant-message"}
    assert events[-1] == {"type": "finish", "finishReason": "error"}
    assert events[-2]["type"] == "data-error"
    assert sum(event["type"] == "data-error" for event in events) == 1
    assert sum(event["type"] == "finish" for event in events) == 1
    assert any(event["type"] == "text-delta" for event in events) == early_text
    assert SECRET not in response.text
    assert "RuntimeError" not in response.text
    assert "ResearchLimitExceeded" not in response.text
    assert "GroundingError" not in response.text
    assert [row["role"] for row in h.store.rows] == ["user", "assistant"]
    failure = h.store.rows[-1]
    assert failure["content"] == ""
    assert failure["message_data"]["error_code"] == events[-2]["data"]["code"]
    assert set(failure["message_data"]) == {"phase", "route", "error_code"}
    assert events[-2] == {"type": "data-error", "data": {"code": code}}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure,route,code",
    [
        ("retrieval", "quick_rag", "retrieval_failed"),
        ("retrieval", "deep_rag", "retrieval_failed"),
        ("grounding", "quick_rag", "grounding_failed"),
        ("grounding", "deep_rag", "grounding_failed"),
        ("request_limit", "deep_rag", "processing_failed"),
        ("tool_limit", "deep_rag", "processing_failed"),
        ("bare_reply", "deep_rag", "processing_failed"),
        ("write", "quick_rag", "processing_failed"),
        ("write", "instant", "processing_failed"),
        ("unexpected_route", "quick_rag", "processing_failed"),
        ("unexpected", "quick_rag", "processing_failed"),
        ("unexpected", "deep_rag", "processing_failed"),
    ],
)
async def test_stream_failures_are_typed_sanitized_and_persisted(harness, failure, route, code):
    h = harness
    h.route = "direct" if route == "instant" else route
    if failure == "retrieval":
        h.retriever.fail = "search"
        h.research_reply = lambda _: tool("search_filings", {"query": QUESTION})
    elif failure == "grounding":
        h.quick_reply = lambda _: AgentAnswer(answer=SECRET)
        h.research_reply = lambda _: tool("submit_answer", {"answer": SECRET})
    elif failure == "request_limit":
        h.research_reply = lambda _: tool("search_filings", {"query": QUESTION})
    elif failure == "tool_limit":
        h.research_reply = lambda _: AIMessage(
            content="",
            tool_calls=[
                {"name": "read_chunk", "args": {"chunk_id": str(CHUNK)}, "id": f"read-{i}"}
                for i in range(9)
            ],
        )
    elif failure == "bare_reply":
        h.research_reply = lambda _: AIMessage(content=SECRET)
    elif failure == "write":
        h.store.write_error = True
    elif failure == "unexpected_route":

        async def fail_route(_messages):
            raise RuntimeError(SECRET)

        h.route_reply = fail_route
    else:
        h.quick_reply = h.research_reply = explode
    response = await h.post("hi" if route == "instant" else QUESTION)
    assert_failure(response, h, code, early_text=route == "instant")
    assert h.store.rows[-1]["message_data"]["route"] == (
        None if failure == "unexpected_route" else route
    )
    if failure == "grounding" and route == "deep_rag":
        assert len(h.model_calls["research"]) == 2  # Real grounding retry, then failure.
    if failure == "request_limit":
        assert len(h.model_calls["research"]) == 6
    if failure == "write":
        assert h.store.save_attempts == 1


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["read_chunk", "read_surrounding_chunks"])
async def test_evidence_read_failures_have_retrieval_error_code(harness, name):
    harness.route = "deep_rag"
    harness.retriever.fail = name
    harness.research_reply = lambda _: tool(name, {"chunk_id": str(CHUNK)})
    response = await harness.post()
    assert_failure(response, harness, "retrieval_failed")


@pytest.mark.anyio
async def test_no_evidence_quick_rag_streams_explicit_answer_without_answer_model(harness):
    harness.retriever.empty = True
    response = await harness.post()
    events = parse_sse(response.text)
    assert events[-1] == {"type": "finish", "finishReason": "stop"}
    assert {"type": "data-citations", "data": []} in events
    assert [event["data"]["stage"] for event in events if event["type"] == "data-status"] == [
        "routing",
        "searching",
        "saving",
    ]
    assert harness.model_calls["quick"] == []
    assert harness.store.rows[-1]["content"] == (
        "The corpus does not contain enough evidence to answer that question."
    )


@pytest.mark.anyio
async def test_unused_prefetch_failure_cannot_break_direct_reply(harness):
    harness.route = "direct"
    harness.retriever.fail = "search"
    response = await harness.post("How do I use this workspace?")
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1] == {"type": "finish", "finishReason": "stop"}
    assert not any(event["type"] == "data-error" for event in events)
    assert harness.store.rows[-1]["content"] == DIRECT
    assert harness.store.rows[-1]["message_data"]["route"] == "direct"


def asgi_exchange(body, on_frame):
    """Unbuffered ASGI 2.3 exchange so disconnects use Starlette's receive listener."""
    disconnected = asyncio.Event()
    request_sent = False
    wire = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        if message["type"] == "http.response.body" and message.get("body"):
            chunk = message["body"].decode()
            wire.append(chunk)
            for event in parse_sse(chunk):
                on_frame(event)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/stream",
        "raw_path": b"/chat/stream",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "root_path": "",
    }
    return asyncio.create_task(app(scope, receive, send)), disconnected, wire


async def finish_exchange(task):
    await asyncio.wait_for(task, 2)
    if _running_turns:
        await asyncio.wait_for(asyncio.gather(*tuple(_running_turns)), 2)
    await asyncio.sleep(0)  # Let the drain task's done callback release its strong reference.


@pytest.mark.anyio
async def test_disconnect_mid_stream_still_finishes_and_saves_real_graph_answer(harness):
    route_entered, release_route, routed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def paused_route(_messages):
        route_entered.set()
        await release_route.wait()
        return ModelRouteDecision(route="quick_rag")

    harness.route_reply = paused_route
    task, disconnected, wire = asgi_exchange(
        payload(), lambda event: routed.set() if event == status("routing") else None
    )
    try:
        await asyncio.wait_for(route_entered.wait(), 2)
        await asyncio.wait_for(routed.wait(), 2)
        disconnected.set()
        await asyncio.wait_for(asyncio.shield(task), 2)
        assert len(harness.store.rows) == 1
        assert not harness.store.saved.is_set()
        release_route.set()
        await asyncio.wait_for(harness.store.saved.wait(), 2)
    finally:
        release_route.set()
        disconnected.set()
        await finish_exchange(task)
    assert harness.store.rows[-1]["content"] == FACT
    assert harness.store.rows[-1]["message_data"]["citations"] == citations()
    assert not any(event["type"] == "finish" for event in parse_sse("".join(wire)))
    assert not _running_turns


@pytest.mark.anyio
async def test_instant_text_is_on_wire_while_answer_save_is_blocked(harness):
    text_seen = asyncio.Event()
    harness.store.save_gate = asyncio.Event()
    task, _, wire = asgi_exchange(
        payload("hi"),
        lambda event: text_seen.set() if event["type"] == "text-delta" else None,
    )
    try:
        await asyncio.wait_for(text_seen.wait(), 2)
        assert not harness.store.saved.is_set()
        assert [row["role"] for row in harness.store.rows] == ["user"]
        assert not any(harness.model_calls.values())
    finally:
        harness.store.save_gate.set()
        await finish_exchange(task)
    events = parse_sse("".join(wire))
    first_delta = next(i for i, event in enumerate(events) if event["type"] == "text-delta")
    assert first_delta < events.index(status("saving"))
    assert events[-1] == {"type": "finish", "finishReason": "stop"}
    assert harness.store.saved.is_set()
