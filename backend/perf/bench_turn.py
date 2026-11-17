"""Offline latency benchmark; run from backend with `python perf/bench_turn.py`.

Only the model, retriever, persistence and resource-construction seams are fake.
The ASGI application, lifespan, authentication, graphs, tools and grounding run
unchanged. Timings describe injected sleeps, not production service latency.
"""

import asyncio
import json
import os
import sys
from contextlib import ExitStack, asynccontextmanager, redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Standalone execution needs no credentials or .env configuration.
    os.environ.update(
        {
            "SUPABASE_URL": "https://offline.invalid",
            "SUPABASE_ANON_KEY": "offline",
            "SUPABASE_SERVICE_ROLE_KEY": "offline",
            "DATABASE_URL": "postgresql://offline:offline@offline.invalid/offline",
            "OPENAI_API_KEY": "offline",
            "OPENAI_CHAT_MODEL": "openai:offline",
            "OPENAI_FAST_MODEL": "openai:offline",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
        }
    )

import httpx
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.api import chat as chat_api
from app.assistant.agent import research
from app.assistant.deps import DocumentAgentDeps
from app.assistant.models import ChatModels
from app.assistant.outputs import AgentAnswer, CitationDraft
from app.auth.verifier import SupabaseTokenVerifier
from app.chat.graph import TurnContext
from app.chat.routing import ModelRouteDecision
from app.chat.turn import ChatTurn
from app.grounding.validator import GroundingValidator
from app.main import app
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage

THREAD_ID = UUID(int=1)
USER = {"id": str(UUID(int=2)), "email": "offline@example.com"}
QUESTION = "What happened to Services revenue?"
FACT = "Services revenue increased 14% in fiscal 2025."


def passages() -> list[SourcePassage]:
    chunk = ChunkReference(
        chunk_id=UUID(int=3),
        document_id=UUID(int=4),
        chunk_index=1,
        text=FACT,
        page_number=42,
        section="Results",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="offline",
        source_url="https://offline.invalid",
    )
    return [SourcePassage(center=FusedChunk(chunk=chunk, rrf_score=1.0))]


class Models:
    def __init__(self, *, route="quick_rag", route_delay=0.0, answer_delay=0.0) -> None:
        self.route = route
        self.route_delay = route_delay
        self.answer_delay = answer_delay
        self.calls = {"route": [], "quick": [], "research": []}
        self.spans = {"route": [], "quick": [], "research": []}
        self.models = ChatModels(
            route=RunnableLambda(self.choose),
            quick_answer=RunnableLambda(self.answer),
            research=RunnableLambda(self.research),
        )

    async def wait(self, role: str, messages: list[BaseMessage], delay: float) -> None:
        self.calls[role].append(messages)
        start = perf_counter()
        await asyncio.sleep(delay)
        self.spans[role].append((start, perf_counter()))

    async def choose(self, messages: list[BaseMessage]) -> ModelRouteDecision:
        await self.wait("route", messages, self.route_delay)
        return ModelRouteDecision(route=self.route)

    async def answer(self, messages: list[BaseMessage]) -> AgentAnswer:
        await self.wait("quick", messages, self.answer_delay)
        evidence = json.loads(messages[-1].content)["evidence"]
        return AgentAnswer(
            answer=FACT,
            citations=[CitationDraft(evidence_id=evidence[0]["evidence_id"])],
        )

    async def research(self, messages: list[BaseMessage]) -> AIMessage:
        await self.wait("research", messages, self.answer_delay)
        evidence = [m for m in messages if isinstance(m, ToolMessage)]
        if not evidence:
            call = {"name": "search_filings", "args": {"query": QUESTION}, "id": "search"}
        else:
            evidence_id = json.loads(evidence[-1].content)[0]["evidence_id"]
            call = {
                "name": "submit_answer",
                "id": "submit",
                "args": {"answer": FACT, "citations": [{"evidence_id": evidence_id}]},
            }
        return AIMessage(content="", tool_calls=[call])


class Retriever:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = []
        self.spans = []

    async def retrieve(self, query: str, **kwargs) -> list[SourcePassage]:
        self.calls.append((query, kwargs))
        start = perf_counter()
        await asyncio.sleep(self.delay)
        self.spans.append((start, perf_counter()))
        return passages()


class Store:
    def __init__(self, *, user_delay=0.0, save_delay=0.0) -> None:
        self.user_delay = user_delay
        self.save_delay = save_delay
        self.saved = []
        self.user_finished = None
        self.save_started = None
        self.save_finished = None

    async def get_thread(self, user_id: UUID, thread_id: UUID) -> dict:
        assert user_id == UUID(USER["id"]) and thread_id == THREAD_ID
        return {"id": str(thread_id)}

    async def append_message(self, thread_id, role, content, message_data) -> dict:
        await asyncio.sleep(self.user_delay)
        self.user_finished = perf_counter()
        return {}

    async def append_grounded_answer(self, thread_id, content, message_data, citations) -> dict:
        self.save_started = perf_counter()
        await asyncio.sleep(self.save_delay)
        self.saved.append((content, message_data, citations))
        self.save_finished = perf_counter()
        return {}


def turn(models: Models, retriever: Retriever, store: Store) -> ChatTurn:
    return ChatTurn(store=store, context=TurnContext(models=models.models, retriever=retriever))


async def quick_measurement() -> SimpleNamespace:
    models = Models(route_delay=0.2)
    retriever, store = Retriever(0.2), Store()
    started = perf_counter()
    events = [
        event
        async for event in turn(models, retriever, store).run(
            thread_id=THREAD_ID,
            question=QUESTION,
        )
    ]
    return SimpleNamespace(
        models=models,
        retriever=retriever,
        store=store,
        events=events,
        elapsed=perf_counter() - started,
        phase=max(models.spans["route"][0][1], retriever.spans[0][1]) - started,
    )


async def research_measurement(*, prefetched: bool) -> SimpleNamespace:
    models, retriever = Models(answer_delay=0.04), Retriever(0.04)
    deps = DocumentAgentDeps(retriever=retriever, grounding_validator=GroundingValidator())
    started = perf_counter()
    answer = await research(
        QUESTION,
        model=models.models.research,
        deps=deps,
        prefetched=passages() if prefetched else None,
    )
    return SimpleNamespace(
        elapsed=perf_counter() - started,
        models=models,
        retriever=retriever,
        answer=answer,
    )


@asynccontextmanager
async def offline_app(models: Models, retriever: Retriever, store: Store):
    """Count lifespan builders without replacing the lifespan or API dependencies."""

    async def authenticate(request):
        auth_calls.append(request)
        await asyncio.sleep(0.005)
        return httpx.Response(200, json=USER)

    auth_calls, adapters = [], []
    engine, factory, openai, supabase = Mock(), Mock(), Mock(close=AsyncMock()), Mock()
    factory.side_effect = AssertionError("the fake retriever must not open a DB session")

    def build_retriever(*, session_factory, embedder):
        adapters.append((session_factory, embedder._client))
        return retriever

    state = app.state._state.copy()
    overrides = app.dependency_overrides.copy()
    async with httpx.AsyncClient(transport=httpx.MockTransport(authenticate)) as auth_http:
        with ExitStack() as stack:
            builders = {}
            for name, result in {
                "create_database_engine": engine,
                "create_session_factory": factory,
                "AsyncOpenAI": openai,
                "create_service_role_client": supabase,
                "build_chat_models": models.models,
                "build_tracer": None,
            }.items():
                builders[name] = stack.enter_context(
                    patch(f"app.resources.{name}", return_value=result)
                )
            http_builder = Mock(return_value=auth_http)
            stack.enter_context(
                patch("app.resources.httpx", SimpleNamespace(AsyncClient=http_builder))
            )
            store_builder = stack.enter_context(
                patch.object(chat_api, "ChatStore", return_value=store)
            )
            stack.enter_context(
                patch.object(chat_api, "DocumentRetriever", side_effect=build_retriever)
            )
            app.dependency_overrides.clear()
            try:
                async with app.router.lifespan_context(app):
                    yield SimpleNamespace(
                        builders=builders,
                        http_builder=http_builder,
                        auth_calls=auth_calls,
                        adapters=adapters,
                        engine=engine,
                        factory=factory,
                        openai=openai,
                        supabase=supabase,
                        store_builder=store_builder,
                    )
            finally:
                app.dependency_overrides.clear()
                app.dependency_overrides.update(overrides)
                app.state._state.clear()
                app.state._state.update(state)


async def reuse_measurement(count: int = 20) -> SimpleNamespace:
    models, retriever, store = Models(), Retriever(), Store()
    async with offline_app(models, retriever, store) as rig:
        started = perf_counter()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            responses = [
                await client.post(
                    "/chat/stream",
                    headers={"Authorization": "Bearer cached"},
                    json={
                        "threadId": str(THREAD_ID),
                        "messages": [{"role": "user", "content": QUESTION}],
                    },
                )
                for _ in range(count)
            ]
        rig.elapsed = perf_counter() - started
        rig.responses, rig.store = responses, store
    return rig


async def cache_measurement(*, count: int = 12, concurrent: bool = False) -> SimpleNamespace:
    calls = []
    now = [1000.0]
    release = asyncio.Event()

    async def authenticate(request):
        calls.append(request)
        if concurrent:
            await release.wait()
        await asyncio.sleep(0.02)
        return httpx.Response(200, json=USER)

    async with httpx.AsyncClient(transport=httpx.MockTransport(authenticate)) as http:
        verifier = SupabaseTokenVerifier(
            http=http,
            supabase_url="https://offline.invalid",
            anon_key="offline",
            clock=lambda: now[0],
            cache_seconds=30,
        )
        started = perf_counter()
        if concurrent:
            tasks = [asyncio.create_task(verifier.verify("same-token")) for _ in range(count)]
            # Every task reaches its first suspension before transport is released.
            # This also completes if verification later gains a single-flight lock.
            await asyncio.sleep(0)
            release.set()
            users = await asyncio.gather(*tasks)
        else:
            users = []
            for index in range(count):
                now[0] = 1000.0 + 29 * index / max(1, count - 1)
                users.append(await verifier.verify("same-token"))
        elapsed = perf_counter() - started
        within_ttl = len(calls)
        if not concurrent:
            now[0] = 1030.0
            await verifier.verify("same-token")
        return SimpleNamespace(
            calls=within_ttl,
            after_expiry=len(calls),
            elapsed=elapsed,
            users=users,
        )


async def sse_measurement() -> SimpleNamespace:
    models = Models(route_delay=0.12, answer_delay=0.12)
    retriever, store = Retriever(0.12), Store(user_delay=0.12, save_delay=0.12)
    body = json.dumps(
        {
            "threadId": str(THREAD_ID),
            "messages": [{"role": "user", "content": QUESTION}],
        }
    ).encode()
    frames, statuses = [], []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Future()

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message)
        elif message["type"] == "http.response.body" and message.get("body"):
            frames.append((perf_counter(), message["body"]))

    # httpx ASGITransport buffers the body. Observe actual ASGI sends instead.
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/stream",
        "raw_path": b"/chat/stream",
        "query_string": b"",
        "root_path": "",
        "server": ("test", 80),
        "client": ("test", 123),
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer cached")],
    }
    async with offline_app(models, retriever, store):
        started = perf_counter()
        await asyncio.wait_for(app(scope, receive, send), timeout=3)
    events = [(at, json.loads(frame.removeprefix(b"data: ").strip())) for at, frame in frames]
    return SimpleNamespace(
        first_byte=frames[0][0] - started,
        elapsed=frames[-1][0] - started,
        events=events,
        statuses=statuses,
        models=models,
        retriever=retriever,
        store=store,
    )


async def benchmark() -> list[tuple[str, str, str]]:
    reuse = await reuse_measurement()
    quick = await quick_measurement()
    warm = await research_measurement(prefetched=True)
    cold = await research_measurement(prefetched=False)
    cache = await cache_measurement()
    burst = await cache_measurement(concurrent=True)
    sse = await sse_measurement()
    stages = [event["data"]["stage"] for _, event in sse.events if event["type"] == "data-status"]
    return [
        (
            "20 ASGI turns",
            f"{reuse.elapsed * 1000:.1f} ms",
            "builders: "
            + "/".join(
                str(reuse.builders[name].call_count)
                for name in ("create_database_engine", "create_session_factory", "AsyncOpenAI")
            ),
        ),
        ("route + search (200 + 200)", f"{quick.phase * 1000:.1f} ms", "overlapped"),
        (
            "research seeded",
            f"{warm.elapsed * 1000:.1f} ms",
            f"model/search {len(warm.models.calls['research'])}/{len(warm.retriever.calls)}",
        ),
        (
            "research cold",
            f"{cold.elapsed * 1000:.1f} ms",
            f"model/search {len(cold.models.calls['research'])}/{len(cold.retriever.calls)}",
        ),
        (
            "auth sequential x12",
            f"{cache.elapsed * 1000:.1f} ms",
            f"{cache.calls} upstream; {cache.after_expiry} at expiry",
        ),
        ("auth cold burst x12", f"{burst.elapsed * 1000:.1f} ms", f"{burst.calls} upstream"),
        (
            "SSE first / complete",
            f"{sse.first_byte * 1000:.1f} / {sse.elapsed * 1000:.1f} ms",
            f"{len(stages)} stage frames",
        ),
    ]


if __name__ == "__main__":
    # Keep normal structured turn logs out of the small result table.
    with redirect_stdout(StringIO()):
        rows = asyncio.run(benchmark())
    print("Offline sleeps only; research seeded excludes the routing prefetch cost.")
    print(f"{'Scenario':<29} {'Measured':<22} Calls / observation")
    for scenario, measured, observation in rows:
        print(f"{scenario:<29} {measured:<22} {observation}")
