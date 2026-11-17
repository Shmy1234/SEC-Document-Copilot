"""The chat turn as a LangGraph state graph: choose a route, then answer by it.

    choose_route ─┬─ instant / direct ─▶ reply_directly
                  ├─ quick_rag ────────▶ quick_rag   (prefetched passages, one model call)
                  └─ deep_rag ─────────▶ deep_rag    (tool-calling research loop)

Progress is published through the custom stream ({"stage": ...}, {"route": ...});
the answer lands in state under "answer", already held to the grounding contract.
Persistence is not part of the graph: see app/chat/turn.py.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, TypedDict
from uuid import UUID

import structlog
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.assistant.agent import ResearchLimits, research
from app.assistant.deps import DocumentAgentDeps
from app.assistant.models import ChatModels
from app.assistant.outputs import GroundedAnswer
from app.chat.quick_rag import answer_from_passages
from app.chat.routing import ChatRoute, RouteDecision, instant_response, route_question
from app.chat.stages import RetrievalError, Stage
from app.grounding.validator import GroundingValidator
from app.retrieval.schemas import ChunkReference, SourcePassage

logger = structlog.get_logger(__name__)

PREFETCH_TOP_K = 5
PREFETCH_CANDIDATE_K = 50
_RAG_ROUTES = frozenset({"quick_rag", "deep_rag"})


class Retriever(Protocol):
    async def retrieve(self, query: str, **kwargs) -> list[SourcePassage]: ...

    async def read_chunk(self, chunk_id: UUID) -> ChunkReference | None: ...

    async def read_surrounding_chunks(
        self, chunk_id: UUID, *, window: int = 1
    ) -> dict[str, list[ChunkReference]] | None: ...


@dataclass
class TurnContext:
    models: ChatModels
    retriever: Retriever
    grounding_validator: GroundingValidator = field(default_factory=GroundingValidator)
    research_limits: ResearchLimits = field(default_factory=ResearchLimits)


class TurnState(TypedDict, total=False):
    question: str
    route: ChatRoute
    reply: str | None
    prefetched: list[SourcePassage] | None
    prefetch_failed: bool
    answer: GroundedAnswer


def _publish_stage(stage: Stage) -> None:
    get_stream_writer()({"stage": stage})


async def choose_route(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    _publish_stage("routing")
    question = state["question"]
    context = runtime.context

    if instant_response(question) is not None:
        decision = await route_question(question, context.models.route)
        prefetched, failed = None, False
    else:
        # Routing is a model round trip; the first search does not depend on its
        # outcome, so it runs alongside and is discarded if no search is needed.
        search = asyncio.create_task(
            context.retriever.retrieve(
                question, top_k=PREFETCH_TOP_K, candidate_k=PREFETCH_CANDIDATE_K
            )
        )
        try:
            decision = await route_question(question, context.models.route)
        except BaseException:
            await _abandon(search)
            raise
        prefetched, failed = await _settle_prefetch(search, decision)

    get_stream_writer()({"route": decision.route})
    return {
        "route": decision.route,
        "reply": decision.answer,
        "prefetched": prefetched,
        "prefetch_failed": failed,
    }


async def _settle_prefetch(
    search: asyncio.Task[list[SourcePassage]],
    decision: RouteDecision,
) -> tuple[list[SourcePassage] | None, bool]:
    if decision.route not in _RAG_ROUTES:
        await _abandon(search)
        return None, False
    try:
        return await search, False
    except Exception:
        logger.warning("prefetch_search_failed", exc_info=True)
        return None, True


async def _abandon(search: asyncio.Task) -> None:
    # Wait for the cancellation to land so no search outlives its turn, and
    # swallow its outcome: an unused prefetch's failure is not the turn's.
    search.cancel()
    await asyncio.gather(search, return_exceptions=True)


def pick_route(state: TurnState) -> str:
    return {
        "instant": "reply_directly",
        "direct": "reply_directly",
        "quick_rag": "quick_rag",
        "deep_rag": "deep_rag",
    }[state["route"]]


async def reply_directly(state: TurnState) -> TurnState:
    return {"answer": GroundedAnswer(answer=state["reply"] or "")}


async def quick_rag(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    _publish_stage("searching")
    if state.get("prefetch_failed") or state.get("prefetched") is None:
        raise RetrievalError("filing search failed")
    answer = await answer_from_passages(
        state["question"],
        state["prefetched"],
        answer_model=runtime.context.models.quick_answer,
        grounding_validator=runtime.context.grounding_validator,
        report_stage=_stage_reporter(),
    )
    return {"answer": answer}


async def deep_rag(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    _publish_stage("searching")
    context = runtime.context
    deps = DocumentAgentDeps(
        retriever=context.retriever,
        grounding_validator=context.grounding_validator,
        on_stage=_stage_reporter(),
    )
    answer = await research(
        state["question"],
        model=context.models.research,
        deps=deps,
        prefetched=state.get("prefetched"),
        limits=context.research_limits,
    )
    return {"answer": answer}


def _stage_reporter():
    write = get_stream_writer()

    async def report(stage: Stage) -> None:
        write({"stage": stage})

    return report


def build_turn_graph():
    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("choose_route", choose_route)
    graph.add_node("reply_directly", reply_directly)
    graph.add_node("quick_rag", quick_rag)
    graph.add_node("deep_rag", deep_rag)
    graph.add_edge(START, "choose_route")
    graph.add_conditional_edges(
        "choose_route", pick_route, ["reply_directly", "quick_rag", "deep_rag"]
    )
    for node in ("reply_directly", "quick_rag", "deep_rag"):
        graph.add_edge(node, END)
    return graph.compile(name="chat_turn")


turn_graph = build_turn_graph()
