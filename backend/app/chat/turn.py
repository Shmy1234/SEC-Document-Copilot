"""One chat turn end to end: persist the question, run the turn graph, persist the answer.

`ChatTurn.run` is the whole interface. It yields progress as typed events and
raises when the turn fails, after recording the failure against the thread.
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from app.assistant.outputs import GroundedAnswer
from app.chat.graph import TurnContext, turn_graph
from app.chat.routing import ChatRoute
from app.chat.stages import Stage, error_code_for

logger = structlog.get_logger(__name__)


class ChatStore(Protocol):
    async def append_message(
        self,
        thread_id: UUID,
        role: str,
        content: str,
        message_data: dict[str, object],
    ) -> dict[str, object]: ...

    async def append_grounded_answer(
        self,
        thread_id: UUID,
        content: str,
        message_data: dict[str, object],
        citations: list[dict[str, object]],
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class StageEvent:
    stage: Stage


@dataclass(frozen=True)
class RouteEvent:
    route: ChatRoute


@dataclass(frozen=True)
class AnswerEvent:
    answer: GroundedAnswer


TurnEvent = StageEvent | RouteEvent | AnswerEvent


class ChatTurn:
    def __init__(
        self,
        *,
        store: ChatStore,
        context: TurnContext,
        callbacks: list[BaseCallbackHandler] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._store = store
        self._context = context
        self._callbacks = callbacks or []
        self._clock = clock

    async def run(
        self,
        *,
        thread_id: UUID,
        question: str,
    ) -> AsyncIterator[TurnEvent]:
        await self._store.append_message(thread_id, "user", question, {"phase": 8})

        route: ChatRoute | None = None
        try:
            started = self._clock()
            routed_at = started
            answer: GroundedAnswer | None = None
            async for part in turn_graph.astream(
                {"question": question},
                context=self._context,
                config={
                    "callbacks": self._callbacks,
                    "run_name": "chat_turn",
                    "metadata": {"thread_id": str(thread_id)},
                },
                stream_mode=["custom", "updates"],
                version="v2",
            ):
                if part["type"] == "custom":
                    data = part["data"]
                    if "stage" in data:
                        yield StageEvent(data["stage"])
                    if "route" in data:
                        route = data["route"]
                        routed_at = self._clock()
                        yield RouteEvent(route)
                    continue
                for update in part["data"].values():
                    if update and "answer" in update:
                        answer = update["answer"]
            finished = self._clock()
            if answer is None or route is None:
                raise RuntimeError("turn graph finished without an answer")

            # Canned replies are final the moment they exist; show them before
            # the write. Grounded answers wait for the write so a reload agrees.
            if route == "instant":
                yield AnswerEvent(answer)

            routing_ms = _elapsed_ms(started, routed_at)
            execution_ms = _elapsed_ms(routed_at, finished)
            citations = [citation.model_dump(mode="json") for citation in answer.citations]
            yield StageEvent("saving")
            await self._store.append_grounded_answer(
                thread_id,
                answer.answer,
                {
                    "phase": 8,
                    "route": route,
                    "routing_ms": routing_ms,
                    "execution_ms": execution_ms,
                    "citations": citations,
                },
                citations,
            )
        except Exception as error:
            # The user message is already committed. Record the failure beside it
            # so a reloaded thread shows what happened and a retry does not read as
            # a duplicate question. Persistence must never mask the original failure.
            await self._persist_failure(thread_id, route, error)
            raise

        logger.info(
            "chat_turn_completed",
            route=route,
            routing_ms=routing_ms,
            execution_ms=execution_ms,
        )
        if route != "instant":
            yield AnswerEvent(answer)

    async def _persist_failure(
        self,
        thread_id: UUID,
        route: ChatRoute | None,
        error: BaseException,
    ) -> None:
        code = error_code_for(error)
        try:
            await self._store.append_message(
                thread_id,
                "assistant",
                "",
                {"phase": 8, "route": route, "error_code": code},
            )
        except Exception:
            logger.exception("chat_turn_failure_not_persisted", error_code=code, route=route)


def _elapsed_ms(started: float, finished: float) -> int:
    return max(0, round((finished - started) * 1000))
