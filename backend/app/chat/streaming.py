"""Turn events → AI SDK UI message stream (server-sent events)."""

import asyncio
import json
from collections.abc import AsyncIterator

import structlog

from app.assistant.outputs import GroundedAnswer
from app.chat.stages import Stage, error_code_for
from app.chat.turn import AnswerEvent, RouteEvent, StageEvent, TurnEvent

logger = structlog.get_logger(__name__)

# A browser that navigates away or closes mid-turn makes Starlette cancel the
# response generator. The turn is drained by its own task, held here because
# asyncio keeps only weak references to tasks: turns finish and save whether or
# not anyone is listening.
_running_turns: set[asyncio.Task[None]] = set()

_DONE = object()


def _release_turn(task: asyncio.Task[None]) -> None:
    _running_turns.discard(task)
    # A turn whose reader disconnected is never awaited. Retrieve the outcome so
    # a failure does not resurface as an unretrieved task exception; every
    # failure is recorded against the thread by ChatTurn either way.
    if not task.cancelled():
        task.exception()


async def stream_chat_turn(
    events: AsyncIterator[TurnEvent],
    *,
    chunk_size: int = 80,
) -> AsyncIterator[str]:
    """Emit each event as soon as the turn produces it.

    The event iterator is drained by a detached task into a queue, so a status
    part is sent only once its work has started and the turn survives the
    client going away.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")

    queue: asyncio.Queue[TurnEvent | object] = asyncio.Queue()
    current_stage: Stage | None = None

    async def drain() -> None:
        try:
            async for event in events:
                await queue.put(event)
        finally:
            await queue.put(_DONE)

    task = asyncio.create_task(drain())
    _running_turns.add(task)
    task.add_done_callback(_release_turn)
    yield _event({"type": "start", "messageId": "assistant-message"})

    answered = False
    while (update := await queue.get()) is not _DONE:
        if isinstance(update, StageEvent):
            current_stage = update.stage
            yield _event({"type": "data-status", "data": {"stage": update.stage}})
        elif isinstance(update, RouteEvent):
            yield _event({"type": "data-route", "data": {"route": update.route}})
        elif isinstance(update, AnswerEvent):
            answered = True
            async for event in _answer_content_events(update.answer, chunk_size):
                yield event

    try:
        await task
    except Exception as error:
        # Fail closed: the stream is already open, so every failure must become a
        # typed code rather than a dropped connection. The traceback stays here.
        code = error_code_for(error)
        logger.exception("chat_turn_failed", error_code=code, stage=current_stage)
    else:
        if not answered:
            code = "processing_failed"
            logger.error("chat_turn_ended_without_answer", stage=current_stage)
        else:
            yield _event({"type": "finish", "finishReason": "stop"})
            return

    # Only the code crosses the wire — exception text never reaches the browser.
    yield _event({"type": "data-error", "data": {"code": code}})
    yield _event({"type": "finish", "finishReason": "error"})


async def _answer_content_events(answer: GroundedAnswer, chunk_size: int) -> AsyncIterator[str]:
    yield _event({"type": "text-start", "id": "text-1"})
    for start in range(0, len(answer.answer), chunk_size):
        chunk = answer.answer[start : start + chunk_size]
        yield _event({"type": "text-delta", "id": "text-1", "delta": chunk})

    citation_data = [citation.model_dump(mode="json") for citation in answer.citations]
    yield _event({"type": "text-end", "id": "text-1"})
    yield _event({"type": "data-citations", "data": citation_data})


def _event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
