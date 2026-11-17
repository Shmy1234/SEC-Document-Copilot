import asyncio
import json

from app.assistant.outputs import GroundedAnswer
from app.chat.stages import RetrievalError
from app.chat.streaming import stream_chat_turn
from app.chat.turn import AnswerEvent, RouteEvent, StageEvent
from app.grounding.validator import GroundingError

FINISH_STOP = 'data: {"type": "finish", "finishReason": "stop"}\n\n'
FINISH_ERROR = 'data: {"type": "finish", "finishReason": "error"}\n\n'


async def scripted(*events, error: Exception | None = None):
    for event in events:
        yield event
    if error is not None:
        raise error


def collect(events) -> list[str]:
    async def run():
        return [part async for part in stream_chat_turn(events, chunk_size=10)]

    return asyncio.run(run())


def payloads(parts: list[str]) -> list[dict]:
    return [json.loads(part.removeprefix("data: ").strip()) for part in parts]


def stages_in(parts: list[str]) -> list[str]:
    return [p["data"]["stage"] for p in payloads(parts) if p["type"] == "data-status"]


def test_answer_is_chunked_into_text_deltas_followed_by_citations_and_finish():
    answer = GroundedAnswer(answer="Services revenue increased.")

    parts = collect(scripted(AnswerEvent(answer)))
    kinds = [p["type"] for p in payloads(parts)]

    assert parts[0] == 'data: {"type": "start", "messageId": "assistant-message"}\n\n'
    assert kinds[1] == "text-start"
    deltas = [p["delta"] for p in payloads(parts) if p["type"] == "text-delta"]
    assert "".join(deltas) == answer.answer
    assert all(len(delta) <= 10 for delta in deltas)
    assert kinds[-3:] == ["text-end", "data-citations", "finish"]
    assert parts[-1] == FINISH_STOP


def test_stages_stream_in_order_before_the_answer():
    stages = ["routing", "searching", "analyzing", "validating", "saving"]
    parts = collect(
        scripted(*(StageEvent(s) for s in stages), AnswerEvent(GroundedAnswer(answer="Done.")))
    )

    assert stages_in(parts) == stages
    last_status = max(i for i, part in enumerate(parts) if '"data-status"' in part)
    first_delta = min(i for i, part in enumerate(parts) if '"text-delta"' in part)
    assert last_status < first_delta


def test_route_is_streamed_as_a_data_part():
    parts = collect(
        scripted(
            StageEvent("routing"),
            RouteEvent("direct"),
            AnswerEvent(GroundedAnswer(answer="Hello.")),
        )
    )

    routes = [p["data"]["route"] for p in payloads(parts) if p["type"] == "data-route"]
    assert routes == ["direct"]


def test_disconnected_turn_keeps_running_after_the_stream_closes():
    """A closed browser tab must not abandon a turn before it persists."""

    async def verify():
        finished = asyncio.Event()

        async def events():
            await asyncio.sleep(0)
            finished.set()
            yield AnswerEvent(GroundedAnswer(answer="Persisted anyway."))

        stream = stream_chat_turn(events())
        await anext(stream)
        await stream.aclose()
        await asyncio.wait_for(finished.wait(), timeout=0.5)

    asyncio.run(verify())


def test_retrieval_failure_is_typed_and_has_no_answer_text():
    parts = collect(
        scripted(
            StageEvent("searching"),
            error=RetrievalError("pgvector connection reset at 10.0.0.4:5432"),
        )
    )

    assert stages_in(parts) == ["searching"]
    assert '"code": "retrieval_failed"' in parts[-2]
    assert parts[-1] == FINISH_ERROR
    assert not any('"text-delta"' in part for part in parts)


def test_grounding_failure_is_typed():
    parts = collect(scripted(StageEvent("validating"), error=GroundingError("bad citation")))

    assert '"code": "grounding_failed"' in parts[-2]


def test_exception_text_never_reaches_the_browser():
    secret = "postgres://analyst:hunter2@db.internal:5432"

    body = "".join(collect(scripted(StageEvent("searching"), error=RuntimeError(secret))))

    assert '"code": "processing_failed"' in body
    assert secret not in body
    assert "RuntimeError" not in body


def test_unexpected_failure_is_logged_with_its_stage(monkeypatch):
    logged = []

    class FakeLogger:
        def exception(self, event, **context):
            logged.append((event, context))

    monkeypatch.setattr("app.chat.streaming.logger", FakeLogger())

    collect(scripted(StageEvent("searching"), error=RuntimeError("database unavailable")))

    assert logged == [("chat_turn_failed", {"error_code": "processing_failed", "stage": "searching"})]


def test_turn_that_ends_without_an_answer_fails_closed():
    parts = collect(scripted(StageEvent("routing")))

    assert '"code": "processing_failed"' in parts[-2]
    assert parts[-1] == FINISH_ERROR


def test_stream_opens_before_slow_turn_work_completes():
    async def verify():
        release = asyncio.Event()

        async def events():
            await release.wait()
            yield AnswerEvent(GroundedAnswer(answer="Finished."))

        stream = stream_chat_turn(events())
        first = await asyncio.wait_for(anext(stream), timeout=0.1)
        release.set()
        await stream.aclose()
        return first

    assert asyncio.run(verify()) == 'data: {"type": "start", "messageId": "assistant-message"}\n\n'


def test_early_answer_text_arrives_before_later_work_finishes():
    async def verify():
        release = asyncio.Event()

        async def events():
            yield StageEvent("routing")
            yield AnswerEvent(GroundedAnswer(answer="Hello."))
            yield StageEvent("saving")
            await release.wait()

        stream = stream_chat_turn(events())
        observed = []
        while not any('"text-delta"' in part for part in observed):
            observed.append(await asyncio.wait_for(anext(stream), timeout=0.1))
        assert not release.is_set()
        release.set()
        observed.extend([part async for part in stream])
        return observed

    parts = asyncio.run(verify())

    first_delta = next(i for i, part in enumerate(parts) if '"text-delta"' in part)
    saving = next(i for i, part in enumerate(parts) if '"stage": "saving"' in part)
    assert first_delta < saving
    assert parts[-1] == FINISH_STOP


def test_failure_after_early_text_finishes_with_typed_error():
    parts = collect(
        scripted(
            AnswerEvent(GroundedAnswer(answer="Hello.")),
            error=RuntimeError("database unavailable"),
        )
    )

    assert any('"text-delta"' in part for part in parts)
    assert '"code": "processing_failed"' in parts[-2]
    assert parts[-1] == FINISH_ERROR
