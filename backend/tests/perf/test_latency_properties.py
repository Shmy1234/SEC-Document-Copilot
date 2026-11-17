"""Measured latency properties, using real graphs/tools/grounding and offline seams."""

import asyncio

import pytest

from app.chat.turn import AnswerEvent, RouteEvent
from perf.bench_turn import (
    FACT,
    QUESTION,
    THREAD_ID,
    Models,
    Retriever,
    Store,
    cache_measurement,
    quick_measurement,
    research_measurement,
    reuse_measurement,
    sse_measurement,
    turn,
)


def test_twenty_asgi_stream_requests_share_lifespan_clients():
    result = asyncio.run(reuse_measurement())

    assert len(result.responses) == len(result.adapters) == len(result.store.saved) == 20
    for response in result.responses:
        assert response.status_code == 200
        assert '"finishReason": "stop"' in response.text
        assert '"type": "data-error"' not in response.text
    for name, builder in result.builders.items():
        assert builder.call_count == 1, name
    result.builders["create_session_factory"].assert_called_once_with(result.engine)
    result.http_builder.assert_called_once_with(timeout=5.0)
    assert all(
        factory is result.factory and client is result.openai for factory, client in result.adapters
    )
    assert all(call.args == (result.supabase,) for call in result.store_builder.call_args_list)
    assert len(result.auth_calls) == 1
    result.openai.close.assert_awaited_once()
    result.engine.dispose.assert_called_once()
    assert all(citations for _, _, citations in result.store.saved)


def test_two_200ms_operations_overlap_in_quick_rag():
    result = asyncio.run(quick_measurement())
    route_start, route_end = result.models.spans["route"][0]
    search_start, search_end = result.retriever.spans[0]

    # A sequential implementation takes at least 400ms. Allow 150ms scheduling
    # headroom, and independently require at least 150ms of observed overlap.
    assert 0.19 <= result.phase < 0.35, result.phase
    assert min(route_end, search_end) - max(route_start, search_start) >= 0.15
    assert len(result.models.calls["route"]) == len(result.models.calls["quick"]) == 1
    assert len(result.retriever.calls) == 1
    assert next(e.route for e in result.events if isinstance(e, RouteEvent)) == "quick_rag"
    answer = next(e.answer for e in result.events if isinstance(e, AnswerEvent))
    assert answer.answer == FACT and answer.citations[0].quoted_text == FACT
    assert 190 <= result.store.saved[0][1]["routing_ms"] < 350


def test_prefetch_saves_one_model_call_and_one_search_in_real_research_loop():
    async def measure():
        return (
            await research_measurement(prefetched=True),
            await research_measurement(prefetched=False),
        )

    warm, cold = asyncio.run(measure())
    assert len(cold.models.calls["research"]) == 2
    assert len(warm.models.calls["research"]) == 1
    assert len(cold.retriever.calls) == 1
    assert len(warm.retriever.calls) == 0
    assert warm.answer == cold.answer
    assert warm.answer.citations[0].quoted_text == FACT


def test_deep_rag_graph_spends_one_prefetch_and_one_research_call_total():
    async def run():
        models, retriever, store = Models(route="deep_rag"), Retriever(), Store()
        events = [
            event
            async for event in turn(models, retriever, store).run(
                thread_id=THREAD_ID,
                question=QUESTION,
            )
        ]
        return models, retriever, events

    models, retriever, events = asyncio.run(run())
    # The saved retrieval is inside research; end-to-end the prefetch still costs one.
    assert len(retriever.calls) == len(models.calls["research"]) == 1
    assert len(models.calls["route"]) == 1
    assert next(e.route for e in events if isinstance(e, RouteEvent)) == "deep_rag"
    assert next(e.answer for e in events if isinstance(e, AnswerEvent)).citations


def test_twelve_verifications_inside_ttl_pay_for_one_upstream_request():
    result = asyncio.run(cache_measurement())

    assert result.calls == 1
    assert result.after_expiry == 2
    assert len(result.users) == 12 and all(user == result.users[0] for user in result.users)


def test_concurrent_cold_verifications_share_one_upstream_call():
    result = asyncio.run(cache_measurement(concurrent=True))

    assert result.calls == 1, f"12 overlapping requests made {result.calls} upstream calls"


@pytest.fixture(scope="module")
def sse():
    return asyncio.run(sse_measurement())


def test_sse_first_body_byte_precedes_slow_turn_work(sse):
    assert sse.statuses[0]["status"] == 200
    headers = dict(sse.statuses[0]["headers"])
    assert headers[b"x-accel-buffering"] == b"no"
    assert sse.events[0][1]["type"] == "start"
    assert sse.first_byte < 0.10, sse.first_byte
    assert sse.elapsed >= 0.46
    assert sse.events[0][0] < sse.store.user_finished
    assert sse.events[-1][1] == {"type": "finish", "finishReason": "stop"}


def test_every_sse_stage_is_sent_before_the_work_it_announces_finishes(sse):
    statuses = [
        (at, event["data"]["stage"]) for at, event in sse.events if event["type"] == "data-status"
    ]
    assert [stage for _, stage in statuses] == [
        "routing",
        "searching",
        "analyzing",
        "validating",
        "saving",
    ]
    sent = {stage: at for at, stage in statuses}
    route_start, route_end = sse.models.spans["route"][0]
    answer_start, answer_end = sse.models.spans["quick"][0]
    search_end = sse.retriever.spans[0][1]
    assert route_start <= sent["routing"] < route_end
    assert max(route_end, search_end) <= sent["searching"] < answer_end
    assert answer_start <= sent["analyzing"] < answer_end
    assert answer_end <= sent["validating"] < sse.store.save_finished
    assert sse.store.save_started <= sent["saving"] < sse.store.save_finished
    # Awaiting the route/answer/store sleeps gives the sender a chance to flush.
    assert route_end - sent["routing"] >= 0.08
    assert answer_end - sent["analyzing"] >= 0.08
    assert sse.store.save_finished - sent["saving"] >= 0.08
    first_text = next(at for at, event in sse.events if event["type"] == "text-delta")
    assert first_text >= sse.store.save_finished
