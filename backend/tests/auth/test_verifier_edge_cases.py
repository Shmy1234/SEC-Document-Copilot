"""Auth HTTP failures, bounded cache behavior and process-lifetime resource wiring."""

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import State

from app import main
from app.auth.verifier import AuthenticatedUser, InvalidTokenError, SupabaseTokenVerifier

USER = {
    "id": "d6f0a9c4-2a6e-4ed0-96ba-8c2603f8b9bb",
    "email": "analyst@example.com",
    "user_metadata": {"name": "Analyst"},
}
SECRET = "private upstream exception: password=do-not-stream"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_application_state(monkeypatch):
    monkeypatch.setattr(main.app, "state", State())


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
async def make_verifier():
    clients = []

    def make(handler=None, *, clock=None, cache_seconds=30, max_entries=2):
        calls = []

        async def record(request):
            calls.append(request)
            if handler is None:
                return httpx.Response(200, json=USER)
            result = handler(request)
            return await result if asyncio.iscoroutine(result) else result

        http = httpx.AsyncClient(transport=httpx.MockTransport(record))
        clients.append(http)
        verifier = SupabaseTokenVerifier(
            http=http,
            supabase_url="https://project.supabase.co",
            anon_key="test-anon-key",
            cache_seconds=cache_seconds,
            max_entries=max_entries,
            clock=clock or Clock(),
        )
        return verifier, calls

    yield make
    for client in clients:
        await client.aclose()


async def auth_request(monkeypatch, verifier, header="Bearer token"):
    monkeypatch.setattr(
        main.app.state, "resources", SimpleNamespace(token_verifier=verifier), raising=False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        return await client.get("/auth/me", headers={"Authorization": header})


@pytest.mark.anyio
@pytest.mark.parametrize("elapsed,round_trips", [(0, 1), (29.999, 1), (30, 2), (30.001, 2)])
async def test_cache_expiration_boundary_is_exclusive(make_verifier, elapsed, round_trips):
    clock = Clock()
    verifier, calls = make_verifier(clock=clock)
    initial = await verifier.verify("token")
    clock.now += elapsed
    result = await verifier.verify("token")
    assert result == initial == AuthenticatedUser.model_validate(USER)
    assert len(calls) == round_trips
    assert all(call.headers["authorization"] == "Bearer token" for call in calls)


@pytest.mark.anyio
async def test_revocation_takes_effect_at_exact_ttl_and_is_never_negative_cached(
    make_verifier, monkeypatch
):
    clock = Clock()
    revoked = False

    def handler(_request):
        return httpx.Response(401, text=SECRET) if revoked else httpx.Response(200, json=USER)

    verifier, calls = make_verifier(handler, clock=clock)
    assert (await auth_request(monkeypatch, verifier)).status_code == 200
    revoked = True
    clock.now += 29.999
    assert (await auth_request(monkeypatch, verifier)).status_code == 200
    clock.now = 1030
    for _ in range(2):
        response = await auth_request(monkeypatch, verifier)
        assert_unauthorized(response)
    assert len(calls) == 3
    assert verifier._cache == {}


@pytest.mark.anyio
async def test_capacity_evicts_oldest_entry_but_preserves_other_live_tokens(make_verifier):
    verifier, calls = make_verifier(max_entries=2)
    for token in ("a", "b", "a", "c", "b", "c"):
        await verifier.verify(token)
        assert len(verifier._cache) <= 2
    assert [request.headers["authorization"] for request in calls] == [
        "Bearer a",
        "Bearer b",
        "Bearer c",
    ]
    await verifier.verify("a")
    assert len(calls) == 4
    assert len(verifier._cache) == 2
    assert set(verifier._cache) == {
        hashlib.sha256(token.encode()).hexdigest() for token in ("c", "a")
    }


@pytest.mark.anyio
async def test_capacity_removes_expired_entries_before_live_entries(make_verifier):
    clock = Clock()
    verifier, calls = make_verifier(clock=clock, max_entries=2)
    await verifier.verify("old")
    clock.now += 10
    await verifier.verify("live")
    clock.now += 20
    await verifier.verify("new")
    await verifier.verify("live")
    assert len(calls) == 3
    assert len(verifier._cache) == 2
    assert hashlib.sha256(b"old").hexdigest() not in verifier._cache


@pytest.mark.anyio
async def test_capacity_one_and_disabled_cache_have_no_hidden_reuse(make_verifier):
    verifier, calls = make_verifier(max_entries=1, cache_seconds=0)
    for token in ("a", "a", "b", "b"):
        await verifier.verify(token)
        assert len(verifier._cache) <= 1
    assert len(calls) == 4


@pytest.mark.anyio
async def test_concurrent_same_token_verifications_share_one_auth_request(make_verifier):
    release = asyncio.Event()
    all_entered = asyncio.Event()
    entered = 0

    async def handler(_request):
        await release.wait()
        return httpx.Response(200, json=USER)

    verifier, calls = make_verifier(handler)

    async def verify():
        nonlocal entered
        entered += 1
        if entered == 5:
            all_entered.set()
        return await verifier.verify("same-token")

    tasks = [asyncio.create_task(verify()) for _ in range(5)]
    try:
        await asyncio.wait_for(all_entered.wait(), 2)
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert results == [AuthenticatedUser.model_validate(USER)] * 5
    await verifier.verify("same-token")
    assert len(verifier._cache) == 1
    assert len(calls) == 1


def assert_unauthorized(response):
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired authentication token"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert SECRET not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "upstream",
    [
        "non_json",
        "null",
        "array",
        "invalid_uuid",
        "missing_id",
        "500",
        "503",
        "401",
        "read_timeout",
        "connect_timeout",
        "connect_error",
    ],
)
async def test_upstream_failures_are_sanitized_401_and_not_cached(
    make_verifier, monkeypatch, upstream
):
    def handler(request):
        if upstream == "read_timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        if upstream == "connect_timeout":
            raise httpx.ConnectTimeout(SECRET, request=request)
        if upstream == "connect_error":
            raise httpx.ConnectError(SECRET, request=request)
        if upstream.isdigit():
            return httpx.Response(int(upstream), text=SECRET)
        if upstream == "non_json":
            return httpx.Response(200, text=f"<html>{SECRET}</html>")
        bodies = {
            "null": None,
            "array": [],
            "invalid_uuid": {"id": "no-uuid"},
            "missing_id": {"email": "x"},
        }
        return httpx.Response(200, json=bodies[upstream])

    verifier, calls = make_verifier(handler)
    responses = [await auth_request(monkeypatch, verifier) for _ in range(2)]
    assert len(calls) == 2
    assert verifier._cache == {}
    for response in responses:
        assert_unauthorized(response)


@pytest.mark.anyio
async def test_non_json_failure_can_recover_on_next_verification(make_verifier):
    responses = iter([httpx.Response(200, text="not-json"), httpx.Response(200, json=USER)])
    verifier, calls = make_verifier(lambda _: next(responses))
    with pytest.raises(InvalidTokenError):
        await verifier.verify("token")
    assert await verifier.verify("token") == AuthenticatedUser.model_validate(USER)
    await verifier.verify("token")
    assert len(calls) == 2


class UntouchableResources:
    def __init__(self):
        self.reads = 0

    @property
    def resources(self):
        self.reads += 1
        raise AssertionError("header rejection must happen before resource lookup")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "header",
    [None, "", "Basic token", "Bearer", "Bearer   ", " Bearer token", "Bearer\ttoken", "token"],
)
async def test_missing_or_malformed_header_never_touches_resources(monkeypatch, header):
    state = UntouchableResources()
    monkeypatch.setattr(main.app, "state", state)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/auth/me", headers={} if header is None else {"Authorization": header}
        )
    assert_unauthorized(response)
    assert state.reads == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "header", ["Bearer token", "bearer token", "BEARER token", "bEaReR    token  "]
)
async def test_bearer_scheme_case_and_surrounding_token_spaces(make_verifier, monkeypatch, header):
    verifier, calls = make_verifier()
    response = await auth_request(monkeypatch, verifier, header)
    assert response.status_code == 200
    assert response.json() == USER
    assert len(calls) == 1
    assert calls[0].url == "https://project.supabase.co/auth/v1/user"
    assert calls[0].headers["authorization"] == "Bearer token"
    assert calls[0].headers["apikey"] == "test-anon-key"


def test_lifespan_shares_resources_across_authenticated_requests_and_closes_on_shutdown(
    monkeypatch,
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=USER)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    verifier = SupabaseTokenVerifier(
        http=http, supabase_url="https://project.supabase.co", anon_key="key", clock=Clock()
    )

    class LifetimeResources:
        token_verifier = verifier
        closes = 0

        async def aclose(self):
            self.closes += 1
            await http.aclose()

    resources = LifetimeResources()
    build = Mock(return_value=resources)
    monkeypatch.setattr(main, "build_resources", build)
    # TestClient's context drives the real lifespan; ASGITransport alone does not.
    with TestClient(main.app) as client:
        for _ in range(12):
            response = client.get("/auth/me", headers={"Authorization": "Bearer same-token"})
            assert response.status_code == 200
            assert response.json() == USER
            assert main.app.state.resources is resources
        assert len(calls) == 1
        build.assert_called_once_with(main.settings)
        assert resources.closes == 0
        assert not http.is_closed
    assert resources.closes == 1
    assert http.is_closed


@pytest.mark.anyio
async def test_lifespan_closes_resources_if_application_context_raises(monkeypatch):
    closes = []

    async def close():
        closes.append(True)

    resources = SimpleNamespace(aclose=close)
    build = Mock(return_value=resources)
    monkeypatch.setattr(main, "build_resources", build)
    with pytest.raises(RuntimeError, match="context failure"):
        async with main.lifespan(main.app):
            assert main.app.state.resources is resources
            raise RuntimeError("context failure")
    build.assert_called_once_with(main.settings)
    assert closes == [True]
