from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.auth.dependencies import AuthenticatedUser, get_current_user
from app.auth.verifier import InvalidTokenError, SupabaseTokenVerifier

USER_PAYLOAD = {
    "id": "d6f0a9c4-2a6e-4ed0-96ba-8c2603f8b9bb",
    "email": "analyst@example.com",
    "user_metadata": {"name": "Analyst"},
}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def verifier_with(handler, *, clock=None, max_entries=1024) -> tuple[SupabaseTokenVerifier, list]:
    calls: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    verifier = SupabaseTokenVerifier(
        http=http,
        supabase_url="https://project.supabase.co",
        anon_key="anon-key",
        clock=clock or Clock(),
        max_entries=max_entries,
    )
    return verifier, calls


def ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=USER_PAYLOAD)


def request_with(verifier: SupabaseTokenVerifier) -> SimpleNamespace:
    resources = SimpleNamespace(token_verifier=verifier)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(resources=resources)))


@pytest.mark.anyio
async def test_missing_authorization_header_returns_401_without_calling_auth() -> None:
    with pytest.raises(HTTPException) as error:
        await get_current_user(SimpleNamespace(), None)

    assert error.value.status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Bearer   ", "token"])
async def test_malformed_authorization_header_returns_401(header) -> None:
    with pytest.raises(HTTPException) as error:
        await get_current_user(SimpleNamespace(), header)

    assert error.value.status_code == 401


@pytest.mark.anyio
async def test_invalid_supabase_token_returns_401() -> None:
    verifier, calls = verifier_with(lambda _request: httpx.Response(401))

    with pytest.raises(HTTPException) as error:
        await get_current_user(request_with(verifier), "Bearer expired-token")

    assert error.value.status_code == 401
    assert len(calls) == 1


@pytest.mark.anyio
async def test_valid_supabase_token_returns_current_user() -> None:
    verifier, calls = verifier_with(ok)

    user = await get_current_user(request_with(verifier), "Bearer valid-token")

    assert user == AuthenticatedUser.model_validate(USER_PAYLOAD)
    assert calls[0].url == "https://project.supabase.co/auth/v1/user"
    assert calls[0].headers["apikey"] == "anon-key"
    assert calls[0].headers["authorization"] == "Bearer valid-token"


@pytest.mark.anyio
async def test_malformed_auth_response_is_rejected() -> None:
    verifier, _ = verifier_with(lambda _request: httpx.Response(200, json={"email": "x"}))

    with pytest.raises(InvalidTokenError):
        await verifier.verify("token")


@pytest.mark.anyio
async def test_verified_token_is_reused_within_cache_window() -> None:
    clock = Clock()
    verifier, calls = verifier_with(ok, clock=clock)

    await verifier.verify("token")
    clock.now += 29
    await verifier.verify("token")

    assert len(calls) == 1


@pytest.mark.anyio
async def test_verified_token_is_rechecked_after_cache_window() -> None:
    clock = Clock()
    verifier, calls = verifier_with(ok, clock=clock)

    await verifier.verify("token")
    clock.now += 31
    await verifier.verify("token")

    assert len(calls) == 2


@pytest.mark.anyio
async def test_rejected_token_is_never_cached() -> None:
    responses = iter([httpx.Response(401), httpx.Response(401)])
    verifier, calls = verifier_with(lambda _request: next(responses))

    for _ in range(2):
        with pytest.raises(InvalidTokenError):
            await verifier.verify("token")

    assert len(calls) == 2


@pytest.mark.anyio
async def test_revoked_token_is_dropped_from_cache_on_recheck() -> None:
    clock = Clock()
    responses = iter([httpx.Response(200, json=USER_PAYLOAD), httpx.Response(401)])
    verifier, _ = verifier_with(lambda _request: next(responses), clock=clock)

    await verifier.verify("token")
    clock.now += 31
    with pytest.raises(InvalidTokenError):
        await verifier.verify("token")


@pytest.mark.anyio
async def test_cache_is_bounded() -> None:
    verifier, calls = verifier_with(ok, max_entries=2)

    for token in ("a", "b", "c"):
        await verifier.verify(token)
    await verifier.verify("a")

    assert len(calls) == 4
    assert len(verifier._cache) <= 2
