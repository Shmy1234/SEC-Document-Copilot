import asyncio
import hashlib
from time import monotonic
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field


class AuthenticatedUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID
    email: str | None = None
    user_metadata: dict[str, object] = Field(default_factory=dict)


class InvalidTokenError(Exception):
    """The bearer token was rejected by Supabase Auth."""


class SupabaseTokenVerifier:
    """Verifies Supabase access tokens against Supabase Auth.

    Verified tokens are remembered briefly so a page that fires several API calls
    pays for one Auth round trip instead of one per call. The window is short so
    a revoked session stops working within seconds; rejections are never cached.
    """

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        supabase_url: str,
        anon_key: str,
        cache_seconds: float = 30.0,
        max_entries: int = 1024,
        clock=monotonic,
    ) -> None:
        self._http = http
        self._user_url = f"{supabase_url}/auth/v1/user"
        self._anon_key = anon_key
        self._cache_seconds = cache_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._cache: dict[str, tuple[float, AuthenticatedUser]] = {}
        self._in_flight: dict[str, asyncio.Future[AuthenticatedUser]] = {}

    async def verify(self, token: str) -> AuthenticatedUser:
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None and cached[0] > self._clock():
            return cached[1]

        # A page fires several calls with the same token at once; they share one
        # upstream check instead of each paying for it.
        pending = self._in_flight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        pending = asyncio.get_running_loop().create_future()
        self._in_flight[key] = pending
        try:
            user = await self._fetch(key, token)
        except BaseException as error:
            pending.set_exception(error)
            # Waiters re-raise it; mark it retrieved so a lone caller does not warn.
            pending.exception()
            raise
        else:
            pending.set_result(user)
            return user
        finally:
            del self._in_flight[key]

    async def _fetch(self, key: str, token: str) -> AuthenticatedUser:
        try:
            response = await self._http.get(
                self._user_url,
                headers={"apikey": self._anon_key, "Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError:
            # Fail closed: an unreachable Auth service must not authenticate anyone,
            # and its error text must not reach the client.
            raise InvalidTokenError() from None
        if response.status_code != httpx.codes.OK:
            self._cache.pop(key, None)
            raise InvalidTokenError()
        try:
            user = AuthenticatedUser.model_validate(response.json())
        except (TypeError, ValueError):
            raise InvalidTokenError() from None

        now = self._clock()
        if len(self._cache) >= self._max_entries:
            self._evict(now)
        self._cache[key] = (now + self._cache_seconds, user)
        return user

    def _evict(self, now: float) -> None:
        for key in [key for key, (expires, _) in self._cache.items() if expires <= now]:
            del self._cache[key]
        while len(self._cache) >= self._max_entries:
            del self._cache[next(iter(self._cache))]
