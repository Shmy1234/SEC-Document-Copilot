from fastapi import Header, HTTPException, Request, status

from app.auth.verifier import AuthenticatedUser, InvalidTokenError, SupabaseTokenVerifier

__all__ = ["AuthenticatedUser", "get_current_user", "get_token_verifier"]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise _unauthorized()

    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise _unauthorized()
    return token.strip()


def get_token_verifier(request: Request) -> SupabaseTokenVerifier:
    return request.app.state.resources.token_verifier


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AuthenticatedUser:
    """Verify a Supabase JWT and return the authenticated user."""
    # Parse before touching app state so a missing header is rejected cheaply.
    token = _bearer_token(authorization)
    try:
        return await get_token_verifier(request).verify(token)
    except InvalidTokenError:
        raise _unauthorized() from None
