from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.resources import Resources, build_resources, build_tracer


def test_build_resources_shares_one_engine_and_http_client() -> None:
    with patch("app.resources.create_service_role_client", return_value=Mock()):
        resources = build_resources(settings)

    try:
        assert resources.session_factory.kw["bind"] is resources.engine
        assert resources.token_verifier._http is resources.http
        assert str(resources.engine.url).startswith("postgresql+psycopg://")
    finally:
        resources.engine.dispose()


@pytest.mark.anyio
async def test_aclose_releases_every_pool() -> None:
    openai = AsyncMock()
    http = AsyncMock()
    engine = Mock()
    tracer = Mock()
    resources = Resources(
        session_factory=Mock(),
        openai=openai,
        supabase=Mock(),
        token_verifier=Mock(),
        models=Mock(),
        tracer=tracer,
        engine=engine,
        http=http,
    )

    await resources.aclose()

    tracer.wait_for_futures.assert_called_once()
    openai.close.assert_awaited_once()
    http.aclose.assert_awaited_once()
    engine.dispose.assert_called_once()


def test_lifespan_builds_resources_once_and_closes_them() -> None:
    resources = Mock(aclose=AsyncMock())
    with (
        patch("app.main.build_resources", return_value=resources) as build,
        TestClient(app) as client,
    ):
        client.get("/health")
        client.get("/health")
        assert app.state.resources is resources

    build.assert_called_once()
    resources.aclose.assert_awaited_once()


def test_tracing_is_off_by_default():
    assert build_tracer(settings) is None
    resources = Resources(
        session_factory=Mock(), openai=Mock(), supabase=Mock(), token_verifier=Mock(), models=Mock()
    )
    assert resources.callbacks == []


def test_tracer_targets_the_configured_langsmith_project():
    configured = settings.model_copy(
        update={
            "LANGSMITH_TRACING": True,
            "LANGSMITH_API_KEY": "lsv2-test",
            "LANGSMITH_PROJECT": "copilot-staging",
        }
    )

    tracer = build_tracer(configured)

    assert tracer is not None
    assert tracer.project_name == "copilot-staging"
    assert tracer.client.api_url == "https://api.smith.langchain.com"
