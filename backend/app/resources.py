"""Process-lifetime adapters to external systems.

Everything here holds a connection pool or a TLS session, so it is built once in
the FastAPI lifespan and shared by every request. Building these per request
was the backend's largest source of latency and leaked database connections.
"""

from dataclasses import dataclass

import httpx
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client as LangSmithClient
from openai import AsyncOpenAI
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from supabase import Client

from app.assistant.models import ChatModels, build_chat_models
from app.auth.verifier import SupabaseTokenVerifier
from app.config import Settings
from app.database.engine import create_database_engine, create_session_factory
from app.database.supabase import create_service_role_client


@dataclass
class Resources:
    session_factory: sessionmaker[Session]
    openai: AsyncOpenAI
    supabase: Client
    token_verifier: SupabaseTokenVerifier
    models: ChatModels
    tracer: LangChainTracer | None = None
    engine: Engine | None = None
    http: httpx.AsyncClient | None = None

    @property
    def callbacks(self) -> list[LangChainTracer]:
        return [self.tracer] if self.tracer is not None else []

    async def aclose(self) -> None:
        if self.tracer is not None:
            # Traces are posted in the background; flush them before exit.
            self.tracer.wait_for_futures()
        await self.openai.close()
        if self.http is not None:
            await self.http.aclose()
        if self.engine is not None:
            self.engine.dispose()


def build_resources(settings: Settings) -> Resources:
    engine = create_database_engine(settings.sqlalchemy_database_url)
    http = httpx.AsyncClient(timeout=5.0)
    return Resources(
        engine=engine,
        session_factory=create_session_factory(engine),
        openai=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        supabase=create_service_role_client(),
        http=http,
        token_verifier=SupabaseTokenVerifier(
            http=http,
            supabase_url=settings.SUPABASE_URL,
            anon_key=settings.SUPABASE_ANON_KEY,
        ),
        models=build_chat_models(settings),
        tracer=build_tracer(settings),
    )


def build_tracer(settings: Settings) -> LangChainTracer | None:
    if not settings.LANGSMITH_TRACING:
        return None
    client = LangSmithClient(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT,
    )
    return LangChainTracer(project_name=settings.LANGSMITH_PROJECT, client=client)
