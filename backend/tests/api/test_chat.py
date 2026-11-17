from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api.chat import (
    get_chat_store,
    get_document_retriever,
    get_resources,
)
from app.assistant.models import ChatModels
from app.auth.dependencies import AuthenticatedUser, get_current_user
from app.chat.routing import ModelRouteDecision
from app.database.chats import ForbiddenThreadError, ThreadNotFoundError
from app.main import app


@pytest.fixture
def user() -> AuthenticatedUser:
    return AuthenticatedUser(id=uuid4(), email="analyst@equity-research-assistant.example")


class MemoryChatStore:
    def __init__(self):
        self.thread_id = uuid4()
        self.owner_id: UUID | None = None
        self.messages: list[dict[str, object]] = []
        self.ensured_users: list[tuple[UUID, str | None]] = []

    async def ensure_user(self, user_id: UUID, email: str | None):
        self.ensured_users.append((user_id, email))

    async def list_threads(self, user_id: UUID):
        if self.owner_id != user_id:
            return []
        now = datetime.now(UTC).isoformat()
        return [
            {
                "id": str(self.thread_id),
                "title": None,
                "created_at": now,
                "updated_at": now,
            }
        ]

    async def create_thread(self, user_id: UUID, title: str | None):
        self.owner_id = user_id
        now = datetime.now(UTC).isoformat()
        return {
            "id": str(self.thread_id),
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    async def get_thread(self, user_id: UUID, thread_id: UUID):
        if thread_id != self.thread_id:
            raise ThreadNotFoundError()
        if self.owner_id != user_id:
            raise ForbiddenThreadError()
        return {"id": str(thread_id), "user_id": str(user_id)}

    async def list_messages(self, user_id: UUID, thread_id: UUID):
        await self.get_thread(user_id, thread_id)
        return self.messages

    async def append_message(self, thread_id: UUID, role: str, content: str, message_data):
        message = {
            "id": str(uuid4()),
            "thread_id": str(thread_id),
            "position": len(self.messages),
            "role": role,
            "content": content,
            "message_data": message_data,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.messages.append(message)
        return message

    async def append_grounded_answer(self, thread_id, content, message_data, citations):
        return await self.append_message(thread_id, "assistant", content, message_data)

    async def delete_thread(self, user_id: UUID, thread_id: UUID):
        await self.get_thread(user_id, thread_id)
        self.owner_id = None
        self.messages.clear()


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    async def retrieve(self, query: str, **kwargs):
        self.calls += 1
        return []


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return self.replies.pop(0)


def runaway_research() -> Scripted:
    search = AIMessage(
        content="",
        tool_calls=[{"name": "search_filings", "args": {"query": "x"}, "id": "x"}],
    )
    return Scripted(*([search] * 20))


def use_models(route: ModelRouteDecision | None, *, quick=(), research=None) -> ChatModels:
    models = ChatModels(
        route=Scripted(*([route] if route else [])),
        quick_answer=Scripted(*quick),
        research=research or Scripted(),
    )
    resources = SimpleNamespace(models=models, callbacks=[])
    app.dependency_overrides[get_resources] = lambda: resources
    return models


@pytest.fixture
def store() -> MemoryChatStore:
    return MemoryChatStore()


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_threads_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/chat/threads")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.anyio
async def test_create_thread_returns_thread(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/chat/threads", json={"title": "Apple revenue mix"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == "Apple revenue mix"
    assert store.ensured_users == [(user.id, user.email)]


@pytest.mark.anyio
async def test_delete_thread_removes_it_from_the_thread_list(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    await store.create_thread(user.id, "Apple revenue mix")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/chat/threads/{store.thread_id}")
        remaining = await client.get("/chat/threads")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert remaining.json() == []


@pytest.mark.anyio
async def test_delete_thread_refuses_another_users_thread(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    await store.create_thread(uuid4(), "Someone else's chat")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/chat/threads/{store.thread_id}")

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert store.owner_id is not None


@pytest.mark.anyio
async def test_delete_missing_thread_returns_not_found(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/chat/threads/{uuid4()}")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.anyio
async def test_stream_persists_messages_and_streams_the_answer(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    app.dependency_overrides[get_document_retriever] = lambda: FakeRetriever()
    use_models(ModelRouteDecision(route="direct", answer="I explain how to use this workspace."))
    await store.create_thread(user.id, "Apple revenue mix")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            json={
                "threadId": str(store.thread_id),
                "messages": [{"role": "user", "content": "What changed?"}],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    assert '"delta": "I explain how to use this workspace."' in response.text
    assert '"data": {"route": "direct"}' in response.text
    assert '"finishReason": "stop"' in response.text
    assert [message["role"] for message in store.messages] == ["user", "assistant"]
    assert store.messages[1]["message_data"]["route"] == "direct"


@pytest.mark.anyio
async def test_stream_can_select_quick_rag_without_running_deep_agent(user, store):
    async def override_user():
        return user

    retriever = FakeRetriever()
    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    app.dependency_overrides[get_document_retriever] = lambda: retriever
    models = use_models(ModelRouteDecision(route="quick_rag"), research=runaway_research())
    await store.create_thread(user.id, "Apple revenue mix")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            json={
                "threadId": str(store.thread_id),
                "messages": [{"role": "user", "content": "What changed?"}],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    assert retriever.calls == 1
    assert models.research.calls == 0
    assert '"stage": "routing"' in response.text
    assert '"stage": "searching"' in response.text
    # No passages matched, so the answer is the explicit no-evidence reply.
    assert "does not contain enough evidence" in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "Previous answer"}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "parts": [{"type": "text", "text": "\n\t"}]}],
    ],
)
async def test_stream_rejects_missing_or_blank_user_text_before_orchestration(
    user,
    store,
    messages,
):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    app.dependency_overrides[get_document_retriever] = lambda: FakeRetriever()
    models = use_models(None)
    await store.create_thread(user.id, "Apple revenue mix")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            json={"threadId": str(store.thread_id), "messages": messages},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json() == {"detail": "A non-blank user message is required"}
    assert store.messages == []
    assert models.route.calls == 0


@pytest.mark.anyio
async def test_stream_returns_controlled_error_when_agent_exhausts_budget(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    app.dependency_overrides[get_document_retriever] = lambda: FakeRetriever()
    use_models(ModelRouteDecision(route="deep_rag"), research=runaway_research())
    await store.create_thread(user.id, "Apple revenue mix")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            json={
                "threadId": str(store.thread_id),
                "messages": [{"role": "user", "content": "What changed?"}],
            },
        )

    # The stream opens before orchestration, so a post-stream failure arrives as a
    # typed data-error part rather than an HTTP status.
    assert response.status_code == status.HTTP_200_OK
    assert '"type": "data-error"' in response.text
    assert '"code": "processing_failed"' in response.text
    assert '"type": "text-delta"' not in response.text
    assert "ResearchLimitExceeded" not in response.text
    assert store.messages[-1]["message_data"]["error_code"] == "processing_failed"


@pytest.mark.anyio
async def test_messages_returns_forbidden_for_another_users_thread(user, store):
    other_user = AuthenticatedUser(id=uuid4(), email="other@equity-research-assistant.example")
    await store.create_thread(other_user.id, "Private")

    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/chat/threads/{store.thread_id}/messages")

    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.anyio
async def test_history_survives_a_new_http_client_session(user, store):
    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: store
    await store.create_thread(user.id, "Persistent")
    await store.append_message(store.thread_id, "user", "What changed?", {})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as first:
        initial = await first.get(f"/chat/threads/{store.thread_id}/messages")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as resumed:
        reloaded = await resumed.get(f"/chat/threads/{store.thread_id}/messages")

    assert initial.status_code == status.HTTP_200_OK
    assert reloaded.json() == initial.json()


@pytest.mark.anyio
async def test_citation_source_normalizes_neighbor_chunk_ids(user):
    message_id = uuid4()
    chunk_id = uuid4()
    neighbor_id = uuid4()

    class CitationStore:
        async def get_citation_source(self, user_id, requested_message_id, citation_index):
            assert user_id == user.id
            assert requested_message_id == message_id
            assert citation_index == 0
            return {
                "chunk_id": str(chunk_id),
                "citation_index": 0,
                "quoted_text": "Services revenue increased.",
                "document_chunks": {
                    "id": str(chunk_id),
                    "chunk_index": 2,
                    "text": "Services revenue increased.",
                    "page_number": 42,
                    "section": "Results",
                    "kind": "table",
                    "row_start": 4,
                    "row_end": 4,
                    "source_locator": {"html_id": "sales"},
                    "document_tables": {"title": "Net Sales", "units": "USD millions"},
                    "source_documents": {
                        "ticker": "AAPL",
                        "company_name": "Apple Inc.",
                        "filing_type": "10-K",
                        "filing_date": "2025-10-31",
                        "source_url": "https://www.sec.gov/example",
                    },
                },
                "previous_chunks": [
                    {
                        "id": str(neighbor_id),
                        "chunk_index": 1,
                        "text": "Prior context.",
                        "page_number": 41,
                        "section": "Results",
                    }
                ],
                "next_chunks": [],
            }

    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_chat_store] = lambda: CitationStore()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/chat/messages/{message_id}/citations/0/source")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["previous_chunks"][0]["chunk_id"] == str(neighbor_id)
    assert response.json()["kind"] == "table"
    assert response.json()["table_title"] == "Net Sales"
    assert response.json()["table_units"] == "USD millions"
