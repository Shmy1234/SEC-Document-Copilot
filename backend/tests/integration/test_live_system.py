import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.retrieval.embeddings import OpenAIQueryEmbedder
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.schemas import RetrievalFilters
from ingest.repository import create_sessionmaker

pytestmark = pytest.mark.integration

if os.environ.get("RUN_RETRIEVAL_INTEGRATION") != "1":
    pytest.skip("set RUN_RETRIEVAL_INTEGRATION=1 to run live checks", allow_module_level=True)


def test_live_schema_and_corpus_are_ready():
    session_factory = create_sessionmaker()
    with session_factory() as session:
        revision = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        documents = session.execute(text("SELECT COUNT(*) FROM source_documents")).scalar_one()
        chunks = session.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar_one()
        missing_embeddings = session.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL")
        ).scalar_one()
        dimensions = session.execute(
            text(
                "SELECT DISTINCT vector_dims(embedding) FROM document_chunks "
                "WHERE embedding IS NOT NULL"
            )
        ).scalars().all()

    assert revision == "0003_atomic_chat_persistence"
    assert documents == 25
    assert chunks > documents
    assert missing_embeddings == 0
    assert dimensions == [1536]


def test_atomic_chat_functions_serialize_positions_and_roll_back_bad_citations():
    session_factory = create_sessionmaker()
    user_id = uuid4()
    thread_id = uuid4()
    with session_factory.begin() as session:
        session.execute(
            text("INSERT INTO users (id, email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"integration-{user_id}@example.com"},
        )
        session.execute(
            text("INSERT INTO chat_threads (id, user_id, title) VALUES (:id, :user_id, 'test')"),
            {"id": thread_id, "user_id": user_id},
        )
        first = session.execute(
            text(
                "SELECT position FROM append_chat_message_atomic("
                ":thread_id, 'user', 'first', '{}'::jsonb)"
            ),
            {"thread_id": thread_id},
        ).scalar_one()
        second = session.execute(
            text(
                "SELECT position FROM append_chat_message_atomic("
                ":thread_id, 'user', 'second', '{}'::jsonb)"
            ),
            {"thread_id": thread_id},
        ).scalar_one()

        assert (first, second) == (0, 1)

        before = session.execute(
            text("SELECT COUNT(*) FROM chat_messages WHERE thread_id = :thread_id"),
            {"thread_id": thread_id},
        ).scalar_one()
        savepoint = session.begin_nested()
        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    "SELECT * FROM append_grounded_answer_atomic("
                    ":thread_id, 'bad citation', '{}'::jsonb, "
                    "CAST(:citations AS jsonb))"
                ),
                {
                    "thread_id": thread_id,
                    "citations": json.dumps(
                        [
                            {
                                "chunk_id": str(uuid4()),
                                "citation_index": 0,
                                "quoted_text": "missing chunk",
                            }
                        ]
                    ),
                },
            ).all()
        savepoint.rollback()
        after = session.execute(
            text("SELECT COUNT(*) FROM chat_messages WHERE thread_id = :thread_id"),
            {"thread_id": thread_id},
        ).scalar_one()

        assert after == before
        session.rollback()


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parents[1] / "fixtures/retrieval_eval.json").read_text(encoding="utf-8")
    ),
)
def test_live_hybrid_retrieval_meets_smoke_expectations(case):
    retriever = DocumentRetriever(
        session_factory=create_sessionmaker(),
        embedder=OpenAIQueryEmbedder(AsyncOpenAI(api_key=settings.OPENAI_API_KEY)),
    )
    passages = asyncio.run(
        retriever.retrieve(
            case["query"],
            top_k=5,
            candidate_k=30,
            filters=RetrievalFilters(tickers=(case["ticker"],)),
        )
    )

    assert passages
    assert all(passage.center.chunk.ticker == case["ticker"] for passage in passages)
    retrieved_chunks = []
    for passage in passages:
        retrieved_chunks.extend(passage.previous_chunks)
        retrieved_chunks.append(passage.center.chunk)
        retrieved_chunks.extend(passage.next_chunks)
    combined = " ".join(chunk.text.casefold() for chunk in retrieved_chunks)
    assert all(keyword.casefold() in combined for keyword in case["keywords"])
