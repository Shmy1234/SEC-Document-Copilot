import asyncio
from contextlib import contextmanager
from datetime import date
from uuid import UUID

from app.retrieval.retriever import DocumentRetriever
from app.retrieval.schemas import ChunkReference, RankedChunk, RetrievalFilters


class FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        assert query == "Apple revenue mix"
        return [0.1, 0.2, 0.3]


@contextmanager
def fake_session_factory():
    yield object()


def make_chunk(number: int) -> ChunkReference:
    return ChunkReference(
        chunk_id=UUID(f"00000000-0000-0000-0000-{number:012d}"),
        document_id=UUID("00000000-0000-0000-0000-999999999999"),
        chunk_index=number,
        text=f"chunk {number}",
        page_number=number,
        section="Results of Operations",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="0000320193-25-000079",
        source_url="https://www.sec.gov/example",
    )


def test_retrieve_runs_both_queries_fuses_and_attaches_neighbors(monkeypatch):
    calls = []

    def fake_semantic_search(session, query_embedding, *, limit, filters):
        calls.append(("semantic", query_embedding, limit, filters))
        return [RankedChunk(chunk=make_chunk(1), rank=1, score=0.9, source="semantic")]

    def fake_full_text_search(session, query, *, limit, filters):
        calls.append(("full_text", query, limit, filters))
        return [RankedChunk(chunk=make_chunk(1), rank=1, score=2.5, source="full_text")]

    def fake_neighbors(session, chunks, *, window):
        assert window == 1
        calls.append(("neighbors", [chunk.chunk_id for chunk in chunks]))
        return {chunk.chunk_id: ([make_chunk(0)], [make_chunk(2)]) for chunk in chunks}

    monkeypatch.setattr("app.retrieval.retriever.semantic_search", fake_semantic_search)
    monkeypatch.setattr("app.retrieval.retriever.full_text_search", fake_full_text_search)
    monkeypatch.setattr("app.retrieval.retriever.read_neighbors_for_many", fake_neighbors)

    retriever = DocumentRetriever(session_factory=fake_session_factory, embedder=FakeEmbedder())
    passages = asyncio.run(
        retriever.retrieve(
            "Apple revenue mix",
            top_k=5,
            candidate_k=20,
            filters=RetrievalFilters(tickers=("aapl",)),
        )
    )

    assert len(passages) == 1
    assert passages[0].center.semantic_rank == 1
    assert passages[0].center.full_text_rank == 1
    assert passages[0].previous_chunks[0].chunk_index == 0
    assert passages[0].next_chunks[0].chunk_index == 2
    assert {call[0] for call in calls[:2]} == {"semantic", "full_text"}
    assert calls[2] == ("neighbors", [make_chunk(1).chunk_id])


def test_retriever_reads_a_chunk_without_blocking_the_event_loop(monkeypatch):
    expected = make_chunk(1)

    def fake_read_chunk(session, chunk_id):
        assert chunk_id == expected.chunk_id
        return expected

    monkeypatch.setattr("app.retrieval.retriever.read_chunk", fake_read_chunk)
    retriever = DocumentRetriever(session_factory=fake_session_factory, embedder=FakeEmbedder())

    result = asyncio.run(retriever.read_chunk(expected.chunk_id))

    assert result == expected


def test_retriever_reads_surrounding_chunks(monkeypatch):
    expected = make_chunk(1)

    monkeypatch.setattr("app.retrieval.retriever.read_chunk", lambda session, chunk_id: expected)

    def fake_neighbors(session, chunk, *, window):
        assert chunk == expected
        assert window == 2
        return [make_chunk(0)], [make_chunk(2)]

    monkeypatch.setattr("app.retrieval.retriever.read_surrounding_chunks", fake_neighbors)
    retriever = DocumentRetriever(session_factory=fake_session_factory, embedder=FakeEmbedder())

    result = asyncio.run(retriever.read_surrounding_chunks(expected.chunk_id, window=2))

    assert result == {"previous_chunks": [make_chunk(0)], "next_chunks": [make_chunk(2)]}


def test_full_text_search_runs_while_the_query_is_embedded(monkeypatch):
    import threading

    full_text_done = threading.Event()

    class SlowEmbedder:
        async def embed_query(self, query):
            # Full-text must finish without waiting for the embedding.
            for _ in range(200):
                if full_text_done.is_set():
                    return [0.1]
                await asyncio.sleep(0.005)
            raise AssertionError("full-text search waited for the embedding")

    def fake_full_text(session, query, *, limit, filters):
        full_text_done.set()
        return []

    monkeypatch.setattr("app.retrieval.retriever.full_text_search", fake_full_text)
    monkeypatch.setattr("app.retrieval.retriever.semantic_search", lambda *a, **k: [])
    monkeypatch.setattr("app.retrieval.retriever.read_neighbors_for_many", lambda *a, **k: {})
    retriever = DocumentRetriever(session_factory=fake_session_factory, embedder=SlowEmbedder())

    assert asyncio.run(retriever.retrieve("q", top_k=1, candidate_k=5)) == []


def test_neighbors_for_all_results_are_read_in_one_query(monkeypatch):
    batches = []

    def fake_neighbors(session, chunks, *, window):
        batches.append(len(chunks))
        return {chunk.chunk_id: ([], []) for chunk in chunks}

    ranked = [
        RankedChunk(chunk=make_chunk(n), rank=n, score=1.0, source="semantic") for n in range(1, 6)
    ]
    monkeypatch.setattr("app.retrieval.retriever.semantic_search", lambda *a, **k: ranked)
    monkeypatch.setattr("app.retrieval.retriever.full_text_search", lambda *a, **k: [])
    monkeypatch.setattr("app.retrieval.retriever.read_neighbors_for_many", fake_neighbors)
    retriever = DocumentRetriever(session_factory=fake_session_factory, embedder=FakeEmbedder())

    passages = asyncio.run(retriever.retrieve("Apple revenue mix", top_k=5, candidate_k=5))

    assert len(passages) == 5
    assert batches == [5]
