import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager
from uuid import UUID

from sqlalchemy.orm import Session

from app.retrieval.embeddings import QueryEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.queries import (
    full_text_search,
    read_chunk,
    read_surrounding_chunks,
    semantic_search,
)
from app.retrieval.schemas import ChunkReference, FusedChunk, RetrievalFilters, SourcePassage


class DocumentRetriever:
    def __init__(
        self,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]],
        embedder: QueryEmbedder,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int = 50,
        filters: RetrievalFilters | None = None,
        neighbor_window: int = 1,
    ) -> list[SourcePassage]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must not be negative")

        query_embedding = await self._embedder.embed_query(query)
        return await asyncio.to_thread(
            self._retrieve_sync,
            query,
            query_embedding,
            top_k,
            candidate_k,
            filters or RetrievalFilters(),
            neighbor_window,
        )

    async def read_chunk(self, chunk_id: UUID) -> ChunkReference | None:
        return await asyncio.to_thread(self._read_chunk_sync, chunk_id)

    def _read_chunk_sync(self, chunk_id: UUID) -> ChunkReference | None:
        with self._session_factory() as session:
            return read_chunk(session, chunk_id)

    async def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> dict[str, list[ChunkReference]] | None:
        return await asyncio.to_thread(self._read_surrounding_chunks_sync, chunk_id, window)

    def _read_surrounding_chunks_sync(
        self,
        chunk_id: UUID,
        window: int,
    ) -> dict[str, list[ChunkReference]] | None:
        with self._session_factory() as session:
            chunk = read_chunk(session, chunk_id)
            if chunk is None:
                return None
            previous, next_chunks = read_surrounding_chunks(session, chunk, window=window)
            return {"previous_chunks": previous, "next_chunks": next_chunks}

    def _retrieve_sync(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        candidate_k: int,
        filters: RetrievalFilters,
        neighbor_window: int,
    ) -> list[SourcePassage]:
        with self._session_factory() as session:
            semantic = semantic_search(
                session,
                query_embedding,
                limit=candidate_k,
                filters=filters,
            )
            full_text = full_text_search(
                session,
                query,
                limit=candidate_k,
                filters=filters,
            )
            fused = reciprocal_rank_fusion([semantic, full_text])[:top_k]
            return [
                self._passage_for_fused(session, item, neighbor_window)
                for item in fused
            ]

    @staticmethod
    def _passage_for_fused(
        session: Session,
        item: FusedChunk,
        neighbor_window: int,
    ) -> SourcePassage:
        previous, next_chunks = read_surrounding_chunks(
            session,
            item.chunk,
            window=neighbor_window,
        )
        return SourcePassage(
            center=item,
            previous_chunks=previous,
            next_chunks=next_chunks,
        )
