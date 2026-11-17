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
    read_neighbors_for_many,
    read_surrounding_chunks,
    semantic_search,
)
from app.retrieval.schemas import ChunkReference, RankedChunk, RetrievalFilters, SourcePassage


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

        active_filters = filters or RetrievalFilters()
        # Full-text search needs no embedding, so it runs while the query is
        # embedded; each search uses its own pooled session.
        semantic, full_text = await asyncio.gather(
            self._semantic(query, candidate_k, active_filters),
            asyncio.to_thread(self._full_text_sync, query, candidate_k, active_filters),
        )
        fused = reciprocal_rank_fusion([semantic, full_text])[:top_k]
        neighbors = await asyncio.to_thread(
            self._neighbors_sync, [item.chunk for item in fused], neighbor_window
        )
        return [
            SourcePassage(
                center=item,
                previous_chunks=neighbors[item.chunk.chunk_id][0],
                next_chunks=neighbors[item.chunk.chunk_id][1],
            )
            for item in fused
        ]

    async def _semantic(
        self, query: str, limit: int, filters: RetrievalFilters
    ) -> list[RankedChunk]:
        query_embedding = await self._embedder.embed_query(query)
        return await asyncio.to_thread(self._semantic_sync, query_embedding, limit, filters)

    def _semantic_sync(
        self, query_embedding: list[float], limit: int, filters: RetrievalFilters
    ) -> list[RankedChunk]:
        with self._session_factory() as session:
            return semantic_search(session, query_embedding, limit=limit, filters=filters)

    def _full_text_sync(
        self, query: str, limit: int, filters: RetrievalFilters
    ) -> list[RankedChunk]:
        with self._session_factory() as session:
            return full_text_search(session, query, limit=limit, filters=filters)

    def _neighbors_sync(
        self, chunks: list[ChunkReference], window: int
    ) -> dict[UUID, tuple[list[ChunkReference], list[ChunkReference]]]:
        with self._session_factory() as session:
            return read_neighbors_for_many(session, chunks, window=window)

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
