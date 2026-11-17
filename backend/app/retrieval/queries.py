from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from sqlalchemy import Select, and_, desc, func, or_, select
from sqlalchemy.orm import Session

from app.database.models.documents import DocumentChunk, DocumentTable, SourceDocument
from app.retrieval.normalization import normalize_full_text_query
from app.retrieval.schemas import ChunkReference, RankedChunk, RetrievalFilters


def _base_columns():
    return (
        DocumentChunk.id.label("chunk_id"),
        DocumentChunk.document_id.label("document_id"),
        DocumentChunk.chunk_index,
        DocumentChunk.text,
        DocumentChunk.page_number,
        DocumentChunk.section,
        DocumentChunk.kind,
        DocumentChunk.table_id,
        DocumentChunk.row_start,
        DocumentChunk.row_end,
        DocumentChunk.source_locator,
        DocumentTable.title.label("table_title"),
        DocumentTable.units.label("table_units"),
        SourceDocument.ticker,
        SourceDocument.company_name,
        SourceDocument.filing_type,
        SourceDocument.filing_date,
        SourceDocument.fiscal_year,
        SourceDocument.accession_number,
        SourceDocument.source_url,
    )


def apply_filters(statement: Select, filters: RetrievalFilters) -> Select:
    if filters.tickers:
        statement = statement.where(SourceDocument.ticker.in_(filters.tickers))
    if filters.filing_types:
        statement = statement.where(SourceDocument.filing_type.in_(filters.filing_types))
    if filters.fiscal_years:
        statement = statement.where(SourceDocument.fiscal_year.in_(filters.fiscal_years))
    return statement


def chunk_reference_from_row(row) -> ChunkReference:
    return ChunkReference(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        text=row.text,
        page_number=row.page_number,
        section=row.section,
        ticker=row.ticker,
        company_name=row.company_name,
        filing_type=row.filing_type,
        filing_date=row.filing_date,
        fiscal_year=row.fiscal_year,
        accession_number=row.accession_number,
        source_url=row.source_url,
        kind=row.kind,
        table_id=row.table_id,
        table_title=row.table_title,
        table_units=row.table_units,
        row_start=row.row_start,
        row_end=row.row_end,
        source_locator=row.source_locator or {},
    )


def ranked_chunk_from_row(
    row,
    *,
    rank: int,
    source: Literal["semantic", "full_text"],
) -> RankedChunk:
    return RankedChunk(
        chunk=chunk_reference_from_row(row),
        rank=rank,
        score=row.score,
        source=source,
    )


def build_semantic_statement(
    query_embedding: Sequence[float],
    *,
    limit: int,
    filters: RetrievalFilters,
) -> Select:
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(*_base_columns(), (1 - distance).label("score"))
        .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
        .outerjoin(DocumentTable, DocumentTable.id == DocumentChunk.table_id)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    return apply_filters(statement, filters)


def semantic_search(
    session: Session,
    query_embedding: Sequence[float],
    *,
    limit: int,
    filters: RetrievalFilters,
) -> list[RankedChunk]:
    if limit < 1:
        return []
    # pgvector's HNSW scan returns at most hnsw.ef_search rows (default 40), so a
    # larger LIMIT silently came back short. Raise it for this transaction only.
    session.execute(select(func.set_config("hnsw.ef_search", str(max(limit, 40)), True)))
    rows = session.execute(
        build_semantic_statement(query_embedding, limit=limit, filters=filters)
    ).all()
    return [
        ranked_chunk_from_row(row, rank=rank, source="semantic")
        for rank, row in enumerate(rows, start=1)
    ]


def build_full_text_statement(
    query: str,
    *,
    limit: int,
    filters: RetrievalFilters,
) -> Select:
    ts_query = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(DocumentChunk.search_vector, ts_query)
    statement = (
        select(*_base_columns(), rank.label("score"))
        .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
        .outerjoin(DocumentTable, DocumentTable.id == DocumentChunk.table_id)
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(desc(rank))
        .limit(limit)
    )
    return apply_filters(statement, filters)


def full_text_search(
    session: Session,
    query: str,
    *,
    limit: int,
    filters: RetrievalFilters,
) -> list[RankedChunk]:
    normalized = normalize_full_text_query(query)
    if limit < 1 or not normalized:
        return []
    rows = session.execute(
        build_full_text_statement(normalized, limit=limit, filters=filters)
    ).all()
    return [
        ranked_chunk_from_row(row, rank=rank, source="full_text")
        for rank, row in enumerate(rows, start=1)
    ]


def build_read_chunk_statement(chunk_id: UUID) -> Select:
    return (
        select(*_base_columns())
        .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
        .outerjoin(DocumentTable, DocumentTable.id == DocumentChunk.table_id)
        .where(DocumentChunk.id == chunk_id)
    )


def read_chunk(session: Session, chunk_id: UUID) -> ChunkReference | None:
    row = session.execute(build_read_chunk_statement(chunk_id)).one_or_none()
    return chunk_reference_from_row(row) if row is not None else None


def read_surrounding_chunks(
    session: Session,
    chunk: ChunkReference,
    *,
    window: int,
) -> tuple[list[ChunkReference], list[ChunkReference]]:
    if window < 1:
        return [], []
    statement = (
        select(*_base_columns())
        .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
        .outerjoin(DocumentTable, DocumentTable.id == DocumentChunk.table_id)
        .where(DocumentChunk.document_id == chunk.document_id)
        .where(DocumentChunk.chunk_index >= chunk.chunk_index - window)
        .where(DocumentChunk.chunk_index <= chunk.chunk_index + window)
        .where(DocumentChunk.id != chunk.chunk_id)
        .order_by(DocumentChunk.chunk_index)
    )
    if chunk.table_id is not None:
        statement = statement.where(DocumentChunk.table_id == chunk.table_id)
    neighbors = [chunk_reference_from_row(row) for row in session.execute(statement).all()]
    previous = [item for item in neighbors if item.chunk_index < chunk.chunk_index]
    next_chunks = [item for item in neighbors if item.chunk_index > chunk.chunk_index]
    return previous, next_chunks


def read_neighbors_for_many(
    session: Session,
    chunks: Sequence[ChunkReference],
    *,
    window: int,
) -> dict[UUID, tuple[list[ChunkReference], list[ChunkReference]]]:
    """Neighbours for many centre chunks in one round trip.

    Same rules as read_surrounding_chunks: same document, within `window`
    positions, and inside the same table when the centre chunk is a table row.
    """
    if window < 1 or not chunks:
        return {chunk.chunk_id: ([], []) for chunk in chunks}
    ranges = [
        and_(
            DocumentChunk.document_id == chunk.document_id,
            DocumentChunk.chunk_index >= chunk.chunk_index - window,
            DocumentChunk.chunk_index <= chunk.chunk_index + window,
        )
        for chunk in chunks
    ]
    statement = (
        select(*_base_columns())
        .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
        .outerjoin(DocumentTable, DocumentTable.id == DocumentChunk.table_id)
        .where(or_(*ranges))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    rows = [chunk_reference_from_row(row) for row in session.execute(statement).all()]
    return {chunk.chunk_id: _split_neighbors(chunk, rows, window) for chunk in chunks}


def _split_neighbors(
    center: ChunkReference,
    candidates: Sequence[ChunkReference],
    window: int,
) -> tuple[list[ChunkReference], list[ChunkReference]]:
    neighbors = [
        item
        for item in candidates
        if item.document_id == center.document_id
        and item.chunk_id != center.chunk_id
        and abs(item.chunk_index - center.chunk_index) <= window
        and (center.table_id is None or item.table_id == center.table_id)
    ]
    previous = [item for item in neighbors if item.chunk_index < center.chunk_index]
    following = [item for item in neighbors if item.chunk_index > center.chunk_index]
    return previous, following
