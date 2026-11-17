from datetime import date
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy.dialects import postgresql

from app.retrieval.queries import (
    build_full_text_statement,
    build_semantic_statement,
    chunk_reference_from_row,
    ranked_chunk_from_row,
)
from app.retrieval.schemas import RetrievalFilters


def row(score=0.42):
    return SimpleNamespace(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=UUID("00000000-0000-0000-0000-000000000002"),
        chunk_index=3,
        text="Revenue from Services increased.",
        page_number=12,
        section="Results of Operations",
        kind="table",
        table_id=UUID("00000000-0000-0000-0000-000000000003"),
        table_title="Net Sales",
        table_units="USD millions",
        row_start=2,
        row_end=2,
        source_locator={"html_id": "sales"},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="0000320193-25-000079",
        source_url="https://www.sec.gov/example",
        score=score,
    )


def compiled(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_chunk_reference_from_row_maps_database_fields():
    result = chunk_reference_from_row(row())

    assert result.ticker == "AAPL"
    assert result.chunk_index == 3
    assert result.location_label == "page 12, Results of Operations"
    assert result.kind == "table"
    assert result.table_title == "Net Sales"


def test_ranked_chunk_from_row_maps_database_fields():
    result = ranked_chunk_from_row(row(), rank=2, source="semantic")

    assert result.rank == 2
    assert result.score == 0.42
    assert result.source == "semantic"
    assert result.chunk.ticker == "AAPL"
    assert result.chunk.chunk_index == 3


def test_semantic_statement_uses_pgvector_cosine_distance_and_filters():
    statement = build_semantic_statement(
        query_embedding=[0.1, 0.2, 0.3],
        limit=25,
        filters=RetrievalFilters(tickers=("aapl",), fiscal_years=(2025,)),
    )

    sql = compiled(statement)

    assert "document_chunks.embedding <=>" in sql
    assert "source_documents.ticker" in sql
    assert "source_documents.fiscal_year" in sql
    assert "LIMIT" in sql
    assert "document_tables" in sql


def test_full_text_statement_uses_websearch_to_tsquery_and_rank():
    statement = build_full_text_statement(
        query="Apple Services revenue",
        limit=25,
        filters=RetrievalFilters(filing_types=("10-k",)),
    )

    sql = compiled(statement)

    assert "websearch_to_tsquery" in sql
    assert "@@" in sql
    assert "ts_rank_cd" in sql
    assert "source_documents.filing_type" in sql


def neighbor_row(number, *, document=2, table=None):
    values = vars(row()).copy()
    values.update(
        chunk_id=UUID(int=number),
        document_id=UUID(int=document),
        chunk_index=number,
        text=f"chunk {number}",
        table_id=UUID(int=table) if table else None,
    )
    return SimpleNamespace(**values)


class RecordingSession:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(all=lambda: self.rows)


def test_neighbors_for_many_centres_use_one_query_and_respect_scope():
    from app.retrieval.queries import read_neighbors_for_many

    centre_a = chunk_reference_from_row(neighbor_row(10))
    centre_b = chunk_reference_from_row(neighbor_row(20, table=7))
    session = RecordingSession(
        [
            neighbor_row(9),
            neighbor_row(10),
            neighbor_row(11),
            neighbor_row(12),  # outside window 1 of centre_a
            neighbor_row(11, document=3),  # other document
            neighbor_row(19, table=7),
            neighbor_row(21, table=8),  # other table than centre_b's
        ]
    )

    result = read_neighbors_for_many(session, [centre_a, centre_b], window=1)

    assert len(session.statements) == 1
    sql = compiled(session.statements[0])
    assert sql.count("document_chunks.chunk_index >=") == 2
    previous_a, next_a = result[centre_a.chunk_id]
    assert [c.chunk_index for c in previous_a] == [9]
    assert [(c.chunk_index, c.document_id) for c in next_a] == [(11, UUID(int=2))]
    previous_b, next_b = result[centre_b.chunk_id]
    assert [c.chunk_index for c in previous_b] == [19]
    assert next_b == []


def test_neighbors_for_many_skips_the_query_when_nothing_is_needed():
    from app.retrieval.queries import read_neighbors_for_many

    session = RecordingSession([])
    centre = chunk_reference_from_row(neighbor_row(10))

    assert read_neighbors_for_many(session, [centre], window=0) == {centre.chunk_id: ([], [])}
    assert read_neighbors_for_many(session, [], window=1) == {}
    assert session.statements == []


def test_semantic_search_raises_hnsw_ef_search_to_the_requested_limit():
    from app.retrieval.queries import semantic_search

    session = RecordingSession([])

    semantic_search(session, [0.1, 0.2], limit=50, filters=RetrievalFilters())

    setting = str(session.statements[0].compile(compile_kwargs={"literal_binds": True}))
    assert "set_config('hnsw.ef_search', '50', true)" in setting
    assert len(session.statements) == 2
