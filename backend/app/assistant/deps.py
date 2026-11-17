from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol
from uuid import UUID

from app.chat.stages import Stage
from app.grounding.evidence import EvidenceCandidate, build_evidence_candidates
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.schemas import ChunkReference, FusedChunk, RetrievalFilters, SourcePassage

StageCallback = Callable[[Stage], Awaitable[None]]


async def ignore_stage(stage: Stage) -> None:
    """Default for callers that do not stream progress, such as tests and smoke runs."""


class GroundingValidator(Protocol):
    def validate(
        self,
        answer,
        passages: Sequence[SourcePassage],
        *,
        evidence_candidates: dict[UUID, EvidenceCandidate] | None = None,
    ):
        """Validate an answer against the current retrieved passages."""


class DocumentAgentDeps:
    """Evidence gathered during one turn: what was retrieved and what may be cited."""

    def __init__(
        self,
        *,
        retriever: DocumentRetriever,
        grounding_validator: GroundingValidator,
        retrieved_passages: list[SourcePassage] | None = None,
        retrieval_filters: RetrievalFilters | None = None,
        on_stage: StageCallback | None = None,
    ) -> None:
        self.retriever = retriever
        self.grounding_validator = grounding_validator
        self.on_stage = on_stage or ignore_stage
        self.retrieved_passages = retrieved_passages if retrieved_passages is not None else []
        self.retrieval_filters = retrieval_filters or RetrievalFilters()
        self.search_cache: dict[tuple[str, int], list[EvidenceCandidate]] = {}
        self.evidence_candidates: dict[UUID, EvidenceCandidate] = {}
        self.evidence_query = ""
        self.retrieval_completed = False

    def add_passages(self, passages: Sequence[SourcePassage]) -> None:
        known_centers = {passage.center.chunk.chunk_id for passage in self.retrieved_passages}
        for passage in passages:
            if passage.center.chunk.chunk_id not in known_centers:
                self.retrieved_passages.append(passage)
                known_centers.add(passage.center.chunk.chunk_id)

    def add_chunk(self, chunk: ChunkReference) -> None:
        if self.has_chunk(chunk.chunk_id):
            return
        self.retrieved_passages.append(
            SourcePassage(center=FusedChunk(chunk=chunk, rrf_score=0.0))
        )

    def register_passage_evidence(
        self,
        passages: Sequence[SourcePassage],
        query: str,
    ) -> list[EvidenceCandidate]:
        self.evidence_query = query
        candidates = build_evidence_candidates(passages, query)
        registered: list[EvidenceCandidate] = []
        for candidate in candidates:
            existing = next(
                (
                    item
                    for item in self.evidence_candidates.values()
                    if item.chunk_id == candidate.chunk_id
                    and item.exact_quote == candidate.exact_quote
                ),
                None,
            )
            if existing is not None:
                registered.append(existing)
                continue
            self.evidence_candidates[candidate.evidence_id] = candidate
            registered.append(candidate)
        return registered

    def register_chunk_evidence(
        self,
        chunks: Sequence[ChunkReference],
    ) -> list[EvidenceCandidate]:
        passages = []
        for chunk in chunks:
            self.add_chunk(chunk)
            passages.append(SourcePassage(center=FusedChunk(chunk=chunk, rrf_score=0.0)))
        return self.register_passage_evidence(passages, self.evidence_query)

    def has_chunk(self, chunk_id: UUID) -> bool:
        return any(
            passage.center.chunk.chunk_id == chunk_id
            or any(item.chunk_id == chunk_id for item in passage.previous_chunks)
            or any(item.chunk_id == chunk_id for item in passage.next_chunks)
            for passage in self.retrieved_passages
        )
