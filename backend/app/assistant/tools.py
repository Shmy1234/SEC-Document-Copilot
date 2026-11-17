"""Tools the deep-research loop can call, and how each one runs.

Tool schemas are plain Pydantic models handed to the model as OpenAI function
specs; execution stays here so every tool registers evidence with the turn's
DocumentAgentDeps and the grounding contract can trust it.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import CitationDraft
from app.chat.stages import RetrievalError
from app.grounding.evidence import EvidenceCandidate
from app.retrieval.schemas import SourcePassage

NO_MATCHES = (
    "No matching filing passages were found. Stop searching and report insufficient evidence."
)
NO_CHUNK = "No chunk was found for that id."

ToolResult = list[EvidenceCandidate] | str


class SearchFilingsArgs(BaseModel):
    query: str = Field(min_length=1, description="Search wording for the filing corpus.")
    top_k: int = Field(default=5, ge=1, description="Passages to return; capped at 10.")


class ReadChunkArgs(BaseModel):
    chunk_id: UUID


class ReadSurroundingChunksArgs(BaseModel):
    chunk_id: UUID
    window: int = Field(default=1, ge=0, le=3)


class SubmitAnswerArgs(BaseModel):
    answer: str = Field(description="Final answer text, without citation notation or ids.")
    citations: list[CitationDraft] = Field(
        default_factory=list,
        description="Evidence ids whose exact quotes support every factual sentence.",
    )


SEARCH_FILINGS = "search_filings"
READ_CHUNK = "read_chunk"
READ_SURROUNDING_CHUNKS = "read_surrounding_chunks"
SUBMIT_ANSWER = "submit_answer"

TOOL_ARGS: dict[str, type[BaseModel]] = {
    SEARCH_FILINGS: SearchFilingsArgs,
    READ_CHUNK: ReadChunkArgs,
    READ_SURROUNDING_CHUNKS: ReadSurroundingChunksArgs,
    SUBMIT_ANSWER: SubmitAnswerArgs,
}

_DESCRIPTIONS = {
    SEARCH_FILINGS: "Hybrid search over the SEC filing corpus. Returns evidence candidates.",
    READ_CHUNK: "Read one chunk by id and return evidence candidates from it.",
    READ_SURROUNDING_CHUNKS: "Read a chunk and its neighbours to recover missing context.",
    SUBMIT_ANSWER: "Submit the final answer with the evidence ids that support it.",
}


def _spec(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _DESCRIPTIONS[name],
            "parameters": TOOL_ARGS[name].model_json_schema(),
        },
    }


RESEARCH_TOOL_SPECS = [_spec(name) for name in TOOL_ARGS]


async def search_filings(
    deps: DocumentAgentDeps,
    args: SearchFilingsArgs,
    *,
    prefetched: list[SourcePassage] | None = None,
) -> ToolResult:
    effective_top_k = min(args.top_k, 10)
    cache_key = (" ".join(args.query.split()).casefold(), effective_top_k)
    candidates = deps.search_cache.get(cache_key)
    if candidates is None:
        if prefetched is not None:
            passages = prefetched
        else:
            try:
                passages = await deps.retriever.retrieve(
                    args.query,
                    top_k=effective_top_k,
                    candidate_k=50,
                    filters=deps.retrieval_filters,
                )
            except Exception as error:
                # Embeddings and pgvector are external boundaries; surface a typed
                # failure so the stream can report retrieval rather than processing.
                raise RetrievalError("filing search failed") from error
        deps.add_passages(passages)
        candidates = deps.register_passage_evidence(passages, args.query)
        deps.search_cache[cache_key] = candidates
    deps.retrieval_completed = True
    await deps.on_stage("analyzing")
    return candidates or NO_MATCHES


async def read_chunk(deps: DocumentAgentDeps, args: ReadChunkArgs) -> ToolResult:
    try:
        chunk = await deps.retriever.read_chunk(args.chunk_id)
    except Exception as error:
        raise RetrievalError("chunk read failed") from error
    if chunk is None:
        return NO_CHUNK
    candidates = deps.register_chunk_evidence([chunk])
    return candidates or "The chunk did not contain evidence matching the current question."


async def read_surrounding_chunks(
    deps: DocumentAgentDeps,
    args: ReadSurroundingChunksArgs,
) -> ToolResult:
    try:
        result = await deps.retriever.read_surrounding_chunks(args.chunk_id, window=args.window)
        if result is None:
            return NO_CHUNK
        center = await deps.retriever.read_chunk(args.chunk_id)
    except Exception as error:
        raise RetrievalError("surrounding chunk read failed") from error
    chunks = result["previous_chunks"] + result["next_chunks"]
    if center is not None:
        chunks.insert(0, center)
    candidates = deps.register_chunk_evidence(chunks)
    return candidates or (
        "The surrounding chunks did not contain evidence matching the current question."
    )


async def run_tool(deps: DocumentAgentDeps, name: str, args: BaseModel) -> ToolResult:
    if isinstance(args, SearchFilingsArgs):
        return await search_filings(deps, args)
    if isinstance(args, ReadChunkArgs):
        return await read_chunk(deps, args)
    if isinstance(args, ReadSurroundingChunksArgs):
        return await read_surrounding_chunks(deps, args)
    raise ValueError(f"{name} is not an evidence tool")
