import asyncio
import json
from datetime import date
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from app.assistant.outputs import AgentAnswer, CitationDraft
from app.chat.quick_rag import INSUFFICIENT_EVIDENCE_ANSWER, answer_from_passages
from app.grounding.validator import GroundingError, GroundingValidator
from app.retrieval.schemas import ChunkReference, FusedChunk, SourcePassage


def passage() -> SourcePassage:
    chunk = ChunkReference(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=uuid4(),
        chunk_index=1,
        text="Services revenue increased 14% in fiscal 2025.",
        page_number=42,
        section="Results",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        accession_number="0000320193-25-000079",
        source_url="https://www.sec.gov/example",
    )
    return SourcePassage(center=FusedChunk(chunk=chunk, rrf_score=1.0))


class EvidenceAnswerModel:
    def __init__(self, answer: str = "Services revenue increased 14% in fiscal 2025."):
        self.answer = answer
        self.calls = []

    async def ainvoke(self, messages):
        system, human = messages
        assert isinstance(system, SystemMessage)
        assert isinstance(human, HumanMessage)
        payload = json.loads(human.content)
        self.calls.append(payload)
        return AgentAnswer(
            answer=self.answer,
            citations=[CitationDraft(evidence_id=payload["evidence"][0]["evidence_id"])],
        )


def answer(passages, model, stages=None):
    async def report(stage):
        if stages is not None:
            stages.append(stage)

    return asyncio.run(
        answer_from_passages(
            "What happened to Services revenue?",
            passages,
            answer_model=model,
            grounding_validator=GroundingValidator(),
            report_stage=report,
        )
    )


def test_quick_rag_makes_one_model_call_and_returns_validated_citation():
    model = EvidenceAnswerModel()
    stages = []

    result = answer([passage()], model, stages)

    assert len(model.calls) == 1
    assert model.calls[0]["question"] == "What happened to Services revenue?"
    assert model.calls[0]["evidence"][0]["exact_quote"] == passage().center.chunk.text
    assert result.citations[0].chunk_id == passage().center.chunk.chunk_id
    assert result.citations[0].quoted_text == passage().center.chunk.text
    assert stages == ["analyzing", "validating"]


def test_quick_rag_empty_result_skips_answer_model():
    model = EvidenceAnswerModel()

    result = answer([], model)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []
    assert model.calls == []


def test_quick_rag_rejects_unsupported_model_answer():
    unsupported = RunnableLambda(
        lambda _messages: AgentAnswer(answer="Unsupported claim.", citations=[])
    )

    with pytest.raises(GroundingError):
        answer([passage()], unsupported)


def test_quick_rag_rejects_invented_evidence_ids():
    invented = RunnableLambda(
        lambda _messages: AgentAnswer(
            answer="Services revenue increased 14% in fiscal 2025.",
            citations=[CitationDraft(evidence_id=uuid4())],
        )
    )

    with pytest.raises(GroundingError):
        answer([passage()], invented)


def test_quick_rag_rejects_numbers_the_evidence_does_not_state():
    model = EvidenceAnswerModel(answer="Services revenue increased 41% in fiscal 2025.")

    with pytest.raises(GroundingError):
        answer([passage()], model)
