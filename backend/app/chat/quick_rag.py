"""One-pass RAG: one search (already run during routing), one model call."""

import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.assistant.deps import StageCallback
from app.assistant.outputs import AgentAnswer, GroundedAnswer
from app.grounding.evidence import build_evidence_candidates
from app.grounding.validator import GroundingValidator
from app.retrieval.schemas import SourcePassage

INSUFFICIENT_EVIDENCE_ANSWER = "The corpus does not contain enough evidence to answer that question."

QUICK_RAG_INSTRUCTIONS = """Answer the question only from the supplied evidence candidates.
Return concise text and select the evidence ids whose exact quotes support every factual sentence.
Never invent facts, figures, dates, calculations, or citations. If the candidates are insufficient,
say the corpus does not contain enough evidence and return no citations. Do not put citation notation
or evidence ids in the answer text.
"""


async def answer_from_passages(
    question: str,
    passages: Sequence[SourcePassage],
    *,
    answer_model: Runnable[list[BaseMessage], AgentAnswer],
    grounding_validator: GroundingValidator,
    report_stage: StageCallback,
) -> GroundedAnswer:
    candidates = build_evidence_candidates(passages, question)
    if not candidates:
        return GroundedAnswer(answer=INSUFFICIENT_EVIDENCE_ANSWER)

    await report_stage("analyzing")
    payload = {
        "question": question,
        "evidence": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    draft = await answer_model.ainvoke(
        [SystemMessage(QUICK_RAG_INSTRUCTIONS), HumanMessage(json.dumps(payload))]
    )
    await report_stage("validating")
    return grounding_validator.validate(
        draft,
        passages,
        evidence_candidates={candidate.evidence_id: candidate for candidate in candidates},
    )
