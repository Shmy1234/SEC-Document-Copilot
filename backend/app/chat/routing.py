import re
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, model_validator

ChatRoute = Literal["instant", "direct", "quick_rag", "deep_rag"]
ModelChatRoute = Literal["direct", "quick_rag", "deep_rag"]

ROUTING_INSTRUCTIONS = """You route requests for an SEC filing research assistant.

Choose direct only for guidance about how to use this product and its filing-research features.
When choosing direct, provide the concise final answer in the answer field.
A request that mixes product guidance with a question about what the corpus contains is still
direct: describe what the assistant does and invite a specific company-and-period question.

Choose quick_rag for a focused company or filing question likely answerable with one corpus search.
Choose deep_rag for comparisons, multi-part synthesis, longitudinal analysis, broad summaries, or
questions likely to require multiple searches or surrounding context.

Never choose direct for company, filing, business, financial, dated, comparative, source,
verification, or citation facts. Ambiguous factual requests must use quick_rag or deep_rag.
RAG routes must leave the answer field empty.
"""

_TRAILING_PUNCTUATION = re.compile(r"[.!?]+$")
_SAFE_DIRECT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        (
            r"^(?:what can you (?:do|help me (?:do|with))|what do you do|"
            r"how can you help(?: me)?|how do i use (?:this|the) (?:app|workspace)|"
            r"help me use (?:this|the) (?:app|workspace))\??$"
        ),
    )
)
_FILING_FACT = re.compile(
    r"\b(?:apple|microsoft|nvidia|amazon|alphabet|google|aapl|msft|nvda|amzn|googl|"
    r"revenue|sales|income|earnings|eps|margin|cash flow|guidance|fiscal|"
    r"10-k|10-q|8-k|q[1-4]|20\d{2})\b"
)
# Questions about what the workspace covers. Answered from a fixed description of
# scope rather than retrieval: the corpus is not itself a citable passage, so RAG
# can only fail the grounding contract on these.
_CORPUS_QUESTION = re.compile(
    r"\b(?:what (?:filings|documents|companies|data)|which (?:filings|companies)|"
    r"what (?:are you|were you|have you been) trained on|what(?:'s| is) in (?:the|your) corpus|"
    r"what do you (?:cover|know about)|how (?:much|many) (?:filings|documents))\b"
)
_CORPUS_ANSWER = (
    "I answer questions from the SEC filings loaded into this workspace, quoting the "
    "passages an answer rests on. I can't list the corpus back to you, so the quickest "
    "way to see what's here is to ask about a specific company and period — for example, "
    "\"What were Apple's total net sales in fiscal 2025?\" If a filing isn't loaded, "
    "you'll get an explicit no-evidence answer rather than a guess."
)
_INSTANT_RESPONSES = {
    "hi": "Hi! How can I help with your filing research?",
    "hello": "Hello! How can I help with your filing research?",
    "hey": "Hey! How can I help with your filing research?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
}


class RouteDecision(BaseModel):
    route: ChatRoute
    answer: str | None = None

    @model_validator(mode="after")
    def require_route_appropriate_answer(self) -> "RouteDecision":
        has_answer = bool(self.answer and self.answer.strip())
        if self.route in {"instant", "direct"} and not has_answer:
            raise ValueError("instant and direct routes require an answer")
        if self.route in {"quick_rag", "deep_rag"} and has_answer:
            raise ValueError("RAG routes must not include an answer")
        return self


class ModelRouteDecision(BaseModel):
    route: ModelChatRoute
    answer: str | None = None

    @model_validator(mode="after")
    def require_route_appropriate_answer(self) -> "ModelRouteDecision":
        has_answer = bool(self.answer and self.answer.strip())
        if self.route == "direct" and not has_answer:
            raise ValueError("direct routes require an answer")
        if self.route in {"quick_rag", "deep_rag"} and has_answer:
            raise ValueError("RAG routes must not include an answer")
        return self


def _normalize(prompt: str) -> str:
    normalized = " ".join(prompt.split()).casefold()
    return _TRAILING_PUNCTUATION.sub("", normalized).strip()


def instant_response(prompt: str) -> str | None:
    normalized = _normalize(prompt)
    if normalized in _INSTANT_RESPONSES:
        return _INSTANT_RESPONSES[normalized]
    if _CORPUS_QUESTION.search(normalized) and not _FILING_FACT.search(normalized):
        return _CORPUS_ANSWER
    return None


def direct_response_allowed(prompt: str) -> bool:
    """Gate on the presence of a filing fact, not on recognised help phrasing.

    The model already declines `direct` for research questions. The gate exists to
    catch the cases where it doesn't, so it only has to reject prompts carrying a
    company, metric, form, or period. Requiring a phrasing allowlist instead sent
    compound questions like "what can you do, and what filings do you have?"
    through retrieval, where they could only fail grounding.
    """
    normalized = _normalize(prompt)
    if any(pattern.match(normalized) for pattern in _SAFE_DIRECT_PATTERNS):
        return True
    return not _FILING_FACT.search(normalized)


def routing_messages(prompt: str) -> list[BaseMessage]:
    return [SystemMessage(ROUTING_INSTRUCTIONS), HumanMessage(prompt)]


async def route_question(
    prompt: str,
    route_model: Runnable[list[BaseMessage], ModelRouteDecision],
) -> RouteDecision:
    response = instant_response(prompt)
    if response is not None:
        return RouteDecision(route="instant", answer=response)

    decision = await route_model.ainvoke(routing_messages(prompt))
    if decision.route == "direct" and not direct_response_allowed(prompt):
        return RouteDecision(route="quick_rag")
    return RouteDecision(route=decision.route, answer=decision.answer)
