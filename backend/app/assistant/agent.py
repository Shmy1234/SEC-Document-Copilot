"""Deep research: a tool-calling loop over the filing corpus, as a LangGraph graph.

The model searches and reads until it calls submit_answer. That answer is held
to the grounding contract before the loop ends; one failed check is sent back
to the model to fix, a second fails the turn.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import AgentAnswer, GroundedAnswer
from app.assistant.tools import (
    SEARCH_FILINGS,
    SUBMIT_ANSWER,
    TOOL_ARGS,
    SearchFilingsArgs,
    ToolResult,
    run_tool,
    search_filings,
)
from app.grounding.validator import GroundingError
from app.retrieval.schemas import SourcePassage

INSTRUCTIONS = Path(__file__).with_name("instructions.md").read_text()
PREFETCH_CALL_ID = "prefetched-search"


class ResearchLimitExceeded(RuntimeError):
    """The model used more requests or tool calls than one turn allows."""


@dataclass(frozen=True)
class ResearchLimits:
    request_limit: int = 6
    tool_calls_limit: int = 8
    grounding_retries: int = 1


@dataclass
class ResearchContext:
    model: Runnable[list[AnyMessage], AIMessage]
    deps: DocumentAgentDeps
    limits: ResearchLimits


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    requests: int
    tool_calls: int
    grounding_failures: int
    answer: GroundedAnswer


async def call_model(state: ResearchState, runtime: Runtime[ResearchContext]) -> ResearchState:
    requests = state.get("requests", 0)
    if requests >= runtime.context.limits.request_limit:
        raise ResearchLimitExceeded("model request limit reached")
    reply = await runtime.context.model.ainvoke(
        [SystemMessage(INSTRUCTIONS), *state["messages"]]
    )
    if not reply.tool_calls:
        # Tools are forced, so a bare text reply means the provider ignored the
        # contract; there is no structured answer to ground.
        raise ResearchLimitExceeded("model replied without calling a tool")
    return {"messages": [reply], "requests": requests + 1}


async def run_tools(state: ResearchState, runtime: Runtime[ResearchContext]) -> ResearchState:
    context = runtime.context
    # The newest model reply, found by type: add_messages merges messages by id,
    # so position alone is not a safe way to find it.
    calls = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage)).tool_calls
    evidence_calls = [call for call in calls if call["name"] != SUBMIT_ANSWER]
    tool_calls = state.get("tool_calls", 0) + len(evidence_calls)
    if tool_calls > context.limits.tool_calls_limit:
        raise ResearchLimitExceeded("tool call limit reached")

    # Evidence tools run first and concurrently so a submit in the same step can
    # cite what they register.
    results = await asyncio.gather(*(_evidence_message(context.deps, c) for c in evidence_calls))
    update: ResearchState = {"messages": list(results), "tool_calls": tool_calls}

    for call in calls:
        if call["name"] != SUBMIT_ANSWER:
            continue
        outcome = await _submit(context, state.get("grounding_failures", 0), call)
        update["messages"].append(outcome.message)
        if outcome.answer is not None:
            update["answer"] = outcome.answer
        else:
            update["grounding_failures"] = state.get("grounding_failures", 0) + 1
    return update


def after_tools(state: ResearchState) -> str:
    return END if "answer" in state else "call_model"


async def _evidence_message(deps: DocumentAgentDeps, call: dict) -> ToolMessage:
    args_type = TOOL_ARGS.get(call["name"])
    if args_type is None:
        return _tool_message(call, f"Unknown tool {call['name']}.")
    try:
        args = args_type.model_validate(call["args"])
    except ValidationError as error:
        return _tool_message(call, f"Invalid arguments: {error.errors(include_url=False)}")
    return _tool_message(call, await run_tool(deps, call["name"], args))


@dataclass
class _SubmitOutcome:
    message: ToolMessage
    answer: GroundedAnswer | None


async def _submit(context: ResearchContext, failures: int, call: dict) -> _SubmitOutcome:
    try:
        draft = AgentAnswer.model_validate(call["args"])
    except ValidationError as error:
        problem = f"Invalid answer: {error.errors(include_url=False)}"
    else:
        await context.deps.on_stage("validating")
        try:
            answer = context.deps.grounding_validator.validate(
                draft,
                context.deps.retrieved_passages,
                evidence_candidates=context.deps.evidence_candidates,
            )
        except GroundingError as error:
            problem = str(error)
            if failures >= context.limits.grounding_retries:
                raise
        else:
            return _SubmitOutcome(_tool_message(call, "Answer accepted."), answer)
    return _SubmitOutcome(
        _tool_message(call, f"Answer rejected: {problem}. Fix it and submit again."),
        None,
    )


def _tool_message(call: dict, result: ToolResult | str) -> ToolMessage:
    if isinstance(result, list):
        content = json.dumps([item.model_dump(mode="json") for item in result])
    else:
        content = result
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])


def _build_research_graph():
    graph = StateGraph(ResearchState, context_schema=ResearchContext)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", "run_tools")
    graph.add_conditional_edges("run_tools", after_tools, ["call_model", END])
    return graph.compile(name="deep_research")


research_graph = _build_research_graph()


async def research(
    question: str,
    *,
    model: Runnable[list[AnyMessage], AIMessage],
    deps: DocumentAgentDeps,
    prefetched: list[SourcePassage] | None = None,
    limits: ResearchLimits | None = None,
) -> GroundedAnswer:
    messages: list[AnyMessage] = [HumanMessage(question)]
    if prefetched is not None:
        # The routing step already searched the question verbatim. Present that as
        # the model's first search so it starts from evidence instead of spending
        # a request asking for it.
        args = SearchFilingsArgs(query=question, top_k=5)
        call = {"name": SEARCH_FILINGS, "args": args.model_dump(), "id": PREFETCH_CALL_ID}
        result = await search_filings(deps, args, prefetched=prefetched)
        messages += [AIMessage(content="", tool_calls=[call]), _tool_message(call, result)]

    final = await research_graph.ainvoke(
        {"messages": messages},
        context=ResearchContext(model=model, deps=deps, limits=limits or ResearchLimits()),
    )
    return final["answer"]
