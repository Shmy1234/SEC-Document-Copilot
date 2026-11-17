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
    # The latest reply's tool calls, carried explicitly: add_messages merges by
    # message id, so the history is not a reliable place to find them.
    pending_calls: list[dict]
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
    return {"messages": [reply], "pending_calls": list(reply.tool_calls), "requests": requests + 1}


async def run_tools(state: ResearchState, runtime: Runtime[ResearchContext]) -> ResearchState:
    context = runtime.context
    calls = state["pending_calls"]
    evidence_calls = [call for call in calls if call["name"] != SUBMIT_ANSWER]
    tool_calls = state.get("tool_calls", 0) + len(evidence_calls)
    if tool_calls > context.limits.tool_calls_limit:
        raise ResearchLimitExceeded("tool call limit reached")

    # Evidence tools run first and concurrently so a submit in the same step can
    # cite what they register. A TaskGroup cancels the siblings if one fails.
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_evidence_message(context.deps, c)) for c in evidence_calls]
    except ExceptionGroup as failure:
        # Surface the first tool's own error (e.g. RetrievalError) so the turn
        # reports a typed failure rather than an opaque group.
        raise failure.exceptions[0] from None
    update: ResearchState = {
        "messages": [task.result() for task in tasks],
        "tool_calls": tool_calls,
    }

    failures = state.get("grounding_failures", 0)
    for call in calls:
        if call["name"] != SUBMIT_ANSWER:
            continue
        if "answer" in update:
            update["messages"].append(_tool_message(call, "An answer was already accepted."))
            continue
        outcome = await _submit(context, failures, call)
        update["messages"].append(outcome.message)
        if outcome.answer is not None:
            update["answer"] = outcome.answer
        else:
            failures += 1
    update["grounding_failures"] = failures
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
    exhausted = failures >= context.limits.grounding_retries
    try:
        draft = AgentAnswer.model_validate(call["args"])
    except ValidationError as error:
        if exhausted:
            raise GroundingError("the model did not submit a well-formed answer") from error
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
            if exhausted:
                raise
            problem = str(error)
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
