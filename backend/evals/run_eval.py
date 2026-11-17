"""Evaluate the chat turn graph against live services, optionally in LangSmith.

    uv run python -m evals.run_eval                 # local report only
    uv run python -m evals.run_eval --langsmith     # upload dataset + experiment
    uv run python -m evals.run_eval --judge         # add an LLM faithfulness judge

Runs the real LangGraph turn (routing, hybrid retrieval, research loop,
grounding) without persisting chats, then scores each answer.
"""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.chat.graph import TurnContext, turn_graph
from app.chat.stages import error_code_for
from app.config import settings
from app.resources import build_resources
from app.retrieval.embeddings import OpenAIQueryEmbedder
from app.retrieval.retriever import DocumentRetriever
from evals.cases import CASES
from evals.scoring import EVALUATORS, summarize

DATASET_NAME = "document-copilot-turns"


async def answer(inputs: dict, context: TurnContext) -> dict:
    started = perf_counter()
    route, stages, final = None, [], None
    try:
        async for part in turn_graph.astream(
            {"question": inputs["question"]},
            context=context,
            stream_mode=["custom", "updates"],
            version="v2",
        ):
            if part["type"] == "custom":
                stages += [part["data"]["stage"]] if "stage" in part["data"] else []
                route = part["data"].get("route", route)
            else:
                for update in part["data"].values():
                    if update and "answer" in update:
                        final = update["answer"]
    except Exception as error:  # noqa: BLE001 - an eval records failures, it does not stop
        return {"route": route, "error": error_code_for(error), "latency_ms": _ms(started)}
    return {
        "route": route,
        "answer": final.answer,
        "citations": [c.model_dump(mode="json") for c in final.citations],
        "stages": stages,
        "latency_ms": _ms(started),
    }


def _ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


class Faithfulness(BaseModel):
    score: float = Field(ge=0, le=1, description="1 if every claim is supported by the quotes.")
    reason: str


def judge_evaluator():
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.OPENAI_FAST_MODEL.removeprefix("openai:"),
        api_key=settings.OPENAI_API_KEY,
    ).with_structured_output(Faithfulness)

    async def faithfulness(run, example) -> dict:
        outputs = run.outputs if hasattr(run, "outputs") else run
        quotes = [c["quoted_text"] for c in outputs.get("citations", [])]
        if not quotes:
            return {"key": "faithfulness", "score": None, "comment": "no citations"}
        verdict = await model.ainvoke(
            [
                SystemMessage(
                    "Grade whether every factual claim in ANSWER is supported by QUOTES. "
                    "Unsupported numbers or entities score 0."
                ),
                HumanMessage(json.dumps({"answer": outputs.get("answer"), "quotes": quotes})),
            ]
        )
        return {"key": "faithfulness", "score": verdict.score, "comment": verdict.reason}

    return faithfulness


async def run_local(context: TurnContext, judge) -> list[dict]:
    rows = []
    for case in CASES:
        outputs = await answer(case["inputs"], context)
        scores = [evaluator(outputs, case["outputs"]) for evaluator in EVALUATORS]
        if judge is not None:
            scores.append(await judge(outputs, case))
        rows.append({"question": case["inputs"]["question"], "outputs": outputs, "scores": scores})
        print(
            f"{case['inputs']['question'][:60]:<60} {outputs.get('route')} {outputs['latency_ms']}ms"
        )
    return rows


async def run_langsmith(context: TurnContext, judge) -> None:
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    client = Client(api_key=settings.LANGSMITH_API_KEY, api_url=settings.LANGSMITH_ENDPOINT)
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(DATASET_NAME, description="Document Copilot turn evals")
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[case["inputs"] for case in CASES],
            outputs=[case["outputs"] for case in CASES],
        )

    def wrap(evaluator):
        def run_evaluator(run, example):
            return evaluator(run.outputs, example.outputs)

        run_evaluator.__name__ = evaluator.__name__
        return run_evaluator

    evaluators = [wrap(e) for e in EVALUATORS] + ([judge] if judge else [])
    await aevaluate(
        lambda inputs: answer(inputs, context),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="turn-graph",
        metadata={
            "chat_model": settings.OPENAI_CHAT_MODEL,
            "fast_model": settings.OPENAI_FAST_MODEL,
        },
        max_concurrency=2,
        client=client,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langsmith", action="store_true", help="Upload to LangSmith.")
    parser.add_argument("--judge", action="store_true", help="Add an LLM faithfulness judge.")
    parser.add_argument("--output", type=Path, help="Write the local report as JSON.")
    args = parser.parse_args()

    resources = build_resources(settings)
    try:
        context = TurnContext(
            models=resources.models,
            retriever=DocumentRetriever(
                session_factory=resources.session_factory,
                embedder=OpenAIQueryEmbedder(resources.openai),
            ),
        )
        judge = judge_evaluator() if args.judge else None
        if args.langsmith:
            await run_langsmith(context, judge)
            return
        rows = await run_local(context, judge)
        report = {"summary": summarize(rows), "rows": rows}
        print(json.dumps(report["summary"], indent=2))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2))
    finally:
        await resources.aclose()


if __name__ == "__main__":
    asyncio.run(main())
