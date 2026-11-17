"""The three LLM roles a chat turn uses, as LangChain runnables.

Each role is a runnable over a message list so the turn graph never depends on a
provider class. Production builds them from ChatOpenAI; tests pass
RunnableLambdas that return scripted outputs.
"""

from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from app.assistant.outputs import AgentAnswer
from app.assistant.tools import RESEARCH_TOOL_SPECS
from app.chat.routing import ModelRouteDecision
from app.config import Settings

Messages = list[BaseMessage]


@dataclass(frozen=True)
class ChatModels:
    route: Runnable[Messages, ModelRouteDecision]
    quick_answer: Runnable[Messages, AgentAnswer]
    research: Runnable[Messages, AIMessage]


def _chat_model(settings: Settings, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model.removeprefix("openai:"),
        api_key=SecretStr(settings.OPENAI_API_KEY),
        timeout=60,
        max_retries=2,
    )


def build_chat_models(settings: Settings) -> ChatModels:
    fast = _chat_model(settings, settings.OPENAI_FAST_MODEL)
    chat = _chat_model(settings, settings.OPENAI_CHAT_MODEL)
    return ChatModels(
        # One retry when the model returns a decision that fails validation,
        # matching the old agent's single output retry.
        route=fast.with_structured_output(ModelRouteDecision).with_retry(
            retry_if_exception_type=(ValidationError, OutputParserException),
            stop_after_attempt=2,
        ),
        quick_answer=fast.with_structured_output(AgentAnswer),
        # tool_choice="any" makes every step a tool call, so the run always ends
        # through submit_answer with a structured answer.
        research=chat.bind_tools(RESEARCH_TOOL_SPECS, tool_choice="any"),
    )
