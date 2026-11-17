from typing import Protocol

from openai import AsyncOpenAI

from app.config import settings


class QueryEmbedder(Protocol):
    async def embed_query(self, query: str) -> list[float]:
        """Return an embedding vector for a user query."""


class OpenAIQueryEmbedder:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def embed_query(self, query: str) -> list[float]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be blank")

        response = await self._client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=[normalized],
            dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        vector = list(response.data[0].embedding)
        if len(vector) != settings.OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"query embedding dimension {len(vector)} did not match "
                f"{settings.OPENAI_EMBEDDING_DIMENSIONS}"
            )
        return vector
