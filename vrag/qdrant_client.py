import uuid
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from typing import Any


class Result(BaseModel):
    id: int | str
    score: float
    payload: dict[str, Any]


class InMemoryQdrant:
    def __init__(self) -> None:
        self.client = AsyncQdrantClient(":memory:")

    async def does_collection_exist(self, name: str) -> bool:
        return await self.client.collection_exists(name)

    async def create_collection(self, name: str):
        exists = await self.client.collection_exists(name)

        if exists:
            return

        return await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    async def search_collection(
        self, collection: str, query_vector: list[list[float]], count: int
    ) -> list[Result]:
        result = await self.client.query_points(
            collection, query=query_vector, limit=count
        )
        return [
            Result(id=point.id, score=point.score, payload=point.payload or {})
            for point in result.points
        ]

    async def upsert_points(
        self,
        collection: str,
        pdf_name: str,
        embeddings: list[list[list[float]]],
        encoded_images: list[str],
    ) -> models.UpdateResult:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page": idx + 1,
                    "name": pdf_name,
                    "image": encoded_images[idx],
                },
            )
            for idx, embedding in enumerate(embeddings)
        ]
        return await self.client.upsert(collection, points)
