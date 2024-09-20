import base64
import json
import os
from typing import AsyncGenerator

import cv2
from vrag.colpali import ColPaliModel
from vrag.qdrant_client import InMemoryQdrant, Result
from vrag.pdf_to_image import images_from_pdf_bytes
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from sse_starlette.sse import ServerSentEvent


class VRAG:
    def __init__(self, colpali: ColPaliModel, qdrant: InMemoryQdrant):
        self.colpali = colpali
        self.qdrant = qdrant
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def run_vrag(
        self, collection_name: str, query: str
    ) -> AsyncGenerator[ServerSentEvent, None]:
        results = await self.search(collection_name, query)
        yield ServerSentEvent(
            data=json.dumps(
                {
                    "results": [
                        {
                            "score": result.score,
                            "image": result.payload["image"],
                            "page": result.payload["page"],
                            "name": result.payload["name"],
                        }
                        for result in results
                    ]
                }
            ),
            event="sources",
        )
        augmented = await self.augment(results, query)
        async for completion in self.generate(augmented):
            yield completion

    # index data
    async def add_pdf(
        self, collection_name: str, name: str, pdf: bytes
    ) -> AsyncGenerator[ServerSentEvent, None]:
        await self.qdrant.create_collection(collection_name)

        embeddings: list[list[list[float]]] = []
        idx = 1
        batch_size = 2
        images = images_from_pdf_bytes(pdf)

        yield ServerSentEvent(
            data=json.dumps({"message": f"Indexing page {idx} / {len(images)}"})
        )
        async for embedding in self.colpali.embed_images.remote_gen.aio(
            images, batch_size
        ):
            embeddings.append(embedding)
            idx += batch_size
            if idx <= len(images):
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message": f"Indexing page {idx + 1} / {len(images)}\n"}
                    )
                )

        encoded_images = []

        for img in images:
            img = cv2.imencode(".jpg", img)[1]
            img = base64.b64encode(img).decode("utf-8")  # type: ignore
            encoded_images.append(img)

        await self.qdrant.upsert_points(
            collection_name, name, embeddings, encoded_images
        )

    # retrieve data
    async def search(self, collection_name: str, query: str) -> list[Result]:
        query_vector = await self.colpali.embed_queries.remote.aio([query])
        results = await self.qdrant.search_collection(collection_name, query_vector)
        return results

    # augment
    async def augment(
        self,
        results: list[Result],
        query: str,
    ) -> list[ChatCompletionMessageParam]:

        images = []
        for result in results:
            if result and result.payload and "image" in result.payload:
                image = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{result.payload['image']}"
                    },
                }
                images.append(image)

        payload = [
            {
                "role": "system",
                "content": "Your task is to answer to the user question based on the provided image or images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ]
                + images,
            },
        ]

        return payload

    # generate
    async def generate(
        self, query: list[ChatCompletionMessageParam]
    ) -> AsyncGenerator[ServerSentEvent, None]:
        stream = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=query, stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield ServerSentEvent(
                    data=json.dumps({"chunk": chunk.choices[0].delta.content})
                )
