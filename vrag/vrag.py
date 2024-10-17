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
        self, collection_name: str, query: str, count: int = 3
    ) -> AsyncGenerator[ServerSentEvent, None]:
        results = await self.search(collection_name, query, count)

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

        imgs: list[str] = [result.payload["image"] for result in results]
        heatmaps = self.colpali.generate_heatmaps.remote(imgs, query)
        yield ServerSentEvent(
            data=json.dumps(heatmaps),
            event="heatmaps",
        )

    # index data
    async def add_pdf(
        self, collection_name: str, name: str, pdf: bytes
    ) -> AsyncGenerator[ServerSentEvent, None]:
        await self.qdrant.create_collection(collection_name)

        embeddings: list[list[list[float]]] = []
        idx = 0
        batch_size = 4
        images = images_from_pdf_bytes(pdf)
        count = len(images)

        yield ServerSentEvent(
            data=json.dumps({"message": f"0 % of {count} pages indexed...\n"})
        )

        async for embedding in self.colpali.embed_images.remote_gen.aio(
            images, batch_size
        ):
            embeddings.append(embedding)
            if idx < count:
                percent = int(idx / count * 100)
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message": f"{percent} % of {count} pages indexed...\n"}
                    )
                )
            idx += 1

        encoded_images = []

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.imencode(".jpg", img)[1]
            img = base64.b64encode(img).decode("utf-8")  # type: ignore
            encoded_images.append(img)

        await self.qdrant.upsert_points(
            collection_name, name, embeddings, encoded_images
        )

    # retrieve data
    async def search(
        self, collection_name: str, query: str, count: int
    ) -> list[Result]:
        query_vector = await self.colpali.embed_queries.remote.aio([query])
        results = await self.qdrant.search_collection(
            collection_name, query_vector, count
        )
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
            model="gpt-4o", messages=query, stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield ServerSentEvent(
                    data=json.dumps({"chunk": chunk.choices[0].delta.content})
                )
