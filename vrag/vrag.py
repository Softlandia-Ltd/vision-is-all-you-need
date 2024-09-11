import base64
import os
from typing import AsyncGenerator

import cv2
from vrag.colpali import ColPaliModel
from vrag.qdrant_client import InMemoryQdrant, Result
from vrag.pdf_to_image import images_from_pdf_bytes
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class VRAG:
    def __init__(self):
        self.colpali = ColPaliModel()
        self.qdrant = InMemoryQdrant()
        self.collection_name = "pdfs"
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def run_vrag(self, query: str) -> AsyncGenerator[str, None]:
        yield "Searching for relevant info...\n"
        results = await self.search(query)
        yield f"Found {len(results)} relevant pages...\n"
        yield "Answering..\n"
        augmented = await self.augment(results, query)
        async for completion in self.generate(augmented):
            yield completion

    # index data
    async def add_pdf(self, name: str, pdf: bytes):
        await self.qdrant.create_collection(self.collection_name)

        images = images_from_pdf_bytes(pdf)
        embeddings = self.colpali.embed_images(images)
        encoded_images = []
        for img in images:
            img = cv2.imencode(".jpg", img)[1]
            img = base64.b64encode(img).decode("utf-8")  # type: ignore
            encoded_images.append(img)

        await self.qdrant.upsert_points(
            self.collection_name, name, embeddings, encoded_images
        )

    # retrieve data
    async def search(self, query: str) -> list[Result]:
        query_vector = self.colpali.embed_queries([query])
        results = await self.qdrant.search_collection(
            self.collection_name, query_vector
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
    ) -> AsyncGenerator[str, None]:
        stream = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=query, stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
