import os
from typing import AsyncGenerator, cast
import modal
from vrag.app import app


img = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "colpali_engine==0.3.0",
        "torch",
        "transformers==4.44.2",
    )
    .pip_install("numpy==2.1.1")
)


class NumpyDataset:
    def __init__(self, numpy_list: list):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        self.numpy_list = numpy_list

    def __len__(self) -> int:
        return len(self.numpy_list)

    def __getitem__(self, idx):
        sample = self.numpy_list[idx]
        sample = self.torch.from_numpy(sample)
        return sample


class StringListDataset:
    def __init__(self, string_list: list[str]):
        self.string_list = string_list

    def __len__(self) -> int:
        return len(self.string_list)

    def __getitem__(self, idx) -> str:
        sample = self.string_list[idx]
        return sample


def create_numpy_dataset_class(numpy_list: list):
    from torch.utils.data import Dataset

    class DynamicNumpyDataset(NumpyDataset, Dataset):
        pass

    return DynamicNumpyDataset(numpy_list)


def create_stringlist_dataset_class(string_list: list[str]):
    from torch.utils.data import Dataset

    class DynamicStringListDataset(StringListDataset, Dataset):
        pass

    return DynamicStringListDataset(string_list)


@app.cls(
    gpu="A10G",
    secrets=[modal.Secret.from_dotenv()],
    cpu=4,
    timeout=600,
    container_idle_timeout=300,
    image=img,
)
class ColPaliModel:
    def __init__(self):
        from transformers import PreTrainedModel
        from colpali_engine.models.paligemma.colpali.processing_colpali import (
            ColPaliProcessor,
        )

        self.model_name = "vidore/colpali-v1.2"
        self.base_model = "vidore/colpaligemma-3b-pt-448-base"
        self.model: PreTrainedModel
        self.token = os.environ.get("HF_TOKEN")
        self.processor: ColPaliProcessor
        self.mock_image = self.create_mock_image()

    @modal.build()
    @modal.enter()
    def load_model(self):
        import torch
        from colpali_engine.models import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import (
            ColPaliProcessor,
        )

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[1] >= 8 * 1024**3:
            device = torch.device("cuda")
            torch_type = torch.bfloat16
        else:
            device = torch.device("cpu")
            torch_type = torch.float32

        self.model = ColPali.from_pretrained(
            self.base_model,
            torch_dtype=torch_type,
            device_map=device,
            token=self.token,
        ).eval()

        self.model.load_adapter(self.model_name)

        self.processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(self.model_name),
        )

    def create_mock_image(self):
        import numpy as np

        """Creates a blank 448x448 RGB image."""
        return 255 * np.ones((448, 448, 3), dtype=np.uint8)

    def process_images(self, images: list):
        """This function is modified version of the function in colpali_processing_utils.py to remove PIL dependency."""
        texts_doc = ["Describe the image."] * len(images)
        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_queries(
        self,
        queries: list[str],
        max_length: int = 50,
        suffix: str | None = None,
    ):
        """This function is modified version of the function in colpali_processing_utils.py to remove PIL dependency."""

        if suffix is None:
            suffix = "<pad>" * 10
        texts_query: list[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self.processor(
            images=[self.mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )

        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][
            ..., self.processor.image_seq_length :
        ]
        batch_query["attention_mask"] = batch_query["attention_mask"][
            ..., self.processor.image_seq_length :
        ]

        return batch_query

    @modal.method()
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        import torch

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            create_stringlist_dataset_class(queries),
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: self.process_queries(x),
        )

        query_embeddings: list[torch.Tensor] = []

        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {
                    k: v.to(self.model.device) for k, v in batch_query.items()
                }
                embedding_query = self.model(**batch_query)
            query_embeddings.extend(torch.unbind(embedding_query.to("cpu")))

        embeddings = query_embeddings[0].tolist()
        return embeddings

    @modal.method()
    async def embed_images(
        self, images: list, batch_size: int
    ) -> AsyncGenerator[list[list[float]], None]:
        import torch
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            create_numpy_dataset_class(images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.process_images(x),
        )

        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)

            value: list[list[float]] = torch.unbind(embeddings_doc.to("cpu"))[
                0
            ].tolist()
            yield value
