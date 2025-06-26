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
        "einops==0.8.0",
        "vidore_benchmark==4.0.1",
    )
    .pip_install("numpy==2.1.1")
    .pip_install("opencv_python_headless==4.10.0.84")
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
    scaledown_window=300,
    image=img,
)
class ColPaliModel:
    def __init__(self):
        from transformers import PreTrainedModel
        from colpali_engine.models import (
            ColPaliProcessor,
        )

        self.model_name = "vidore/colpali-v1.2"
        self.model: PreTrainedModel
        self.token = os.environ.get("HF_TOKEN")
        self.processor: ColPaliProcessor
        self.mock_image = self.create_mock_image()

    @modal.build()
    @modal.enter()
    def load_model(self):
        import torch
        from colpali_engine.models import ColPali
        from colpali_engine.models import (
            ColPaliProcessor,
        )

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[1] >= 8 * 1024**3:
            device = torch.device("cuda")
            torch_type = torch.bfloat16
        else:
            device = torch.device("cpu")
            torch_type = torch.float32

        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch_type,
            device_map=device,
            token=self.token,
        ).eval()

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
            batch_size=1,
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

                # Shape: [batch_size, 1030, 128]
                # Convert embeddings_doc to CPU, then iterate over the batch dimension
                embeddings_list: list[list[list[float]]] = embeddings_doc.to(
                    "cpu"
                ).tolist()

            # Yield each image's embeddings as a list of lists (1030 embeddings of 128 dimensions each)
            for embedding_per_image in embeddings_list:
                yield embedding_per_image

    @modal.method()
    def generate_heatmaps(self, images: list[str], query: str):
        import cv2
        import numpy as np
        import base64

        heatmaps: list = []

        for image in images:
            b = base64.b64decode(image)
            data = np.frombuffer(b, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_text, query_tokens_list = self.process_text_similarity(query)
            attention_json = self.generate_interpretability_json(
                output_text, query_tokens_list, rgb
            )
            heatmaps.append(attention_json)

        return {
            "heatmaps": heatmaps,
            "query_tokens": [token for token in query_tokens_list if token != ""],
        }

    @staticmethod
    def is_special_token(token: str) -> bool:
        # Check if the token meets any of the special conditions
        if len(token) < 2:
            return True
        if token.startswith("<"):
            return True
        if token.isdigit():
            return True
        if token.isspace():
            return True
        if token == "Question":
            return True
        return False

    def process_text_similarity(self, query):
        import torch

        input_text_processed = self.processor.process_queries([query]).to(
            self.model.device
        )

        with torch.no_grad():
            output_text = self.model.forward(
                **input_text_processed
            )  # (1, query_tokens, dim)

        query_tokens_list = self.processor.tokenizer.tokenize(  # type: ignore
            self.processor.decode(input_text_processed.input_ids[0])
        )

        # Mark special tokens as empty strings
        query_tokens_list = [
            "" if ColPaliModel.is_special_token(token) else token
            for token in query_tokens_list
        ]

        return output_text, query_tokens_list

    def generate_interpretability_json(
        self, output_text, query_tokens_list, image
    ) -> list:
        """
        Generates attention heatmap JSON for a query and image pair using the model.
        Modified version of the function gen_and_save_similarity_map_per_token from vidore_benchmark repo.
        See: https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/interpretability/gen_similarity_maps.py
        """
        import cv2
        from einops import rearrange
        from vidore_benchmark.interpretability.torch_utils import (
            normalize_similarity_map_per_query_token,
        )
        import torch
        import numpy.typing as npt

        patch_size = 14
        resolution = 448
        n_patch_per_dim = resolution // patch_size
        input_image_square = cv2.resize(image, (resolution, resolution))
        input_image_processed = self.process_images([input_image_square]).to(
            self.model.device
        )

        with torch.no_grad():
            output_image = self.model.forward(
                **input_image_processed
            )  # (1, n_patches_x * n_patches_y, dim)

        # Remove the special tokens from the output
        output_image = output_image[
            :, : self.processor.image_seq_length, :
        ]  # (1, n_patches_x * n_patches_y, dim)

        # Rearrange the output image tensor to explicitly represent the 2D grid of patches
        output_image = rearrange(
            output_image, "b (h w) c -> b h w c", h=n_patch_per_dim, w=n_patch_per_dim
        )  # (1, n_patches_x, n_patches_y, dim)

        # Get the similarity map
        similarity_map = torch.einsum(
            "bnk,bijk->bnij", output_text, output_image
        )  # (1, query_tokens, n_patches_x, n_patches_y)

        # Normalize the similarity map
        similarity_map_normalized = normalize_similarity_map_per_query_token(
            similarity_map
        )  # (1, query_tokens, n_patches_x, n_patches_y)

        # Prepare the attention map data as a JSON structure
        attention_data = []
        sim_map = cast(
            npt.NDArray, similarity_map_normalized.to(torch.float32).cpu().numpy()
        )

        # Iterate over the tokens and collect similarity map data for each token
        for token_idx, token in enumerate(query_tokens_list):
            # skip special tokens
            if token == "":
                continue
            token_data = {
                "token": query_tokens_list[token_idx],
                "attention_map": sim_map[0, token_idx, :, :].tolist(),
            }
            attention_data.append(token_data)

        return attention_data
