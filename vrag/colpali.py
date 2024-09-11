import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from transformers import AutoProcessor


class NumpyDataset(Dataset):
    def __init__(self, numpy_list: list[np.ndarray]):
        self.numpy_list = numpy_list

    def __len__(self) -> int:
        return len(self.numpy_list)

    def __getitem__(self, idx) -> torch.Tensor:
        sample = self.numpy_list[idx]
        sample = torch.from_numpy(sample)
        return sample


class StringListDataset(Dataset):
    def __init__(self, string_list: list[str]):
        self.string_list = string_list

    def __len__(self) -> int:
        return len(self.string_list)

    def __getitem__(self, idx) -> str:
        sample = self.string_list[idx]
        return sample


class ColPaliModel:
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        hf_model: str = "google/paligemma-3b-mix-448",
    ):
        self.model_name = model_name
        self.hf_model = hf_model
        self.token = os.environ.get("HF_TOKEN")
        self.model = self.load_model()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, token=self.token
        )
        self.mock_image = self.create_mock_image()

    def load_model(self):
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[1] >= 8 * 1024**3:
            device = torch.device("cuda")
            torch_type = torch.bfloat16
        else:
            device = torch.device("cpu")
            torch_type = torch.float32

        model = ColPali.from_pretrained(
            self.hf_model,
            torch_dtype=torch_type,
            device_map=device,
            token=self.token,
        )

        model.load_adapter(self.model_name)
        return model.eval()

    def create_mock_image(self):
        """Creates a blank 448x448 RGB image."""
        return 255 * np.ones((448, 448, 3), dtype=np.uint8)

    def process_images(self, images: list[np.ndarray], max_length: int = 50):
        """This function is modified version of the function
        in colpali_processing_utils.py to remove PIL dependency.
        """
        texts_doc = ["Describe the image."] * len(images)
        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )
        return batch_doc

    def process_queries(
        self,
        queries: list[str],
        max_length: int = 50,
    ):
        """This function is modified version of the function
        in colpali_processing_utils.py to remove PIL dependency.
        """
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        batch_query = self.processor(
            images=[self.mock_image] * len(texts_query),
            # NOTE: the image is not used in batch_query but it is required for calling the processor
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

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embeds a text query and returns the vector representation for use in a vector database."""

        dataloader = DataLoader(
            StringListDataset(queries),
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

        embeddings = [tensor.tolist() for tensor in query_embeddings]
        return embeddings[0]

    def embed_images(self, images: list[np.ndarray]) -> list[list[list[float]]]:
        """Run inference on images."""
        embedding_list: list[torch.Tensor] = []

        dataloader = DataLoader(
            NumpyDataset(images),
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: self.process_images(x),
        )

        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            embedding_list.extend(torch.unbind(embeddings_doc.to("cpu")))

        embeddings = [tensor.tolist() for tensor in embedding_list]
        return embeddings
