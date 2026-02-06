import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from utils.trace import trace


class TextEmbedder:
    """
    Thin wrapper around SentenceTransformers.
    This is intentionally simple.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    @trace
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)
