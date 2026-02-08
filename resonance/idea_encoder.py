from typing import Dict
import numpy as np
from utils.trace import trace

from profiling.embedding.embedder import TextEmbedder


@trace
def encode_idea(
    idea_text: str,
    embedder: TextEmbedder,
) -> Dict:
    """
    Encode a user-submitted idea into embedding space.
    """

    embedding = embedder.embed_texts([idea_text])[0]

    return {
        "text": idea_text,
        "embedding": embedding,
        "dim": embedding.shape[0],
    }
