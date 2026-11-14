import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Provide the encode interface used by indexing and querying."""

    def __init__(self):
        self._model = TextEmbedding(model_name=MODEL_NAME, threads=1)

    def encode(self, texts):
        vectors = list(self._model.embed(texts, batch_size=8))
        return np.asarray(vectors)
