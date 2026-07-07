# Week 2	Convert chunks into "embeddings" using a free local model
# What embeddings are, how to use Hugging Face models
# This is a local embedding module that can be imported into other scripts
# src/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Returns list of 384-dimensional vectors for MiniLM
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()