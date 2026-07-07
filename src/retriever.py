# src/retriever.py
from src.vector_store import VectorStore
from src.embedder import Embedder


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vs = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> str:
        query_embedding = self.embedder.embed_query(query)
        results = self.vs.search(query_embedding, top_k)

        # Format chunks into a single context string
        context = "\n\n".join([
            f"[Source: {r['metadata']['source']}]\n{r['text']}"
            for r in results
        ])
        return context, results