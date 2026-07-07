# Week 3	Store embeddings in ChromaDB, search for similar chunks
# Vector databases, similarity search
# Methods: add_documents(chunks, embeddings), search(query_embedding, top_k=3)
# This is a vector store module that can be imported into other scripts
# use embeddings from embedder.py to add documents and search for similar chunks
# src/vector_store.py


from typing import List, Dict
import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("notes")

    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        chunks: [{"text": "...", "metadata": {...}}, ...]
        embeddings: [[0.1, 0.2, ...], ...]
        """
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{m['source']}_{m['chunk_index']}" for m in metadatas]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results nicely
        output = []
        for i in range(len(results['documents'][0])):
            output.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return output