# this is the Pipeline class that will be used to run the entire pipeline
# src/pipeline.py

'''
Interview question: "What is 'temperature' in LLM generation?"
Answer: Controls randomness. 0 = deterministic/greedy, good for factual tasks.
Higher = more creative. For RAG, keep it low (0.1–0.3) if you want strict factuality
from your notes.

'''


from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator


class RAGPipeline:
    def __init__(self, data_dir: str = "./data/notes"):
        self.data_dir = data_dir
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.generator = Generator()
        self._is_indexed = False

    def index(self):
        """Run once to load notes into the vector DB."""
        if self._is_indexed:
            return
        chunks = chunk_documents(self.data_dir)
        embeddings = self.embedder.embed([c["text"] for c in chunks])
        self.vector_store.add_documents(chunks, embeddings)
        self._is_indexed = True
        print(f"Indexed {len(chunks)} chunks.")

    def ask(self, question: str) -> Dict:
        self.index()
        context, sources = self.retriever.retrieve(question)
        answer = self.generator.generate(question, context)
        return {
            "answer": answer,
            "sources": sources
        }