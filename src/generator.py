# src/generator.py communicates with Ollama (via its REST API or the ollama Python library).
# src/generator.py


import ollama

class Generator:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know based on your notes."

Context:
{context}

Question: {query}

Answer:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        return response['response']