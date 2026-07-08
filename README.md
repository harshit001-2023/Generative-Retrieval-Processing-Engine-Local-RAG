# Generative Retrieval Processing Engine (Local RAG)

Chat with your personal notes using local AI. No API keys. No data leaves your machine.

## What It Does
- Ingests `.txt` and `.md` files from a folder
- Splits them into chunks, embeds them, and stores them in ChromaDB
- Answers questions using a local LLM (Ollama) based ONLY on your notes

## Tech Stack
- Python, Sentence-Transformers, ChromaDB, Ollama, Streamlit

## Setup

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull llama3.2`
3. Install Python deps: `pip install -r requirements.txt`
4. Add notes to `data/notes/`
5. Run: `streamlit run src/app.py`

## Screenshot
[Add screenshot here]

## What I Learned
- Vector embeddings and similarity search
- How RAG pipelines work end-to-end
- Prompt engineering for grounded generation
- Building local AI apps without cloud APIs