# Creating a Python script that reads text files and splits them into chunks
# File I/O, string manipulation
# This is a Chunker module that can be imported into other scripts
# src/chunker.py — The "Lego brick" version
import os
from typing import List, Dict


def chunk_documents(data_dir: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Reads all .txt/.md files in data_dir and returns chunks with metadata.
    Returns: [{"text": "...", "metadata": {"source": "notes1.txt", "chunk_index": 0}}, ...]
    """
    chunks = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.txt', '.md')):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Better: chunk by paragraphs or sentences, not just characters
            # But character-based is fine for learning
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 0:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": filename,
                            "chunk_index": i // (chunk_size - overlap)
                        }
                    })
    return chunks