# Week 2	Convert chunks into "embeddings" using a free local model
# What embeddings are, how to use Hugging Face models
import torch
from transformers import AutoTokenizer, AutoModel

# Import the FUNCTION from chunker.py
from chunker import split_file_into_chunks


def convert_chunks_to_embeddings(chunks):
    """Convert a list of text chunks into embeddings using a local model."""
    # Load the tokenizer and model (all-MiniLM-L6-v2 is an excellent, fast choice)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []
    print(f"\nProcessing {len(chunks)} chunks...")

    for chunk in chunks:
        # Tokenize the chunk
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)

        # Get the model output without calculating gradients (saves memory/time)
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling to compress token vectors into a single sentence embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)

    return embeddings


if __name__ == "__main__":
    # 1. Get the chunks by running the imported chunker function
    path = input("Enter the path to the text file: ")
    size = int(input("Enter the chunk size: "))

    my_chunks = split_file_into_chunks(path, size)

    # 2. Check if we actually got chunks back before embedding
    if my_chunks:
        vectors = convert_chunks_to_embeddings(my_chunks)
        print(f"Successfully generated {len(vectors)} embeddings!")
        print(f"Vector shape of first chunk: {vectors[0].shape}")
        # (Should output (384,) for MiniLM)