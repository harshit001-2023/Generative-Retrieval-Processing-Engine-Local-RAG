# Week 2	Convert chunks into "embeddings" using a free local model
# What embeddings are, how to use Hugging Face models

def convert_chunks_to_embeddings(chunks):
    """
    Convert a list of text chunks into embeddings using a local model.

    Args:
        chunks (list of str): The text chunks to convert.

    Returns:
        list of np.ndarray: The embeddings for each chunk.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []
    for chunk in chunks:
        # Tokenize the chunk and convert to tensor
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        
        # Get the model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling to get a single vector for the chunk
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return embeddings



    