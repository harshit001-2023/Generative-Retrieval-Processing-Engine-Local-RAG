# Week 3	Store embeddings in ChromaDB, search for similar chunks
# Vector databases, similarity search
# Methods: add_documents(chunks, embeddings), search(query_embedding, top_k=3)
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("embeddings")

def add_documents(chunks, embeddings):
    """
    Add documents to the ChromaDB collection.

    Args:
        chunks (list): List of text chunks to be added.
        embeddings (list): List of corresponding embeddings for the chunks.
    """
    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            documents=[chunk],
            embeddings=[embedding]
        )

def search(query_embedding, top_k=3):
    """
    Search for similar chunks in the ChromaDB collection.

    Args:
        query_embedding (list): The embedding of the query.
        top_k (int): Number of top similar chunks to retrieve.

    Returns:
        list: List of tuples containing the chunk and its similarity score.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Extracting the chunks and their similarity scores
    similar_chunks = []
    for i in range(len(results['documents'][0])):
        chunk = results['documents'][0][i]
        score = results['distances'][0][i]
        similar_chunks.append((chunk, score))
    
    return similar_chunks




