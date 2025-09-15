import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

if __name__ == "__main__":
    # Load chunks
    chunks_path = os.path.join("..", "data", "chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and popular for RAG

    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Save embeddings and chunks (for FAISS)
    with open(os.path.join("..", "data", "embeddings.pkl"), "wb") as f:
        pickle.dump((chunks, embeddings), f)
    print(f"Embeddings generated for {len(chunks)} chunks.")
