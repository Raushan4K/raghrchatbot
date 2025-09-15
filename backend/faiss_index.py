import os
import faiss
import pickle
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(np.array(embeddings).astype('float32'))
    return index

if __name__ == "__main__":
    data_path = os.path.join("..", "data")
    emb_file = os.path.join(data_path, "embeddings.pkl")

    # Load embeddings and chunks
    with open(emb_file, "rb") as f:
        chunks, embeddings = pickle.load(f)

    print(f"Building FAISS index for {len(chunks)} chunks...")

    faiss_index = build_faiss_index(embeddings)

    # Save index file
    index_file = os.path.join(data_path, "hr_policy.index")
    faiss.write_index(faiss_index, index_file)
    print(f"FAISS index saved to {index_file}")
