import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def rerank(query_embedding, candidate_ids, embeddings, top_k=5):
    candidate_vecs = [embeddings[i] for i in candidate_ids]
    scores = cosine_similarity([query_embedding], candidate_vecs)[0]
    scored_candidates = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in scored_candidates[:top_k]]

if __name__ == "__main__":
    # Example usage - load embeddings and pick some candidate indexes
    data_path = os.path.join("..", "data")
    emb_file = os.path.join(data_path, "embeddings.pkl")
    
    with open(emb_file, "rb") as f:
        chunks, embeddings = pickle.load(f)
    
    # Dummy query embedding for testing
    query_embedding = embeddings[0] 
    
    candidate_ids = list(range(min(10, len(embeddings))))  # e.g. top 10 FAISS candidates

    top_ids = rerank(query_embedding, candidate_ids, embeddings)
    print("Top reranked chunk ids:", top_ids)
