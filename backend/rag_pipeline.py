import os
import pickle
import faiss
import numpy as np
from rerank import rerank
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class RAGPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

        # Load chunks and embeddings
        emb_file = os.path.join(self.data_dir, "embeddings.pkl")
        with open(emb_file, "rb") as f:
            self.chunks, self.embeddings = pickle.load(f)

        # Load FAISS index
        index_file = os.path.join(self.data_dir, "hr_policy.index")
        self.index = faiss.read_index(index_file)

        # Load embedding model for queries
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Groq API client setup
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")
        self.client = Groq(api_key=api_key)
        self.llm_model = "llama-3.1-8b-instant"


    def retrieve(self, query, top_k=10, rerank_k=5):
        # Embed query
        q_emb = self.embed_model.encode([query])

        # Retrieve top_k from FAISS
        distances, indices = self.index.search(np.array(q_emb).astype('float32'), top_k)
        candidate_ids = indices[0].tolist()

        # Rerank top_k to get top rerank_k
        top_ids = rerank(q_emb[0], candidate_ids, self.embeddings, top_k=rerank_k)

        # Aggregate context from top reranked chunks
        context = "\n\n".join([self.chunks[idx] for idx in top_ids])

        return context

    def generate_answer(self, query, context):
        # Construct prompt for Groq LLM
        prompt = (
            "You are an HR policy assistant. Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, reply 'The answer is not available in the policy.'\n"
            f"Context:\n{context}\n\nQ: {query}\nA:"
        )

        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()

    def query(self, question):
        context = self.retrieve(question)
        answer = self.generate_answer(question, context)
        return answer, context

if __name__ == "__main__":
    pipeline = RAGPipeline()
    test_query = "What is the leave policy for new employees?"
    answer, ctx = pipeline.query(test_query)
    print("Answer:", answer)
    print("Context fetched:", ctx[:500], "...")
