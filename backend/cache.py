from functools import lru_cache

@lru_cache(maxsize=256)
def cached_query(query):
    # Placeholder: connect this function to your RAG + LLM pipeline function to get answers
    # Example:
    # answer = call_rag_llm_pipeline(query)
    # return answer
    return f"Cached answer for query: {query}"
