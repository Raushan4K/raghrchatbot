from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

# Request schema
class QueryRequest(BaseModel):
    question: str

# Response schema
class QueryResponse(BaseModel):
    answer: str
    context: str

# Initialize FastAPI and RAG pipeline once
app = FastAPI()
pipeline = RAGPipeline()

@app.get("/")
async def root():
    return {"message": "HR RAG chatbot API is running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question text is required")
    
    try:
        answer, context = pipeline.query(question)
        return QueryResponse(answer=answer, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
