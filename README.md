# RAG HR Policy Chatbot

This project is a Retrieval-Augmented Generation (RAG) HR chatbot using Streamlit (frontend) and FastAPI (backend). It allows users to ask questions about HR policy using answers extracted from a provided PDF, making retrieval more accurate and transparent.

---

## Features

- RAG pipeline: FAISS vector search, Cosine similarity re-ranking, LLM for answer generation
- End-to-end containerization for backend and frontend
- Answers cite original HR policy source snippets
- Streamlit user interface for querying
- Docker Images
 
---

## Getting Started

### **Local Development**

**Clone the  git repository:**
**Terminal :**
 - python -m venv hrragchatbot - create virtual environment 
 - Set-ExecutionPolicy -Scope Process
 - Type Bypass - for ExecutionPolicy
 - .\hrchatbot\Scripts\Activate.ps1  - Activate virtual environment
 - change the directory in backend folder and run : uvicorn api:app --host 0.0.0.0 --port 8000 reload 
 - now run the frontend code by changing new terminal and "cd frontend" and run : streamlit run app.py
 - pip install requirements.txt for frontend and backend for required python libraries
**security instructions**

 - create env file 
 - use GROQ LLM models api key

**Docker Hub instructions**
- created Dockerfile in both the directory backend and frontend
- created docker-compose.yml automatically links backend and frontend containers, maps ports, and sets environment variables.
- Simplifies local development/testing: `docker-compose up --build` brings the full stack up with   one command.

**How to run docker repo on localhost**
- pull the docker hub steps :
- docker pull raushan4k/rag-backend:v1
- docker run -p 8000:8000 -e GROQ_API_KEY=enter_groq_key  raushan4k/rag-backend:v1 
- docker pull raushan4k/rag-frontend:v2
- docker run -p 8501:8501 -e API_HOST=host.docker.internal raushan4k/rag-frontend:v2

## Usage Instructions

1. **Start both backend and frontend** (using Docker Compose or manually as above).
2. **Open the web UI** in your browser (default: http://localhost:8501).
3. **Ask questions** about HR policy, such as:
    - "What is the probation period?"
    - "How many earned leaves am I allowed?"
    - "Describe the dress code policy."
4. **Get cited answers** with source snippets from the HR PDF (as per the ingested data).


## Usage Instructions

1. **Start both backend and frontend** (using Docker Compose or manually as above).
2. **Open the web UI** in your browser (default: http://localhost:8501).
3. **Ask questions** about HR policy, such as:
    - "What is the probation period?"
    - "How many earned leaves am I allowed?"
    - "Describe the dress code policy."
4. **Get cited answers** with source snippets from the HR PDF (as per the ingested data).

## Environment Variables & Security Notes

- **Backend:**  
  - Set `GROQ_API_KEY` for LLM queries, ideally using an `.env` file.
- **Frontend:**  
  - Set `API_HOST` env variable to backend's address (e.g., `localhost:8000` )




