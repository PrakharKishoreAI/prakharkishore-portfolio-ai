from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.rag_service import (
    search,
    generate_answer,
    get_sources
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Prakhar Portfolio AI",
    description="RAG-based AI assistant for my portfolio",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",

        # Production Vercel frontend
        "https://prakharkishore-portfolio-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class ChatRequest(BaseModel):
    question: str


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Prakhar Portfolio AI API is running",
        "status": "success"
    }


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # 1. Search relevant information
    # --------------------------------------------------------

    results = search(request.question)

    # --------------------------------------------------------
    # 2. Generate answer using Groq
    # --------------------------------------------------------

    answer = generate_answer(
        request.question,
        results
    )

    # --------------------------------------------------------
    # 3. Get sources used for the answer
    # --------------------------------------------------------

    sources = get_sources(results)

    # --------------------------------------------------------
    # 4. Return response to React
    # --------------------------------------------------------

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }