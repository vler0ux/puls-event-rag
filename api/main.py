"""
main.py
API REST FastAPI exposant le système RAG Puls-Events.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.chain import ask, build_rag_chain
from scripts.build_index import load_documents, split_documents, build_faiss_index, save_index
from pathlib import Path

load_dotenv()

# --- Modèles de données ---
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]

class RebuildResponse(BaseModel):
    status: str
    message: str


# --- Chargement au démarrage ---
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge la chaîne RAG au démarrage de l'API."""
    global rag_chain
    print("🚀 Chargement de la chaîne RAG...")
    rag_chain = build_rag_chain()
    print("✅ Chaîne RAG prête.")
    yield
    print("🛑 Arrêt de l'API.")


# --- Application ---
app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API de recommandations d'événements culturels pour Grenoble / Isère. "
        "Basée sur un système RAG (FAISS + Mistral AI)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/", tags=["Santé"])
def root():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "message": "Puls-Events RAG API is running 🎭"}


@app.get("/health", tags=["Santé"])
def health():
    """Endpoint de santé pour Docker / monitoring."""
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    """
    Pose une question sur les événements culturels de Grenoble.
    Retourne une réponse générée par le LLM augmentée des données indexées.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        result = ask(request.question)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur RAG : {str(e)}")


@app.post("/rebuild", response_model=RebuildResponse, tags=["Administration"])
def rebuild_index():
    """
    Reconstruit l'index vectoriel FAISS à partir des données collectées.
    À utiliser après une mise à jour des données OpenAgenda.
    """
    global rag_chain
    try:
        csv_path = Path("data/processed/events_clean.csv")
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Données introuvables. Lance d'abord collect_data.py."
            )

        docs   = load_documents(csv_path)
        chunks = split_documents(docs)
        vs     = build_faiss_index(chunks)
        save_index(vs, Path("index/faiss_index"))

        # Recharge la chaîne avec le nouvel index
        rag_chain = build_rag_chain()

        return RebuildResponse(
            status="ok",
            message=f"Index reconstruit avec {len(chunks)} chunks."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur rebuild : {str(e)}")