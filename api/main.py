"""
main.py
API REST FastAPI exposant le système RAG Puls-Events.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.chain import ask
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
    """Vérifie que l'index FAISS est disponible au démarrage."""
    print("🚀 Vérification de l'index FAISS...")
    from rag.retriever import load_vectorstore
    load_vectorstore()  # lève une erreur claire si l'index est absent
    print("✅ Index FAISS prêt.")
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
        return RebuildResponse(
            status="ok",
            message=f"Index reconstruit avec {len(chunks)} chunks."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur rebuild : {str(e)}")

@app.get("/chat", response_class=HTMLResponse, tags=["UI"])
def chat_ui():
    """Interface de démo du chatbot."""
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Puls-Events Chatbot</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        #chat { background: white; border-radius: 8px; padding: 20px; min-height: 300px; margin-bottom: 16px; overflow-y: auto; max-height: 500px; }
        .msg { margin: 10px 0; }
        .user { text-align: right; }
        .user span { background: #4a90d9; color: white; padding: 8px 14px; border-radius: 16px; display: inline-block; }
        .bot span { background: #e0e0e0; padding: 8px 14px; border-radius: 16px; display: inline-block; white-space: pre-wrap; }
        #input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 16px; }
        button { padding: 10px 20px; background: #4a90d9; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
        button:disabled { opacity: 0.5; }
    </style>
</head>
<body>
    <h1>🎭 Puls-Events Chatbot</h1>
    <p>Pose une question sur les événements culturels de Grenoble !</p>
    <div id="chat"></div>
    <div id="input-area">
        <input id="question" type="text" placeholder="Ex: Quels concerts ce week-end ?" autofocus />
        <button id="send" onclick="sendMessage()">Envoyer</button>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById("question");
            const q = input.value.trim();
            if (!q) return;
            const chat = document.getElementById("chat");
            const btn = document.getElementById("send");

            chat.innerHTML += `<div class="msg user"><span>${q}</span></div>`;
            input.value = "";
            btn.disabled = true;
            chat.innerHTML += `<div class="msg bot" id="thinking"><span>⏳ Recherche en cours...</span></div>`;
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: q })
                });
                const data = await res.json();
                document.getElementById("thinking").remove();
                chat.innerHTML += `<div class="msg bot"><span>${data.answer}</span></div>`;
            } catch (e) {
                document.getElementById("thinking").remove();
                chat.innerHTML += `<div class="msg bot"><span>❌ Erreur : ${e.message}</span></div>`;
            }
            btn.disabled = false;
            chat.scrollTop = chat.scrollHeight;
        }

        document.getElementById("question").addEventListener("keydown", e => {
            if (e.key === "Enter") sendMessage();
        });
    </script>
</body>
</html>
"""