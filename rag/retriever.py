"""
retriever.py
Chargement de l'index FAISS et recherche sémantique.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import date

load_dotenv()

INDEX_DIR = Path("index/faiss_index")


def load_vectorstore() -> FAISS:
    """Charge l'index FAISS depuis le disque."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_DIR}. "
            "Lance d'abord : uv run python scripts/build_index.py"
        )

    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed",
    )

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore



def retrieve(query: str, k: int = 6, max_date: str | None = None) -> list:
    """
    Recherche les k documents les plus proches sémantiquement de la query.
    max_date : filtre optionnel au format 'YYYY-MM-DD' (ex: '2025-12-31')
    """
    vectorstore = load_vectorstore()

    if max_date:
        # On récupère plus de candidats pour compenser le filtre
        candidates = vectorstore.similarity_search(query, k=k * 6)
        results = [
            doc for doc in candidates
            if doc.metadata.get("date_start", "") <= max_date
        ]
        return results[:k]
    
    return vectorstore.similarity_search(query, k=k)