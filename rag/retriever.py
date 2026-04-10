"""
retriever.py
Chargement de l'index FAISS et recherche sémantique.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

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


def retrieve(query: str, k: int = 4) -> list:
    """
    Recherche les k documents les plus proches sémantiquement de la query.
    Retourne une liste de Documents LangChain avec métadonnées.
    """
    vectorstore = load_vectorstore()
    results = vectorstore.max_marginal_relevance_search(query, k=6, fetch_k=20)
    return results