"""Tests unitaires pour l'indexation FAISS."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def test_index_exists():
    """L'index FAISS doit exister après build_index.py."""
    assert Path("index/faiss_index").exists(), \
        "Le dossier index/faiss_index est introuvable — lance build_index.py d'abord."


def test_index_files():
    """Les fichiers FAISS obligatoires doivent être présents."""
    index_dir = Path("index/faiss_index")
    assert (index_dir / "index.faiss").exists(), "Fichier index.faiss manquant."
    assert (index_dir / "index.pkl").exists(), "Fichier index.pkl manquant."


def test_index_search():
    """Une recherche sémantique doit retourner des résultats pertinents."""
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed",
    )
    vectorstore = FAISS.load_local(
        "index/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    results = vectorstore.similarity_search("concert musique Grenoble", k=3)
    assert len(results) > 0, "Aucun résultat retourné par la recherche."
    assert all(hasattr(r, "page_content") for r in results), \
        "Les résultats ne contiennent pas de page_content."


def test_metadata_present():
    """Les métadonnées (titre, ville, date) doivent être présentes dans les résultats."""
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed",
    )
    vectorstore = FAISS.load_local(
        "index/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    results = vectorstore.similarity_search("exposition peinture", k=1)
    meta = results[0].metadata
    assert "title" in meta, "Métadonnée 'title' manquante."
    assert "city" in meta, "Métadonnée 'city' manquante."
    assert "date_start" in meta, "Métadonnée 'date_start' manquante."