"""
test_api.py
Tests unitaires pour l'API FastAPI Puls-Events.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)


def test_root():
    """GET / retourne un statut ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    """GET /health retourne healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_ui():
    """GET /chat retourne une page HTML."""
    response = client.get("/chat")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


@patch("api.main.ask")
def test_ask_valid(mock_ask):
    """POST /ask avec une question valide retourne une réponse RAG."""
    mock_ask.return_value = {
        "question": "Quels concerts à Grenoble ?",
        "answer": "Voici les concerts trouvés...",
        "sources": [{"title": "Concert de jazz", "date": "2025-06-20", "url": "http://example.com"}],
    }

    response = client.post("/ask", json={"question": "Quels concerts à Grenoble ?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


@patch("api.main.ask")
def test_ask_empty_question(mock_ask):
    """POST /ask avec une question vide retourne une erreur 400."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


@patch("api.main.ask")
def test_ask_rag_error(mock_ask):
    """POST /ask retourne 500 si la chaîne RAG plante."""
    mock_ask.side_effect = Exception("Erreur FAISS")
    response = client.post("/ask", json={"question": "Test ?"})
    assert response.status_code == 500