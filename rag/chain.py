"""
chain.py
Chaîne RAG complète : retrieval FAISS + génération Mistral.
"""

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.retriever import retrieve

load_dotenv()

CUTOFF_DATE = "2025-12-31"

# --- Prompt ---
SYSTEM_PROMPT = """Tu es un assistant culturel spécialisé dans les événements \
de la région de Grenoble. Tu aides les utilisateurs à trouver des événements \
culturels (concerts, expositions, spectacles, festivals, etc.) en t'appuyant \
uniquement sur les informations fournies dans le contexte ci-dessous.

RÈGLE ABSOLUE : Tu ne dois mentionner QUE les événements explicitement présents \
dans le contexte fourni. N'invente jamais un titre, un lieu, une date ou une URL. \
Si le contexte ne contient pas de réponse pertinente, dis-le clairement.

Pour chaque événement cité, indique :
- Le titre exact
- Le lieu
- La date
- L'URL (si disponible dans le contexte)

Réponds en français, de manière chaleureuse et utile."""

HUMAN_PROMPT = """Contexte :
{context}

Question : {question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def format_docs(docs) -> str:
    """Formate les documents récupérés en texte pour le prompt."""
    formatted = []
    for doc in docs:
        m = doc.metadata
        block = (
            f"---\n"
            f"Titre : {m.get('title', 'N/A')}\n"
            f"Lieu : {m.get('place', 'N/A')} — {m.get('city', 'N/A')}\n"
            f"Date : {m.get('date_start', 'N/A')}\n"
            f"URL : {m.get('url', 'N/A')}\n"
            f"Description : {doc.page_content}\n"
        )
        formatted.append(block)
    return "\n".join(formatted)


def ask(question: str) -> dict:
    """
    Pose une question au système RAG.
    Retourne la réponse et les sources utilisées.
    """
    # 1. Retrieval — MMR + filtre de date, une seule fois
    docs = retrieve(question, k=6, max_date=CUTOFF_DATE)
    context_text = format_docs(docs)

    # 2. Génération — on passe le contexte directement, pas de retrieval interne
    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-small-latest",
        temperature=0.3,
    )

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})

    # 3. Sources extraites des mêmes docs que la réponse
    sources = [
        {
            "title": doc.metadata.get("title", ""),
            "city":  doc.metadata.get("city", ""),
            "date":  doc.metadata.get("date_start", ""),
            "url":   doc.metadata.get("url", ""),
        }
        for doc in docs
    ]

    return {
        "question": question,
        "answer":   answer,
        "context":  context_text,
        "sources":  sources,
    }