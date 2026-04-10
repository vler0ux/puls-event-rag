"""
chain.py
Chaîne RAG complète : retrieval FAISS + génération Mistral.
"""

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import load_vectorstore

load_dotenv()

# --- Prompt système ---
SYSTEM_PROMPT = """Tu es un assistant culturel spécialisé dans les événements \
de la région de Grenoble. Tu aides les utilisateurs à trouver des événements \
culturels (concerts, expositions, spectacles, festivals, etc.) en t'appuyant \
uniquement sur les informations fournies dans le contexte ci-dessous.

Règles :
- Réponds toujours en français, de manière chaleureuse et enthousiaste.
- Cite le titre, le lieu, la date et l'URL de chaque événement recommandé.
- Si le contexte ne contient pas d'événement pertinent, dis-le honnêtement.
- Ne génère jamais d'événements fictifs.

Contexte :
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
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


def build_rag_chain():
    """Construit et retourne la chaîne RAG LangChain."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="open-mistral-7b",
       # model="mistral-small-latest",
        temperature=0.3,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> dict:
    """
    Pose une question au système RAG.
    Retourne la réponse et les sources utilisées.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

    # Récupère les sources pour les retourner avec la réponse
    docs = retriever.invoke(question)

    # Génère la réponse
    chain = build_rag_chain()
    answer = chain.invoke(question)

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
        "sources":  sources,
    }