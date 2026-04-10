"""
build_index.py
Vectorise les événements nettoyés et construit l'index FAISS.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INPUT_CSV  = Path("data/processed/events_clean.csv")
OUTPUT_DIR = Path("index/faiss_index")

def clean_field(value, default=""):
    """Nettoie un champ : NaN pandas → valeur par défaut."""
    if pd.isna(value) or str(value).strip().lower() == "nan":
        return default
    return str(value).strip()


def load_documents(csv_path: Path) -> list[Document]:
    """Charge le CSV et construit des Documents LangChain avec métadonnées."""
    df = pd.read_csv(csv_path)
    print(f"📄 {len(df)} événements chargés depuis {csv_path}")

    documents = []
    for _, row in df.iterrows():
        # Texte principal : titre + description (enrichi pour le RAG)
        content = (
            f"Titre : {clean_field(row.get('title'))}\n"
            f"Lieu : {clean_field(row.get('place'))} — {clean_field(row.get('city'))}\n"
            f"Date : {row.get('date_start', '')[:10] if pd.notna(row.get('date_start')) else 'Non précisée'}\n"
            f"Description : {clean_field(row.get('description'))}\n"
            f"Tags : {clean_field(row.get('tags'))}"
    )

        metadata = {
            "id":         clean_field(row.get("id")),
            "title":      clean_field(row.get("title")),
            "city":       clean_field(row.get("city")),
            "place":      clean_field(row.get("place")),
            "date_start": str(row.get("date_start", ""))[:10] if pd.notna(row.get("date_start")) else "",
            "date_end":   str(row.get("date_end", ""))[:10] if pd.notna(row.get("date_end")) else "",
            "url":        clean_field(row.get("url")),
            "tags":       clean_field(row.get("tags")),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Découpe les documents longs en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  {len(documents)} documents → {len(chunks)} chunks")
    return chunks


def build_faiss_index(chunks: list[Document]) -> FAISS:
    """Génère les embeddings et construit l'index FAISS."""
    print("🔢 Génération des embeddings via Mistral...")

    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed",
    )

    # Traitement par lots pour éviter les timeouts
    batch_size = 50
    all_chunks = chunks
    first_batch = all_chunks[:batch_size]

    print(f"  Batch 1/{(len(all_chunks) // batch_size) + 1} ({len(first_batch)} chunks)...")
    vectorstore = FAISS.from_documents(first_batch, embeddings)

    for i in range(batch_size, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(all_chunks) // batch_size) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

    print(f"✅ Index FAISS construit — {len(all_chunks)} vecteurs")
    return vectorstore


def save_index(vectorstore: FAISS, output_dir: Path):
    """Sauvegarde l'index FAISS sur disque."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    print(f"💾 Index sauvegardé → {output_dir}")


def test_search(vectorstore: FAISS):
    """Test rapide de recherche sémantique."""
    print("\n🔍 Test de recherche sémantique...")
    queries = [
        "concert de musique ce week-end",
        "exposition art contemporain",
        "spectacle pour enfants",
    ]
    for query in queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n  Query : '{query}'")
        for r in results:
            print(f"    → {r.metadata['title']} ({r.metadata['city']}, {r.metadata['date_start']})")


def main():
    print("=== Construction de l'index FAISS ===\n")

    # 1. Chargement
    documents = load_documents(INPUT_CSV)

    # 2. Chunking
    chunks = split_documents(documents)

    # 3. Embeddings + index FAISS
    vectorstore = build_faiss_index(chunks)

    # 4. Sauvegarde
    save_index(vectorstore, OUTPUT_DIR)

    # 5. Test rapide
    test_search(vectorstore)

    print("\n✅ Étape 3 terminée — index FAISS prêt !")


if __name__ == "__main__":
    main()