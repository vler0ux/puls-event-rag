"""
evaluate_rag.py
Évaluation automatique du système RAG avec Ragas.
"""

import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from rag.retriever import load_vectorstore

load_dotenv()

QA_DATASET = Path("docs/qa_dataset.json")
OUTPUT     = Path("docs/evaluation_results.json")


def build_ragas_dataset() -> Dataset:
    """Construit le dataset Ragas à partir du jeu de test annoté."""
    with open(QA_DATASET, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    from rag.chain import build_rag_chain
    chain = build_rag_chain()

    questions  = []
    answers    = []
    contexts   = []
    references = []

    for i, item in enumerate(qa_pairs):
        q = item["question"]
        print(f"  [{i+1}/{len(qa_pairs)}] {q}")

        # Réponse générée
        answer = chain.invoke(q)

        # Contextes récupérés
        docs = retriever.invoke(q)
        ctx  = [doc.page_content for doc in docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        references.append(item["reference"])

    return Dataset.from_dict({
        "question":  questions,
        "answer":    answers,
        "contexts":  contexts,
        "reference": references,
    })


def main():
    print("=== Évaluation RAG avec Ragas ===\n")

    print("📊 Génération des réponses sur le jeu de test...")
    dataset = build_ragas_dataset()

    print("\n🔍 Calcul des métriques Ragas...")

    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-small-latest",
        temperature=0,
    )
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed",
    )

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== Résultats ===")
    scores = results.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision"]
    ].mean()

    for metric, score in scores.items():
        print(f"  {metric:25s} : {score:.3f}")

    # Sauvegarde
    output_data = {
        "scores": scores.to_dict(),
        "details": results.to_pandas().to_dict(orient="records"),
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Résultats sauvegardés → {OUTPUT}")


if __name__ == "__main__":
    main()