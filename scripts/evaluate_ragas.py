"""
evaluate_rag.py
Évaluation automatique du système RAG avec Ragas.
"""

import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx

@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5)
)
def invoke_with_retry(chain, question):
    return chain.invoke(question)

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from rag.retriever import load_vectorstore
from ragas.run_config import RunConfig
from rag.chain import ask

load_dotenv()

QA_DATASET = Path("docs/qa_dataset.json")
OUTPUT     = Path("docs/evaluation_resultsRagas.json")


def load_dataset(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nan_to_none(v) -> float | None:
    """Convertit NaN en None pour une sérialisation JSON propre."""
    try:
        return None if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


def main():
    print("=== Évaluation Ragas (faithfulness + context_precision) ===\n")

    dataset_raw = load_dataset(QA_DATASET)

    # Ragas v0.2+ utilise les noms de champs : user_input / response /
    # retrieved_contexts / reference  (et non question/answer/contexts/ground_truth)
    data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    print("📊 Génération des réponses sur le jeu de test...")
    for i, item in enumerate(dataset_raw, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset_raw)}] {question}")
        result = ask(question)
        data["user_input"].append(result["question"])
        data["response"].append(result["answer"])
        data["retrieved_contexts"].append([result["context"]])
        data["reference"].append(item["reference"])

    dataset = Dataset.from_dict(data)

    llm        = LangchainLLMWrapper(ChatMistralAI(model="mistral-small-latest"))
    embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(model="mistral-embed"))

    print("\n🔍 Calcul des métriques Ragas...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    # EvaluationResult n'est pas un dict — conversion via to_pandas()
    df = scores.to_pandas()
    numeric_cols = df.select_dtypes(include="number").columns
    moyennes = {col: nan_to_none(df[col].mean()) for col in numeric_cols}

    print("\n=== Résultats ===")
    for k, v in moyennes.items():
        label = f"{v:.3f}" if v is not None else "N/A"
        print(f"  {k:25s}: {label}")

    # ── Détail par question ──────────────────────────────────────────────────
    # Le df Ragas v0.2+ reprend les mêmes noms que le Dataset d'entrée :
    # user_input, response, retrieved_contexts, reference
    details = []
    for idx, row in df.iterrows():
        entry = {
            "user_input":         row.get("user_input", ""),
            "retrieved_contexts": row.get("retrieved_contexts", []),
            "response":           row.get("response", ""),
            "reference":          row.get("reference", ""),
        }
        for col in numeric_cols:
            entry[col] = nan_to_none(row[col])
        details.append(entry)

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(exist_ok=True)

    output_data = {
        "scores":  moyennes,
        "details": details,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés → {OUTPUT}")

    # Sauvegarde CSV optionnelle (pratique pour Excel / rapport)
    csv_output = OUTPUT.with_suffix(".csv")
    df.to_csv(csv_output, index=False, encoding="utf-8")
    print(f"💾 Scores par question   → {csv_output}")


if __name__ == "__main__":
    main()