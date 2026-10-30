"""
feedback_report.py
Tableau de bord du feedback utilisateur (👍/👎) collecté par l'interface /chat.

Usage : uv run python scripts/feedback_report.py
"""

import json
from collections import Counter
from pathlib import Path

FEEDBACK_PATH = Path("data/feedback.jsonl")
FALLBACK_MARKER = "pas trouvé d"   # même repère que l'interface /chat
EXCERPT_LEN = 120


def load_feedback(path: Path) -> list[dict]:
    """Lit le fichier JSONL en ignorant les lignes vides ou corrompues."""
    entries = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  Ligne {n} illisible, ignorée.")
    return entries


def excerpt(text: str, length: int = EXCERPT_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + "…"


def main():
    if not FEEDBACK_PATH.exists():
        print(f"Aucun feedback pour l'instant ({FEEDBACK_PATH} introuvable).")
        return

    entries = load_feedback(FEEDBACK_PATH)
    if not entries:
        print("Le fichier de feedback est vide.")
        return

    ratings = Counter(e.get("rating") for e in entries)
    total = len(entries)
    positives = ratings.get(1, 0)
    negatives = ratings.get(-1, 0)
    fallbacks = [e for e in entries if FALLBACK_MARKER in e.get("answer", "")]
    dates = sorted(e["timestamp"][:10] for e in entries if "timestamp" in e)

    print("=== Tableau de bord du feedback Puls-Events ===\n")
    print(f"Période             : {dates[0]} → {dates[-1]}" if dates else "Période : inconnue")
    print(f"Avis reçus          : {total}")
    print(f"👍 Satisfaits        : {positives} ({positives / total:.0%})")
    print(f"👎 Insatisfaits      : {negatives} ({negatives / total:.0%})")
    print(f"Réponses de repli   : {len(fallbacks)} (« je n'ai pas trouvé »)")

    negative_entries = [e for e in entries if e.get("rating") == -1]
    if negative_entries:
        print("\n--- Questions notées 👎 (à analyser en priorité) ---")
        for e in negative_entries:
            tag = " [repli]" if FALLBACK_MARKER in e.get("answer", "") else ""
            print(f"\n• {e.get('timestamp', '')[:16]}{tag}")
            print(f"  Q : {e.get('question', '')}")
            print(f"  R : {excerpt(e.get('answer', ''))}")
    else:
        print("\nAucune réponse notée 👎 pour l'instant.")


if __name__ == "__main__":
    main()