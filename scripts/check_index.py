# uv run python scripts/check_index.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 

from rag.retriever import load_vectorstore
import pandas as pd

vs = load_vectorstore()
df = pd.read_csv("data/processed/events_clean.csv")

print(f"📄 Événements dans le CSV   : {len(df)}")
print(f"🔢 Vecteurs dans FAISS      : {vs.index.ntotal}")
print(f"📊 Ratio chunks/événement   : {vs.index.ntotal / len(df):.2f}")