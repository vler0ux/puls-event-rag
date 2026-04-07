"""
collect_data.py
Collecte les événements culturels depuis l'API publique Opendatasoft
(agrégateur OpenAgenda), filtrés sur Grenoble / Isère.
Aucune clé API nécessaire.
"""

import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Configuration ---
BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)

# Fenêtre temporelle : 12 mois en arrière → 12 mois en avant
NOW = datetime.utcnow()
DATE_FROM = (NOW - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
DATE_TO   = (NOW + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

# Villes ciblées (Grenoble + communes de la métropole)
TARGET_CITIES = [
    "Grenoble", "Échirolles", "Meylan", "Gières",
    "Saint-Martin-d'Hères", "Eybens", "Seyssins", "Claix"
]

OUTPUT_RAW = Path("data/raw/events_raw.json")
OUTPUT_CSV = Path("data/processed/events_clean.csv")


def fetch_events_for_city(city: str, max_records: int = 500) -> list[dict]:
    """Récupère les événements pour une ville via l'API Opendatasoft."""
    all_records = []
    offset = 0
    limit = 100  # max autorisé par l'API

    while offset < max_records:
        params = {
            "limit": limit,
            "offset": offset,
            "refine": f'location_city:"{city}"',
            "where": f'lastdate_end >= "{DATE_FROM}" AND firstdate_begin <= "{DATE_TO}"',
            "lang": "fr",
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  ⚠️  Erreur pour {city} (offset={offset}) : {e}")
            break

        records = data.get("results", [])
        if not records:
            break

        all_records.extend(records)

        total = data.get("total_count", 0)
        print(f"  {city} : {len(all_records)}/{total} événements récupérés")

        if len(all_records) >= total:
            break

        offset += limit
        time.sleep(0.3)

    return all_records


def clean_events(raw_records: list[dict]) -> pd.DataFrame:
    """Extrait et nettoie les champs utiles pour le RAG."""
    records = []

    for rec in raw_records:
        title = rec.get("title_fr") or rec.get("title") or "Sans titre"

        description = (
            rec.get("description_fr")
            or rec.get("longdescription_fr")
            or rec.get("description")
            or ""
        )
        if not description.strip():
            continue

        # Dans clean_events, remplace le bloc records.append par :
        keywords = rec.get("keywords_fr", [])
        if isinstance(keywords, list):
            tags = ", ".join(keywords)
        else:
            tags = str(keywords) if keywords else ""

        records.append({
            "id":          rec.get("uid", ""),
            "title":       title.strip(),
            "description": description.strip(),
            "place":       rec.get("location_name", ""),
            "city":        rec.get("location_city", ""),
            "address":     rec.get("location_address", ""),
            "date_start":  rec.get("firstdate_begin", ""),
            "date_end":    rec.get("lastdate_end", ""),
            "tags":        tags,
            "url":         rec.get("canonicalurl", ""),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["title", "date_start"])
    df = df.sort_values("date_start").reset_index(drop=True)
    return df


def main():
    print("=== Collecte Opendatasoft / OpenAgenda — Grenoble Métropole ===")
    print(f"Période : {DATE_FROM[:10]} → {DATE_TO[:10]}\n")

    all_raw = []

    for city in TARGET_CITIES:
        print(f"\n📍 Ville : {city}")
        records = fetch_events_for_city(city, max_records=500)
        all_raw.extend(records)

    print(f"\n✅ Total brut : {len(all_raw)} événements\n")

    # Sauvegarde JSON brut
    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)
    print(f"💾 Données brutes → {OUTPUT_RAW}")

    # Nettoyage
    df = clean_events(all_raw)
    print(f"✅ Après nettoyage : {len(df)} événements utilisables")

    # Sauvegarde CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"💾 Données nettoyées → {OUTPUT_CSV}")

    # Aperçu
    print("\n--- Aperçu ---")
    print(df[["title", "city", "date_start", "tags"]].head(10).to_string())


if __name__ == "__main__":
    main()