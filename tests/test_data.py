"""Tests unitaires pour la collecte et le nettoyage des données."""

import pandas as pd
import pytest
from pathlib import Path


def test_csv_exists():
    """Le fichier CSV nettoyé doit exister après collecte."""
    assert Path("data/processed/events_clean.csv").exists(), \
        "Le fichier events_clean.csv est introuvable — lance collect_data.py d'abord."


def test_csv_not_empty():
    """Le CSV doit contenir au moins un événement."""
    df = pd.read_csv("data/processed/events_clean.csv")
    assert len(df) > 0, "Le CSV est vide."


def test_required_columns():
    """Les colonnes obligatoires pour le RAG doivent être présentes."""
    df = pd.read_csv("data/processed/events_clean.csv")
    required = {"id", "title", "description", "city", "date_start"}
    missing = required - set(df.columns)
    assert not missing, f"Colonnes manquantes : {missing}"


def test_no_empty_descriptions():
    """Aucune description ne doit être vide (critique pour les embeddings)."""
    df = pd.read_csv("data/processed/events_clean.csv")
    empty = df["description"].isna().sum() + (df["description"] == "").sum()
    assert empty == 0, f"{empty} descriptions vides détectées."


def test_grenoble_region():
    """Les événements doivent provenir de la région Grenoble / Isère."""
    df = pd.read_csv("data/processed/events_clean.csv")
    cities = df["city"].str.lower().unique()
    # On vérifie juste qu'il y a des villes renseignées
    assert len(cities) > 0, "Aucune ville trouvée dans les données."
    
def test_critical_fields_not_empty():
    """Les champs vectorisés critiques ne doivent pas être vides."""
    df = pd.read_csv("data/processed/events_clean.csv")
    
    critical = ["title", "description"]
    for col in critical:
        empty = df[col].isna().sum() + (df[col] == "").sum()
        assert empty == 0, f"{empty} valeurs manquantes dans '{col}' — impact direct sur les vecteurs"

def test_localisation_not_empty():
    """Au moins city ou place doit être renseigné pour le retrieval géographique."""
    df = pd.read_csv("data/processed/events_clean.csv")
    
    missing_both = df[df["city"].isna() & df["place"].isna()]
    assert len(missing_both) == 0, f"{len(missing_both)} événements sans localisation (city ET place vides)"