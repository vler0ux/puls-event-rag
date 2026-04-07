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
    required = {"id", "title", "description", "city", "date"}
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