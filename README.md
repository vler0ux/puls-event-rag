# 🎭 Puls-Events RAG — Chatbot de recommandations culturelles

> POC d'un système de Retrieval-Augmented Generation (RAG) pour recommander des événements culturels dans la région de Grenoble / Isère, développé dans le cadre d'une mission freelance pour **Puls-Events**.

---

## 📋 Contexte

Puls-Events souhaite intégrer un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels à venir. Ce POC démontre la faisabilité technique d'un pipeline RAG combinant :

- **Retrieval** : recherche sémantique dans une base vectorielle FAISS
- **Augmented Generation** : génération de réponses naturelles via Mistral AI
- **API REST** : exposition du système via FastAPI pour les équipes produit et marketing

---

## 🏗️ Architecture

```
Utilisateur
    │
    ▼
FastAPI /ask
    │
    ▼
LangChain RAG Chain
    ├── Recherche FAISS (similarité sémantique)
    │       └── Embeddings Mistral
    └── Génération Mistral LLM
            └── Réponse augmentée
                    ▼
            Utilisateur
```

### Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM & Embeddings | Mistral AI (API) |
| Orchestration RAG | LangChain |
| Base vectorielle | FAISS (CPU) |
| API REST | FastAPI + Uvicorn |
| Données | Opendatasoft / OpenAgenda |
| Évaluation | Ragas |
| Conteneurisation | Docker |
| Gestion dépendances | uv |

---

## 📁 Structure du projet

```
puls-events-rag/
├── .env          # Modèle de configuration (jamais versionné : .env)
├── .gitignore
├── README.md
├── pyproject.toml          # Dépendances (uv)
├── rapport_technique.md
├── Dockerfile
│
├── data/
│   ├── raw/                # Données brutes OpenAgenda (ignoré par git)
│   └── processed/          # CSV nettoyé prêt à indexer (ignoré par git)
│
├── index/                  # Index FAISS sauvegardé (ignoré par git)
│
├── scripts/
│   ├── collect_data.py     # Collecte via API Opendatasoft
│   ├── collect_check.py    
│   ├── build_index.py      # Vectorisation + indexation FAISS
│   └── evaluate_rag.py     # Évaluation automatique Ragas
│
├── rag/
│   ├── retriever.py        # Recherche sémantique FAISS
│   └── chain.py            # Chaîne LangChain RAG
│
├── api/
│   └── main.py             # API FastAPI
│
├── tests/
│   ├── test_data.py        # Tests collecte et nettoyage
│   ├── test_index.py       # Tests indexation FAISS
│   └── test_api.py         # Tests API REST
│
└── docs/
    ├──  evaluation_resultsRagas.json
    └── qa_dataset.json     # Jeu de test annoté (questions / réponses de référence)
```

---

## 🚀 Installation et lancement

### Prérequis

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) installé (`pip install uv` ou via script officiel)
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### 1. Cloner le dépôt

```bash
git clone https://github.com/vler0ux/puls-events-rag.git
cd puls-events-rag
```

### 2. Créer l'environnement et installer les dépendances

```bash
uv sync
```

### 3. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env et renseigner MISTRAL_API_KEY
```

### 4. Collecter les données

```bash
uv run python scripts/collect_data.py
```

Récupère les événements culturels de Grenoble Métropole depuis l'API Opendatasoft (OpenAgenda). Produit :
- `data/raw/events_raw.json`
- `data/processed/events_clean.csv` (~500 événements)

### 5. Construire l'index vectoriel

```bash
uv run python scripts/build_index.py
```

Génère les embeddings via Mistral et indexe dans FAISS. Produit :
- `index/faiss_index/`

### 6. Lancer l'API

```bash
uv run uvicorn api.main:app --reload
```

L'API est disponible sur `http://localhost:8000` avec la documentation Swagger sur `http://localhost:8000/docs`.

---

## 🐳 Lancement via Docker

```bash
# Build
docker build -t puls-events-rag .

# Run
docker run -p 8000:8000 --env-file .env puls-events-rag
```

---

## 🔌 Endpoints API

### `POST /ask`
Pose une question et reçoit une réponse augmentée.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts ont lieu à Grenoble ce mois-ci ?"}'
```

Réponse :
```json
{
  "question": "Quels concerts ont lieu à Grenoble ce mois-ci ?",
  "answer": "Voici les concerts à venir à Grenoble ...",
  "sources": [...]
}
```

### `POST /rebuild`
Reconstruit l'index vectoriel à partir des données.

```bash
curl -X POST http://localhost:8000/rebuild
```

---

## 🧪 Tests

```bash
# Tests de collecte des données
uv run pytest tests/test_data.py -v

# Tests d'indexation FAISS
uv run pytest tests/test_index.py -v

# Tests de l'API
uv run pytest tests/test_api.py -v

# Tous les tests
uv run pytest -v
```

---

## 📊 Évaluation

Le système est évalué avec **Ragas** sur un jeu de 20 questions/réponses annotées (`docs/qa_dataset.json`) :

```bash
uv run python scripts/evaluate_rag.py
```

Métriques mesurées : faithfulness, context precision.

---

## 🗺️ Données

- **Source** : [Opendatasoft / OpenAgenda](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda)
- **Zone** : Grenoble Métropole (Grenoble, Échirolles, Meylan, Saint-Martin-d'Hères, et autres communes)
- **Période** : 12 mois glissants (passé + à venir)
- **Volume** : ~500 événements culturels après nettoyage

---

## 📄 Licence

Projet réalisé dans le cadre d'une formation OpenClassRooms. Données événementielles sous licence ouverte OpenAgenda.