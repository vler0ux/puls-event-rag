# Rapport Technique — POC Chatbot RAG Puls-Events

**Auteur** : Véronique  
**Date** : Avril 2025  
**Contexte** : Projet de fin de parcours OpenClassRooms — Ingénieur IA  
**Dépôt** : `puls-events-rag`

---

## 1. Contexte et objectifs

### 1.1 Présentation du projet

Puls-Events est une plateforme de recommandations d'événements culturels destinée à la région de Grenoble et de l'Isère. Dans le cadre de ce POC (Proof of Concept), l'objectif est de démontrer la faisabilité technique d'un chatbot capable de répondre à des questions en langage naturel sur les événements culturels à venir, sans que l'utilisateur n'ait à naviguer dans des listes ou des filtres.

### 1.2 Objectifs du POC

1. Collecter et structurer automatiquement les données événementielles (OpenAgenda)
2. Indexer les descriptions dans une base vectorielle sémantique (FAISS)
3. Permettre des requêtes en langage naturel via un modèle de langage (Mistral AI)
4. Exposer le système via une API REST documentée (FastAPI)
5. Conteneuriser la solution pour un déploiement reproductible (Docker)
6. Évaluer automatiquement la qualité du système (Ragas)

---

## 2. Choix du modèle NLP

### 2.1 Sélection de Mistral AI

Le modèle retenu pour ce projet est **Mistral AI**, combinant deux composants :

| Rôle | Modèle | Usage |
|------|--------|-------|
| Génération de réponses | `mistral-large-latest` | Production des réponses finales |
| Embeddings sémantiques | `mistral-embed` | Vectorisation des événements et des requêtes |

### 2.2 Justification du choix

**Mistral AI a été préféré à d'autres alternatives pour les raisons suivantes :**

**Cohérence vectorielle.** Utiliser le même fournisseur pour les embeddings (`mistral-embed`) et la génération (`mistral-large-latest`) garantit une cohérence dans l'espace vectoriel. Les vecteurs produits lors de l'indexation et lors de la requête partagent la même représentation interne, ce qui améliore la précision du retrieval.

**Qualité francophone.** Mistral AI est une entreprise française dont les modèles sont particulièrement bien calibrés pour le français. Dans un contexte d'événements culturels à Grenoble, avec des noms de lieux, d'artistes et de salles en français, cette caractéristique est déterminante.

**Dimension des embeddings.** `mistral-embed` produit des vecteurs de dimension 1024, plus riches que les modèles légers (384 dimensions pour `all-MiniLM-L6-v2`), ce qui améliore la granularité de la recherche sémantique.

**Comparaison avec les alternatives écartées :**

| Modèle | Avantages | Raisons de l'exclusion |
|--------|-----------|------------------------|
| OpenAI GPT-4 + text-embedding-3 | Très performant | Coût plus élevé, entreprise américaine |
| HuggingFace SBERT (`all-MiniLM-L6-v2`) | Gratuit, rapide | Embeddings moins riches (384 dim), pas de LLM intégré |
| Ollama (modèle local) | Aucun coût API, confidentialité | Infrastructure lourde, performances moindres en prod |
| FastText | Rapide | Pas de représentation contextuelle, inadapté au RAG |

### 2.3 Adaptation aux spécificités métier

Le modèle a été adapté au domaine via un **system prompt métier** défini dans `rag/chain.py` :

```
Tu es un assistant culturel spécialisé dans les événements de la région de Grenoble.
Réponds toujours en français, de manière chaleureuse et enthousiaste.
Cite le titre, le lieu, la date et l'URL de chaque événement recommandé.
Si le contexte ne contient pas d'événement pertinent, dis-le honnêtement.
Ne génère jamais d'événements fictifs.
```

Ce prompt impose trois contraintes métier essentielles :
- **Format de réponse structuré** : titre + lieu + date + URL pour chaque événement
- **Ancrage géographique** : zone Grenoble / Isère
- **Anti-hallucination** : interdiction explicite d'inventer des événements absents de la base

---

## 3. Architecture du système

### 3.1 Vue d'ensemble

Le système repose sur une architecture RAG (Retrieval-Augmented Generation) en deux phases :

**Phase offline (indexation) :**
```
OpenAgenda API → collect_data.py → events_clean.csv → build_index.py → FAISS index
```

**Phase runtime (requête) :**
```
Utilisateur → FastAPI /ask → LangChain RAG Chain → FAISS (top-4 docs) + Mistral AI → Réponse
```

### 3.2 Stack technique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| LLM | Mistral AI (`mistral-large-latest`) | API v2 |
| Embeddings | Mistral AI (`mistral-embed`) | API v2 |
| Orchestration RAG | LangChain + `langchain-mistralai` | ≥ 0.1 |
| Base vectorielle | FAISS CPU (Meta AI) | `faiss-cpu` |
| API REST | FastAPI + Uvicorn | ≥ 0.100 |
| Données | OpenAgenda via API Opendatasoft | — |
| Évaluation | Ragas | ≥ 0.1 |
| Conteneurisation | Docker (`python:3.11-slim`) | — |
| Gestion dépendances | `uv` (Rust) | ≥ 0.4 |

### 3.3 Choix de FAISS

**FAISS (Facebook AI Similarity Search) a été retenu à la place de ChromaDB ou Pinecone pour les raisons suivantes :**

- **Déploiement simplifié** : index persisté sur disque local, aucun service externe requis
- **Performance CPU** : optimisé pour la recherche par similarité cosinus sans GPU
- **Intégration LangChain native** : `FAISS.load_local()` + `allow_dangerous_deserialization`
- **Adéquation POC** : 545 chunks — FAISS excelle à cette échelle sans configuration complexe

Limite identifiée : FAISS ne supporte pas les mises à jour incrémentales (ajout d'un seul document). Une mise à jour des données nécessite une reconstruction complète de l'index via `build_index.py`.

---

## 4. Données et traitement

### 4.1 Source de données

Les données proviennent de l'**API Opendatasoft / OpenAgenda**, qui agrège les événements culturels de la métropole grenobloise. L'accès est public (pas de clé API requise pour la lecture).

### 4.2 Pipeline de collecte et nettoyage

Implémenté dans `scripts/collect_data.py` :

1. **Pagination** : requêtes HTTP successives avec paramètre `offset` jusqu'à épuisement des résultats
2. **Extraction des champs** : titre, description, lieu, date de début, URL de l'événement, tags thématiques
3. **Nettoyage** : suppression des doublons (déduplication par titre+date), normalisation des dates ISO 8601, filtrage des entrées sans description
4. **Export** : sauvegarde dans `data/processed/events_clean.csv` (516 événements)

### 4.3 Vectorisation et indexation

Implémenté dans `scripts/build_index.py` :

- **Modèle d'embedding** : `mistral-embed` via `MistralAIEmbeddings`
- **Stratégie de découpage** : `RecursiveCharacterTextSplitter` avec `chunk_size=800` et `chunk_overlap=80`
- **Résultat** : 545 chunks générés depuis 516 événements
- **Persistance** : index sauvegardé dans `index/faiss_index/`

Le chevauchement de 80 caractères garantit qu'une information à la jonction de deux chunks n'est pas perdue lors du découpage.

---

## 5. Implémentation de la chaîne RAG

### 5.1 Retrieval

Implémenté dans `rag/retriever.py` :

- Chargement de l'index FAISS depuis le disque au démarrage
- Recherche par similarité cosinus avec `k=4` (les 4 documents les plus proches de la requête)
- La requête utilisateur est elle-même convertie en vecteur via `mistral-embed` avant la recherche

### 5.2 Génération augmentée

Implémenté dans `rag/chain.py` :

- Construction d'un prompt augmenté : system prompt + contexte (4 documents) + question utilisateur
- Appel à `mistral-large-latest` pour la génération
- Retour structuré : réponse textuelle + liste des sources (titre, date, URL)

### 5.3 API REST

Implémenté dans `api/main.py` avec FastAPI :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Statut de l'API |
| `/health` | GET | Healthcheck Docker |
| `/ask` | POST | Question → réponse RAG |
| `/rebuild` | POST | Reconstruction de l'index |
| `/chat` | GET | Interface HTML de démo |
| `/docs` | GET | Documentation Swagger auto-générée |

---

## 6. Résultats et évaluation

### 6.1 Exemple de réponse

**Question** : *"Quels concerts ont lieu à Grenoble en juin ?"*

**Réponse du système** :
```
🎵 Musique amplifié en plein air sur grenoble
   📍 Place Grenette — Grenoble   📅 20 juin 2025
   🔗 openagenda.com/culture/events/musique-amplifie-en-plein-air

🎶 La Chimère en Fête
   📍 La Chimère — Grenoble   📅 21 juin 2025

🎤 Pryeur x Veiga000 x Loup & Err
   📍 5 Rue Auguste Gaché, Grenoble   📅 21 juin 2025

🎼 Michel Musique fête la musique
   📍 Michel Musique — Grenoble   📅 21 juin 2025
```

La réponse est pertinente, sourcée, non-hallucinée, et respecte le format métier défini dans le system prompt.

### 6.2 Évaluation automatique Ragas

Le jeu de test annoté (`docs/qa_dataset.json`) contient 10 paires question/réponse de référence sur des événements culturels grenoblois.

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| **Faithfulness** | 0.663 | Les réponses sont fidèles aux documents récupérés à 66% |
| **Answer Relevancy** | N/A | Non calculé (rate limit API Mistral lors de l'évaluation) |
| **Context Precision** | 0.467 | Les documents récupérés sont pertinents à 47% |

**Analyse des scores :**

Le score de `faithfulness` de 0.663 est acceptable pour un POC. Il indique que le LLM reste globalement ancré dans les documents fournis et ne génère pas d'informations fictives dans la majorité des cas.

Le score de `context_precision` de 0.467 est plus faible. Il révèle que parmi les 4 documents récupérés, certains ne sont pas directement pertinents pour la question posée. Ce résultat s'explique par la granularité du chunking (800 caractères) qui peut inclure des informations périphériques à l'événement cible.

Le score `answer_relevancy` n'a pas pu être calculé en raison des limitations du plan Mistral gratuit (rate limit 429 lors de l'évaluation multi-appels de Ragas).

---

## 7. Limites identifiées

### 7.1 Limites techniques

**Absence de mise à jour incrémentale.** FAISS ne supporte pas l'ajout d'un seul document. Toute mise à jour des données (nouvel événement) requiert une reconstruction complète de l'index, ce qui peut prendre plusieurs minutes.

**Rate limit API Mistral.** Sur le plan gratuit, le nombre d'appels par minute est limité. Cela impacte l'évaluation Ragas (qui génère de nombreux appels simultanés) et pourrait poser problème en production avec un fort trafic.

**Taille du jeu de test.** Le jeu d'évaluation de 10 questions est insuffisant pour une évaluation statistiquement robuste. Les scores Ragas sont à interpréter avec prudence.

**Absence de mémoire conversationnelle.** Chaque appel à `/ask` est indépendant. Le chatbot ne mémorise pas les échanges précédents au sein d'une même conversation.

### 7.2 Limites métier

**Fraîcheur des données.** Les données sont collectées une seule fois et indexées statiquement. Les événements ajoutés après l'indexation ne sont pas accessibles sans reconstruction de l'index.

**Couverture géographique.** L'API OpenAgenda est filtrée sur Grenoble Métropole. Des événements dans des communes limitrophes (Vif, Vizille, Crolles) pourraient être absents.

**Langues.** Le chatbot est exclusivement en français. Des utilisateurs non-francophones ne peuvent pas l'utiliser.

---

## 8. Perspectives et améliorations possibles

### 8.1 Court terme

- **Compléter l'évaluation Ragas** : augmenter le jeu de test à 50+ questions et utiliser un plan Mistral payant pour éviter les rate limits
- **Tests unitaires** : compléter les fichiers `tests/` (couverture actuellement partielle)
- **Mise à jour automatique** : planifier `collect_data.py` + `build_index.py` via un cron job ou un pipeline CI/CD

### 8.2 Moyen terme

- **Passer à ChromaDB** : base vectorielle avec persistance cloud, mises à jour incrémentales et filtrage par métadonnées (date, catégorie, lieu) — plus adaptée à une mise en production
- **Interface Streamlit** : interface de démo plus ergonomique avec historique des échanges et filtres visuels
- **Mémoire conversationnelle** : intégrer `ConversationBufferMemory` de LangChain pour maintenir le contexte entre les questions

### 8.3 Long terme

- **Recommandations personnalisées** : profil utilisateur (historique des événements consultés, préférences de genre musical)
- **Support multilingue** : ajout de l'anglais pour les touristes et les publics internationaux
- **Déploiement cloud** : mise en production sur un hébergeur (Railway, Fly.io, Azure) avec CI/CD GitHub Actions

---

## 9. Conclusion

Ce POC démontre la faisabilité technique d'un chatbot RAG pour la recommandation d'événements culturels. Le pipeline complet est opérationnel, de la collecte des données à la réponse en langage naturel, en passant par l'indexation sémantique et la conteneurisation Docker.

Les principaux acquis techniques sont :

- Maîtrise du pipeline RAG de bout en bout (collect → embed → index → retrieve → generate)
- Intégration de l'API Mistral AI pour les embeddings et la génération
- Orchestration avec LangChain et FAISS
- Exposition via FastAPI avec documentation Swagger automatique
- Évaluation automatisée de la qualité des réponses avec Ragas
- Conteneurisation et reproducibilité avec Docker

Les scores Ragas (faithfulness : 0.663, context precision : 0.467) placent ce système dans une plage correcte pour un POC, avec des axes d'amélioration identifiés principalement sur le retrieval (chunking plus fin, filtrage par métadonnées) et la fraîcheur des données.
