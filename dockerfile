# ── Dockerfile ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Métadonnées
LABEL project="puls-events-rag"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# Dépendances système (FAISS a besoin de libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances en premier (cache Docker optimisé)
COPY pyproject.toml ./
COPY uv.lock* ./

# Installer les dépendances
RUN uv pip install -e .

# Copier le reste du projet
COPY rag/ ./rag/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/processed/ ./data/processed/
COPY index/ ./index/

# Port exposé par FastAPI/Uvicorn
EXPOSE 8000

# Lancement
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]