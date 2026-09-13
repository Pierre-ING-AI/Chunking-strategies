# Étude comparative de stratégies de chunking pour RAG

Ce projet évalue et compare différentes stratégies de **chunking** (découpage de texte) utilisées en amont d'un pipeline **RAG** (*Retrieval-Augmented Generation*), afin de déterminer laquelle offre la meilleure qualité de récupération et de génération de réponses.

Le corpus utilisé pour les tests est le texte du livre blanc du **Bitcoin** (`Bitcoin.txt`).

## 🎯 Objectif

Le chunking consiste à découper un document volumineux en sous-sections plus petites afin qu'elles puissent être vectorisées et exploitées efficacement par un LLM. Le choix de la stratégie de découpage a un impact direct sur la pertinence des réponses générées par un système RAG.

Ce notebook implémente plusieurs approches de chunking, construit un pipeline RAG (indexation FAISS + retrieval + génération) pour chacune, puis les évalue avec le framework **RAGAS** sur un jeu de questions/réponses de référence.

## 🧩 Stratégies de chunking étudiées

| # | Stratégie | Principe |
|---|-----------|----------|
| 1 | **Fixed-size chunking** | Découpage strict par nombre de caractères (avec overlap), sans tenir compte de la structure du texte. |
| 2 | **Recursive Character Splitting** | Découpage hiérarchique basé sur des séparateurs structurels (titres, paragraphes, phrases). |
| 3 | **Token Chunking** | Découpage strict basé sur le nombre de tokens (tokenizer `cl100k_base`). |
| 4 | **Recursive Token Chunking** | Combinaison du découpage récursif et d'une taille maximale exprimée en tokens. |
| 5 | **Semantic Chunking** | Découpage basé sur la similarité sémantique entre phrases consécutives (embeddings). |
| 6 | **Cluster Semantic Chunking** *(présenté, non implémenté)* | Regroupement des phrases par clustering non supervisé (K-Means, HDBSCAN, etc.). |
| 7 | **Agentic / LLM-based Chunking** | Un LLM détermine lui-même les ruptures d'idées et découpe le texte selon une logique métier. |
| 8 | **Late Chunking** *(présenté, non implémenté)* | Le document entier est embeddé globalement avant d'être sous-découpé, pour préserver le contexte. |

## 📊 Méthodologie d'évaluation

Pour chaque stratégie de chunking :

1. Les chunks générés sont indexés dans une base vectorielle **FAISS** (embeddings `text-embedding-3-small`).
2. Un **retriever** (top-k = 3) récupère les passages pertinents pour chaque question d'un jeu de test.
3. Un LLM (**gpt-4o-mini**) génère une réponse à partir du contexte récupéré.
4. Les résultats sont évalués avec **RAGAS** sur 4 métriques :
   - `LLMContextPrecisionWithReference`
   - `LLMContextRecall`
   - `Faithfulness`
   - `ResponseRelevancy`

Un tableau comparatif final agrège les scores moyens par stratégie.

## 📦 Prérequis

- Python ≥ 3.10
- Une clé API OpenAI (variable d'environnement `OPENAI_API_KEY`, chargée via un fichier `.env`)
- Le fichier de corpus `Bitcoin.txt` à la racine du projet

### Dépendances principales

```bash
pip install python-dotenv langchain langchain-text-splitters langchain-openai \
            langchain-community langchain-experimental langchain-core \
            faiss-cpu pandas pydantic ragas
```

## ⚙️ Configuration

Créer un fichier `.env` à la racine du projet :

```env
OPENAI_API_KEY=sk-...
```

## ▶️ Utilisation

1. Placer le fichier `Bitcoin.txt` (ou tout autre corpus texte) à la racine du projet.
2. Ouvrir et exécuter le notebook `Chunking_rag.ipynb` cellule par cellule :
   - Chargement du texte
   - Génération des chunks pour chaque stratégie
   - Construction du pipeline RAG (FAISS + retriever + LLM)
   - Évaluation RAGAS de chaque stratégie
   - Génération du tableau comparatif final

⚠️ Le pipeline effectue de nombreux appels à l'API OpenAI (embeddings + complétions + évaluation RAGAS) ; des erreurs `RateLimitError` (429) peuvent survenir selon les quotas du compte utilisé.

## 📈 Résultat

Le notebook produit un tableau comparatif (`summary`) résumant, pour chaque stratégie de chunking, les scores moyens obtenus sur les 4 métriques RAGAS, permettant d'identifier la stratégie la plus performante pour le corpus considéré.

## 🗂️ Structure attendue du projet

```
.
├── Chunking_rag.ipynb   # Notebook principal
├── Bitcoin.txt           # Corpus de texte utilisé pour les tests
├── .env                  # Clé API OpenAI (non versionné)
└── README.md
```
