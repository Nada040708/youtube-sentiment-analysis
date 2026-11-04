# YouTube Sentiment Analysis - MLOps Project

## Project Description

This project aims to develop an end-to-end MLOps system for automatically analyzing the sentiment of YouTube video comments.

The system uses a Machine Learning model based on TF-IDF and Logistic Regression to classify comments according to their sentiment.

## Architecture

- Frontend: Chrome Extension (HTML, CSS, JavaScript)
- Backend: FastAPI REST API
- ML Model: Scikit-learn (TF-IDF + Logistic Regression)
- Deployment: Hugging Face Spaces with Docker
- Version Control: Git and GitHub

## Project Structure

- data/: Raw and processed datasets
- models/: Trained ML models
- src/: Source code
- chrome-extension/: Chrome extension
- logs/: Application logs
- tests/: Automated tests

## Installation

Clone the repository and create a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


    # Phase 2 – Data Pipeline and Preparation

## 1. Dataset Download
- Dataset: Reddit Sentiment Analysis.
- Source: GitHub repository.
- Downloaded using Ubuntu terminal (`wget`).
- Storage location: `data/raw/reddit.csv`.

## 2. Data Cleaning
Script: `src/data/clean_data.py`

- Removal of URLs and user mentions.
- Removal of special characters.
- Conversion of text to lowercase.
- Removal of missing values and duplicates.
- Standardization of columns to `text` and `label`.
- Preservation of emojis.

## 3. Exploratory Data Analysis (EDA)
Script: `src/data/eda.py`

- Analysis of sentiment class distribution.
- Calculation of comment length.
- Descriptive statistics.
- Generation of class distribution and text length histograms.

## 4. Data Balancing and Splitting
Script: `src/data/split_data.py`

- Verification of dataset requirements (1000+ comments and at least 300 per class).
- Train/Test split (80% / 20%).
- Stratified sampling to preserve class distribution.
- Random state set to 42 for reproducibility.
- Undersampling applied to the training set if significant class imbalance is detected.

## 5. Output Files

- `data/raw/reddit.csv`: Original dataset.
- `data/processed/reddit_cleaned.csv`: Cleaned dataset.
- `data/processed/train.csv`: Training dataset.
- `data/processed/test.csv`: Testing dataset.
- `data/processed/class_distribution.png`: Class distribution visualization.
- `data/processed/text_length_distribution.png`: Comment length visualization.


## Phase 3 : Entraînement et évaluation du modèle

### Objectif
Entraîner et évaluer un modèle de classification des sentiments des commentaires YouTube en utilisant TF-IDF et Logistic Regression.

### 1. Chargement des données
- Chargement des jeux de données d'entraînement et de test générés lors de la phase 2.
- Extraction des commentaires et de leurs étiquettes de sentiment.
- Vérification des dimensions des jeux de données et de la distribution des classes.

### 2. Construction du pipeline TF-IDF + Logistic Regression
- Création d'un pipeline Scikit-learn combinant TF-IDF et Logistic Regression.
- Transformation des commentaires textuels en vecteurs numériques à l'aide de TF-IDF.
- Utilisation de Logistic Regression comme algorithme principal de classification.

### 3. Optimisation des hyperparamètres
- Utilisation de `RandomizedSearchCV` avec une validation croisée à 3 plis (3-fold cross-validation).
- Optimisation des paramètres de TF-IDF :
  - `max_features`
  - `ngram_range`
  - `min_df`
  - `max_df`
- Optimisation des paramètres de Logistic Regression :
  - `C`
  - `class_weight`
- Utilisation du Macro F1-score comme métrique d'optimisation.

### 4. Comparaison des algorithmes
Comparaison de trois algorithmes de classification à partir des mêmes caractéristiques TF-IDF optimisées :
- Logistic Regression
- Linear SVM
- Random Forest

Les performances sont comparées à l'aide de l'Accuracy et du Macro F1-score.

### 5. Évaluation du modèle
Évaluation du modèle final Logistic Regression à travers :
- Accuracy.
- Macro F1-score.
- Précision (Precision), rappel (Recall) et F1-score pour chaque classe.
- Matrice de confusion (Confusion Matrix).

### 6. Sauvegarde du modèle
Sauvegarde du modèle entraîné et du vectoriseur TF-IDF avec Joblib :

- `models/best_model.joblib`
- `models/tfidf_vectorizer.joblib`

La matrice de confusion est également enregistrée sous :

- `models/confusion_matrix.png`

### 7. Mesure du temps d'inférence
- Mesure du temps de prédiction sur un lot de 50 commentaires.
- Réalisation de 10 exécutions après une première exécution de chauffe (warm-up).
- Calcul de la médiane des temps mesurés.

### Critères de performance

| Métrique | Objectif |
|---|---|
| Accuracy | ≥ 80 % |
| F1-score de chaque classe | > 0,75 |
| Temps d'inférence (50 commentaires) | < 100 ms |

Le script vérifie automatiquement ces critères et affiche si chaque objectif est atteint.

### Fichiers générés

```text
models/
├── best_model.joblib
├── tfidf_vectorizer.joblib
└── confusion_matrix.png
```

## Phase 4 : Développement de l'API REST

### Objectif
Développer une API REST avec FastAPI pour mettre à disposition le modèle de classification des sentiments entraîné lors de la phase 3.

### 1. Structure de l'API
Organisation du code en plusieurs modules :
- `main.py` : création de l'application FastAPI et définition des endpoints.
- `schemas.py` : validation des données avec Pydantic.
- `model.py` : chargement du modèle et réalisation des prédictions.

### 2. Chargement du modèle
- Chargement du modèle Logistic Regression sauvegardé avec Joblib.
- Chargement du vectoriseur TF-IDF.
- Conservation des deux composants en mémoire pour éviter de les recharger à chaque requête.

### 3. Création des endpoints

**GET `/health`**

Vérifie l'état de fonctionnement de l'API et la disponibilité du modèle.

**POST `/predict_batch`**

Permet de :
- Recevoir une liste de commentaires.
- Vérifier les données reçues.
- Prédire le sentiment de chaque commentaire.
- Retourner les commentaires avec leurs sentiments.
- Calculer les statistiques des sentiments prédits.

### 4. Validation des données
Utilisation de Pydantic pour définir et valider la structure des requêtes et des réponses de l'API.

### 5. Configuration CORS
Configuration du middleware CORS pour autoriser les requêtes provenant de l'extension Chrome.

### 6. Gestion des erreurs
- Vérification des commentaires vides.
- Gestion des erreurs liées aux prédictions.
- Retour de codes HTTP et de messages d'erreur appropriés.

### 7. Tests locaux
Lancement de l'API avec Uvicorn :

```bash
uvicorn src.api.main:app --reload
```

Tests réalisés à travers :
- `/health` : vérification de l'état de l'API et du chargement du modèle.
- `/predict_batch` : test de classification de plusieurs commentaires.
- `/docs` : utilisation de la documentation interactive Swagger générée automatiquement par FastAPI.

### Fichiers créés

```text
src/api/
├── __init__.py
├── main.py
├── schemas.py
└── model.py
```

### Dépendances ajoutées
- FastAPI
- Uvicorn
- Pydantic

## Phase 5 – Développement de l'extension Chrome

### Objectif
Développer une interface Chrome permettant d'extraire les commentaires YouTube et d'afficher leurs analyses de sentiment à partir de l'API FastAPI.

### Réalisations

- Création de l'extension avec Chrome Manifest V3.
- Configuration des permissions et des scripts.
- Extraction des commentaires visibles avec JavaScript.
- Développement d'une interface avec HTML et CSS.
- Communication avec l'API REST FastAPI.
- Envoi des commentaires par lots de 50.
- Affichage des statistiques et de la répartition des sentiments.
- Visualisation graphique avec Chart.js.
- Affichage des commentaires, sentiments et scores de confiance.
- Ajout de filtres par sentiment.
- Implémentation du mode sombre et clair.
- Ajout des fonctions de copie et d'export CSV.
- Gestion des erreurs et des états de chargement.

### Fichiers créés

```text
chrome-extension/
├── manifest.json
├── popup.html
├── popup.css
├── popup.js
└── content.js
```

### Intégration avec l'API

L'extension communique avec l'endpoint `/predict_batch` de FastAPI. Les commentaires sont envoyés par lots et les résultats retournés comprennent le commentaire, le sentiment prédit et le score de confiance.

### Tests

- Chargement de l'extension dans Chrome.
- Extraction des commentaires d'une vidéo YouTube.
- Vérification de la communication avec l'API.
- Affichage des statistiques et du graphique.
- Vérification des filtres et du changement de thème.
- Test de la copie et de l'export CSV.
- Vérification de la gestion des erreurs.

### Questions de réflexion

**1. Gestion des threads :** conserver le lien entre chaque réponse et son commentaire parent.

**2. Extraction dynamique :** utiliser MutationObserver et le défilement progressif.

**3. Optimisation :** utiliser le traitement par lots et éviter les extractions répétées.

