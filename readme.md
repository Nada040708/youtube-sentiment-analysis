# YouTube Sentiment Analyzer

## 1. Description du projet

Application d'analyse automatique des sentiments des commentaires YouTube. Le système classe les commentaires en trois catégories : Positive, Neutral et Negative.

Une extension Chrome extrait les commentaires visibles et les transmet à une API REST qui utilise un modèle de Machine Learning pour effectuer les prédictions et calculer les scores de confiance.

## 2. Architecture technique

- **Prétraitement :** Pandas, NumPy
- **Vectorisation :** TF-IDF
- **Classification :** Logistic Regression
- **Optimisation :** RandomizedSearchCV
- **Comparaison :** SVM et Random Forest
- **API REST :** FastAPI, Pydantic
- **Extension Chrome :** HTML, CSS, JavaScript, Chart.js
- **Conteneurisation :** Docker
- **Déploiement :** GitHub Codespaces
- **Versionnement :** Git, GitHub

### Architecture globale

YouTube → Extension Chrome → API FastAPI → TF-IDF → Logistic Regression → Prédictions → Tableau de bord.

## 3. Installation et utilisation

### Prérequis

- Python 3.12
- Google Chrome
- Docker
- Git

### Installation

Cloner le dépôt :

    git clone https://github.com/Nada040708/youtube-sentiment-analysis.git
    cd youtube-sentiment-analysis

Créer l'environnement virtuel :

    python3 -m venv venv
    source venv/bin/activate

Installer les dépendances :

    pip install -r requirements.txt

### Lancement de l'API locale

    uvicorn src.api.main:app --reload

Documentation interactive : http://127.0.0.1:8000/docs

### Installation de l'extension Chrome

1. Ouvrir chrome://extensions/.
2. Activer le mode développeur.
3. Cliquer sur « Charger l'extension non empaquetée ».
4. Sélectionner le dossier chrome-extension/.

### Utilisation

1. Ouvrir une vidéo YouTube.
2. Faire défiler les commentaires pour charger les commentaires visibles.
3. Ouvrir l'extension YouTube Sentiment Analyzer.
4. Cliquer sur « Analyser ».
5. Consulter les statistiques, le graphique et les prédictions.
6. Utiliser les filtres, le mode sombre, la copie ou l'export CSV.

### Déploiement Docker

Construire l'image :

    docker build -t youtube-sentiment-api .

Lancer le conteneur :

    docker run -d -p 7860:7860 -e PORT=7860 youtube-sentiment-api

L'API est accessible via l'URL HTTPS publique de GitHub Codespaces lorsque le Codespace est actif.
