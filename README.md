# PFA – RL Portfolio API

API FastAPI exposant le système de trading par Reinforcement Learning (PPO) avec optimisation de portefeuille (PyPortfolioOpt).

---

## 🚀 Déploiement sur Railway (gratuit)

### Étape 1 – Préparer le dépôt GitHub

```bash
# Initialiser le repo local
git init
git add .
git commit -m "Initial commit – PFA RL API"

# Pousser sur GitHub (créer d'abord le repo sur github.com)
git remote add origin https://github.com/VOTRE_USERNAME/pfa-rl-api.git
git branch -M main
git push -u origin main
```

### Étape 2 – Déployer sur Railway

1. Aller sur [railway.app](https://railway.app) → **New Project**
2. Choisir **Deploy from GitHub repo**
3. Sélectionner votre repo `pfa-rl-api`
4. Railway détecte automatiquement Python et installe les dépendances
5. Aller dans **Settings → Networking** → **Generate Domain**
6. ✅ Votre API est accessible à l'URL générée, ex : `https://pfa-rl-api-production.up.railway.app`

---

## 📡 Endpoints de l'API

| Méthode | Endpoint             | Description                                      |
|---------|----------------------|--------------------------------------------------|
| GET     | `/`                  | Informations générales sur l'API                 |
| GET     | `/health`            | Healthcheck                                      |
| POST    | `/scrape-news`       | Scraping Wikipedia + embeddings NLP              |
| POST    | `/train`             | Entraînement des agents PPO                      |
| POST    | `/optimize-portfolio`| Optimisation du portefeuille (Max Sharpe)        |
| GET     | `/results`           | Derniers résultats disponibles                   |

### Documentation interactive

Une fois déployée, accéder à :
- **Swagger UI** : `https://VOTRE_URL/docs`
- **ReDoc** : `https://VOTRE_URL/redoc`

---

## 🔧 Exemples d'utilisation

### 1. Scraper les actualités

```bash
curl -X POST https://VOTRE_URL/scrape-news \
  -H "Content-Type: application/json" \
  -d '{"start_year": 2020, "end_year": 2024, "assets": ["AAPL", "NVDA", "BTC-USD"]}'
```

### 2. Entraîner les agents PPO

```bash
curl -X POST https://VOTRE_URL/train \
  -H "Content-Type: application/json" \
  -d '{"train_size": 1800, "n_timesteps": 50000, "window_size": 20}'
```

### 3. Optimiser le portefeuille

```bash
curl -X POST https://VOTRE_URL/optimize-portfolio \
  -H "Content-Type: application/json" \
  -d '{"gamma_l2": 0.1}'
```

### 4. Récupérer les résultats

```bash
curl https://VOTRE_URL/results
```

---

## ⚙️ Variables d'environnement

Aucune variable obligatoire. Railway injecte automatiquement `$PORT`.

---

## 🗂️ Structure du projet

```
pfa_api/
├── main.py            # Application FastAPI
├── requirements.txt   # Dépendances Python
├── Procfile           # Commande de démarrage
├── railway.toml       # Configuration Railway
└── README.md          # Ce fichier
```
