# Guide de Déploiement sur Railway

Ce guide vous explique comment déployer l'API PFA sur Railway en quelques minutes.

## Prérequis

- Un compte GitHub (vous en avez un ✓)
- Un compte Railway (gratuit)

## Étapes de Déploiement

### 1. Créer un compte Railway

1. Allez sur [railway.app](https://railway.app)
2. Cliquez sur "Sign Up"
3. Connectez-vous avec GitHub (recommandé)
4. Autorisez Railway à accéder à vos dépôts

### 2. Créer un nouveau projet

1. Sur le dashboard Railway, cliquez sur "New Project"
2. Sélectionnez "Deploy from GitHub repo"
3. Cherchez et sélectionnez le dépôt `pfa-project`
4. Cliquez sur "Deploy Now"

### 3. Configuration automatique

Railway va automatiquement :
- Détecter le `railway.toml` dans votre projet
- Installer les dépendances Python
- Lancer l'application avec : `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 4. Récupérer l'URL de l'API

Une fois le déploiement terminé (5-10 minutes) :

1. Allez dans votre projet Railway
2. Cliquez sur le service "web"
3. Allez dans l'onglet "Settings"
4. Cherchez "Public URL" ou "Domain"
5. Vous verrez une URL comme : `https://pfa-project-production-xxxx.railway.app`

### 5. Tester l'API

Une fois déployée, testez les endpoints :

```bash
# Health check
curl https://votre-url.railway.app/health

# Voir les infos de l'API
curl https://votre-url.railway.app/

# Scraper les données
curl -X POST https://votre-url.railway.app/scrape-news \
  -H "Content-Type: application/json" \
  -d '{
    "start_year": 2020,
    "end_year": 2025,
    "assets": ["AAPL", "NVDA", "BTC-USD"]
  }'
```

## Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Infos sur l'API |
| `/health` | GET | Healthcheck |
| `/scrape-news` | POST | Scraper Wikipedia + embeddings |
| `/train` | POST | Entraîner les agents PPO |
| `/optimize-portfolio` | POST | Optimiser le portefeuille |
| `/results` | GET | Récupérer les résultats |

## Utiliser l'API avec Lovable

Une fois que vous avez l'URL de l'API, vous pouvez l'utiliser dans Lovable :

```javascript
// Exemple en JavaScript
const API_URL = "https://votre-url.railway.app";

// Appeler l'API
const response = await fetch(`${API_URL}/results`);
const data = await response.json();
```

## Troubleshooting

### L'API ne démarre pas
- Vérifiez que `requirements.txt` contient toutes les dépendances
- Consultez les logs dans Railway : Project → Logs

### Erreur 502 Bad Gateway
- L'API peut être en train de démarrer (peut prendre 2-3 minutes)
- Attendez quelques minutes et réessayez

### Erreur de mémoire
- Railway offre 512MB gratuitement
- Si vous avez besoin de plus, mettez à jour votre plan

## Mise à jour de l'API

À chaque fois que vous poussez des modifications sur GitHub :

1. Railway détecte automatiquement les changements
2. Redéploie l'application
3. Aucune action manuelle requise !

## Support

Pour plus d'aide :
- [Documentation Railway](https://docs.railway.app)
- [Communauté Railway](https://railway.app/community)
