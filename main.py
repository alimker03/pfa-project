"""
API FastAPI – Système de Trading RL (PPO) + Optimisation de Portefeuille
Endpoints exposés :
  GET  /                        → Infos sur l'API
  GET  /health                  → Healthcheck
  POST /scrape-news             → Lance le scraping Wikipedia + embeddings
  POST /train                   → Entraîne les agents PPO
  POST /optimize-portfolio      → Optimise le portefeuille (Max Sharpe)
  GET  /results                 → Retourne les derniers résultats
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from typing import Optional
from collections import deque

warnings.filterwarnings("ignore")

app = FastAPI(
    title="PFA – RL Portfolio API",
    description="API de trading par Reinforcement Learning (PPO) avec optimisation de portefeuille.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── État global en mémoire ────────────────────────────────────────────────────
state = {
    "df_finale": None,
    "best_histories": {},
    "cleaned_weights": {},
    "portfolio_performance": {},
    "metrics": {},
    "status": "idle",
    "last_updated": None,
}

# ══════════════════════════════════════════════════════════════════════════════
# Schémas Pydantic
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    start_year: int = 2020
    end_year: int = 2025
    assets: list[str] = ["AAPL", "NVDA", "BTC-USD"]

class TrainRequest(BaseModel):
    train_size: int = 1800
    n_timesteps: int = 50000
    window_size: int = 20
    learning_rate: float = 3e-4
    assets: list[str] = ["AAPL", "NVDA", "BTC-USD"]

class OptimizeRequest(BaseModel):
    gamma_l2: float = 0.1

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "api": "PFA – RL Portfolio API",
        "version": "1.0.0",
        "description": "Système de trading par PPO avec PyPortfolioOpt",
        "endpoints": {
            "GET  /health": "Healthcheck",
            "POST /scrape-news": "Scraping Wikipedia + embeddings NLP",
            "POST /train": "Entraînement agents PPO",
            "POST /optimize-portfolio": "Optimisation portefeuille (Max Sharpe)",
            "GET  /results": "Derniers résultats",
        },
        "status": state["status"],
        "last_updated": state["last_updated"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─── 1. Scraping + Embeddings ─────────────────────────────────────────────────

@app.post("/scrape-news")
def scrape_news(req: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Lance le scraping Wikipedia (Business & Economy) et génère les embeddings
    sentence-transformers, puis télécharge les cours via yfinance.
    Traitement en arrière-plan – vérifier /results ou /health pour le statut.
    """
    state["status"] = "scraping"
    background_tasks.add_task(_run_scrape, req)
    return {"message": "Scraping lancé en arrière-plan", "config": req.dict()}


def _run_scrape(req: ScrapeRequest):
    try:
        import requests as req_lib
        from bs4 import BeautifulSoup
        import yfinance as yf
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import StandardScaler
        import time

        years_list = list(range(req.start_year, req.end_year + 1))
        month_list = ['January','February','March','April','May','June',
                      'July','August','September','October','November','December']
        headers = {'User-Agent': 'Mozilla/5.0'}

        date_list, event_list = [], []
        for y in years_list:
            for m in month_list:
                url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{m}_{y}"
                try:
                    page = req_lib.get(url, headers=headers, timeout=10)
                    if page.status_code != 200:
                        continue
                    soup = BeautifulSoup(page.content, 'lxml')
                    for day in soup.find_all('div', class_='vevent'):
                        date_tag = day.find('span', class_='bday')
                        current_date = date_tag.get_text(strip=True) if date_tag else np.nan
                        for tag_type, finder in [('div', {'role':'heading'}), ('b', {})]:
                            for h in day.find_all(tag_type, finder):
                                if "Business and economy" in h.get_text():
                                    ul = h.find_next_sibling('ul')
                                    if ul:
                                        for li in ul.find_all('li'):
                                            event_list.append(li.get_text(strip=True))
                                            date_list.append(current_date)
                                    break
                    time.sleep(0.1)
                except Exception:
                    continue

        df_news = pd.DataFrame({'date': date_list, 'events': event_list})
        df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce')
        df_news = df_news.dropna(subset=['date'])
        df_news = df_news.groupby('date')['events'].apply(lambda x: ' '.join(x)).reset_index()

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df_news["events"].tolist())
        df_vectors = pd.DataFrame(embeddings, index=df_news["date"])
        df_daily_vectors = df_vectors.groupby(df_vectors.index).mean()

        actions = req.assets
        df_actions = yf.download(actions, start=f"{req.start_year}-01-04",
                                  end=f"{req.end_year}-12-31", progress=False)["Close"]
        df_actions.index = pd.to_datetime(df_actions.index)
        df_action_no_scaler = df_actions.copy()
        df_action_no_scaler = df_action_no_scaler.rename(
            columns={a: f"{a}_ns" for a in actions})

        df_daily_vectors = df_daily_vectors.reindex(df_actions.index, method='ffill').fillna(0)
        df_finale = df_actions.join(df_daily_vectors, how="left")

        scaler = StandardScaler()
        price_columns = [col for col in actions if col in df_finale.columns]
        df_finale[price_columns] = scaler.fit_transform(df_finale[price_columns])

        for col in price_columns:
            df_finale[f'rendement_{col}'] = df_finale[col].pct_change()
            df_finale[f'mm7_{col}'] = df_finale[f'rendement_{col}'].rolling(7).mean()
            df_finale[f'mm30_{col}'] = df_finale[f'rendement_{col}'].rolling(30).mean()
            df_finale[f'std7_{col}'] = df_finale[f'rendement_{col}'].rolling(7).std()

        df_finale = pd.concat([df_action_no_scaler, df_finale], axis=1)
        df_finale = df_finale.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        state["df_finale"] = df_finale
        state["status"] = "data_ready"
        state["last_updated"] = datetime.utcnow().isoformat()
    except Exception as e:
        state["status"] = f"error_scrape: {e}"


# ─── 2. Entraînement PPO ──────────────────────────────────────────────────────

@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if state["df_finale"] is None:
        raise HTTPException(400, "Données non disponibles. Appelez d'abord /scrape-news.")
    state["status"] = "training"
    background_tasks.add_task(_run_train, req)
    return {"message": "Entraînement PPO lancé en arrière-plan", "config": req.dict()}


def _run_train(req: TrainRequest):
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        df_raw = state["df_finale"].iloc[60:].copy()
        act = req.assets
        act_ns = [f"{a}_ns" for a in act]

        dfs = []
        for asset, asset_ns in zip(act, act_ns):
            cols_obs = (
                [asset]
                + [str(k) for k in range(384)]
                + [f"rendement_{asset}", f"mm7_{asset}", f"mm30_{asset}", f"std7_{asset}"]
            )
            # Garder uniquement les colonnes présentes
            cols_obs = [c for c in cols_obs if c in df_raw.columns]
            temp_df = df_raw[cols_obs].copy()
            if asset_ns in df_raw.columns:
                temp_df['PRICE_REAL'] = df_raw[asset_ns].values
            else:
                temp_df['PRICE_REAL'] = df_raw[asset].values
            dfs.append(temp_df)

        class TradingEngine(gym.Env):
            def __init__(self, df, initial_capital=10_000,
                         transaction_cost=0.001, window_size=20):
                super().__init__()
                self.df = df.reset_index(drop=True)
                self.initial_capital = initial_capital
                self.transaction_cost = transaction_cost
                self.window_size = window_size
                self.features_cols = [c for c in self.df.columns if c != 'PRICE_REAL']
                self.nbr_features = len(self.features_cols)
                self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.window_size, self.nbr_features + 2),
                    dtype=np.float32)

            def _get_observation(self, step):
                row = self.df.iloc[step]
                features = row[self.features_cols].values.astype(np.float32)
                real_price = row['PRICE_REAL']
                portfolio_info = np.array([
                    self.balance / self.initial_capital,
                    (self.holdings * real_price) / self.initial_capital
                ], dtype=np.float32)
                return np.concatenate([features, portfolio_info])

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.balance = self.initial_capital
                self.holdings = 0
                self.current_step = self.window_size
                self.history = deque(maxlen=self.window_size)
                for i in range(self.window_size):
                    self.history.append(self._get_observation(i))
                return np.array(self.history, dtype=np.float32), {}

            def step(self, action):
                action_val = np.clip(action[0], 0, 1)
                current_price = self.df.iloc[self.current_step]['PRICE_REAL']
                net_worth = self.balance + self.holdings * current_price
                target_val = net_worth * action_val
                diff = target_val - self.holdings * current_price
                costs = abs(diff) * self.transaction_cost
                self.holdings = target_val / current_price
                self.balance = net_worth - target_val - costs
                self.current_step += 1
                done = self.current_step >= len(self.df) - 1
                next_price = self.df.iloc[self.current_step]['PRICE_REAL']
                nw_after = self.balance + self.holdings * next_price
                reward = np.clip((nw_after - net_worth)/net_worth -
                                 (next_price - current_price)/current_price, -10, 10)
                self.history.append(self._get_observation(self.current_step))
                return np.array(self.history, dtype=np.float32), reward, done, False, {}

        def evaluate_model(model, df_test, ws):
            env = TradingEngine(df_test, window_size=ws)
            obs, _ = env.reset()
            done = False
            history_values = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                idx = max(0, env.current_step - 1)
                price = env.df.iloc[idx]['PRICE_REAL']
                history_values.append(env.balance + env.holdings * price)
            return np.array(history_values)

        best_histories = {}
        for i, (df, asset) in enumerate(zip(dfs, act)):
            key = f"action_{i+1}"
            df_train = df.iloc[:req.train_size]
            df_test = df.iloc[req.train_size:-1]
            train_env = DummyVecEnv([lambda df=df_train: TradingEngine(df, window_size=req.window_size)])
            model = PPO("MlpPolicy", train_env,
                        learning_rate=req.learning_rate, n_steps=1024,
                        batch_size=64, gamma=0.98, ent_coef=0.01,
                        verbose=0, seed=42)
            model.learn(total_timesteps=req.n_timesteps)
            best_histories[key] = evaluate_model(model, df_test, req.window_size).tolist()

        state["best_histories"] = best_histories
        state["assets"] = act
        state["status"] = "trained"
        state["last_updated"] = datetime.utcnow().isoformat()

        # Métriques individuelles
        metrics = {}
        for i, asset in enumerate(act):
            key = f"action_{i+1}"
            vals = np.array(best_histories[key])
            gain_pct = (vals[-1] - 10_000) / 10_000 * 100
            max_dd = float((np.maximum.accumulate(vals) - vals).max())
            sharpe = float(np.mean(np.diff(vals)) / (np.std(np.diff(vals)) + 1e-9) * np.sqrt(252))
            metrics[asset] = {
                "capital_final": round(float(vals[-1]), 2),
                "gain_pct": round(float(gain_pct), 2),
                "max_drawdown": round(max_dd, 2),
                "sharpe_approx": round(sharpe, 3),
            }
        state["metrics"] = metrics
    except Exception as e:
        state["status"] = f"error_train: {e}"


# ─── 3. Optimisation de portefeuille ─────────────────────────────────────────

@app.post("/optimize-portfolio")
def optimize_portfolio(req: OptimizeRequest):
    if not state.get("best_histories"):
        raise HTTPException(400, "Modèles non entraînés. Appelez d'abord /train.")

    try:
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import risk_models, expected_returns, objective_functions

        assets = state.get("assets", ["AAPL", "NVDA", "BTC-USD"])
        df_prices = pd.DataFrame({
            asset: state["best_histories"][f"action_{i+1}"]
            for i, asset in enumerate(assets)
        })

        mu = expected_returns.mean_historical_return(df_prices)
        S = risk_models.sample_cov(df_prices)
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=req.gamma_l2)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance()

        returns = df_prices.pct_change().dropna()
        portfolio_returns = (returns * np.array(list(cleaned_weights.values()))).sum(axis=1)
        var_95 = float(np.percentile(portfolio_returns, 5))
        cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())

        result = {
            "poids_optimises": {k: round(v, 4) for k, v in cleaned_weights.items()},
            "performance": {
                "rendement_annuel_attendu": round(perf[0], 4),
                "volatilite_annuelle": round(perf[1], 4),
                "ratio_sharpe": round(perf[2], 4),
            },
            "risque": {
                "VaR_95": round(var_95, 4),
                "CVaR_95": round(cvar_95, 4),
            },
        }
        state["cleaned_weights"] = result["poids_optimises"]
        state["portfolio_performance"] = result["performance"]
        state["status"] = "optimized"
        state["last_updated"] = datetime.utcnow().isoformat()
        return result

    except Exception as e:
        raise HTTPException(500, f"Erreur optimisation : {e}")


# ─── 4. Résultats ─────────────────────────────────────────────────────────────

@app.get("/results")
def get_results():
    if state["status"] == "idle":
        return {"message": "Aucune donnée disponible. Lancez /scrape-news puis /train."}
    return {
        "status": state["status"],
        "last_updated": state["last_updated"],
        "metrics_par_actif": state.get("metrics", {}),
        "poids_portefeuille": state.get("cleaned_weights", {}),
        "performance_portefeuille": state.get("portfolio_performance", {}),
    }
