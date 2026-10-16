import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

folder = os.path.dirname(os.path.abspath(__file__))

# To add a league: make a folder for it and add one line here.
# Each folder must hold that league's features_full.csv and team_state.csv.
LEAGUES = {
    "Premier League": "leagues/epl",
    "La Liga": "leagues/laliga",
}

goals = np.arange(0, 11)
total = np.add.outer(goals, goals)
lines = [1.5, 2.5, 3.5, 4.5]
loaded = {}

for name, sub in LEAGUES.items():
    path = os.path.join(folder, sub)
    f_features = os.path.join(path, "features_full.csv")
    f_state = os.path.join(path, "team_state.csv")
    if not (os.path.exists(f_features) and os.path.exists(f_state)):
        print("Skipping", name, "- files not found in", path)
        continue

    data = pd.read_csv(f_features, parse_dates=["Date"])
    state = pd.read_csv(f_state, parse_dates=["LastMatch"]).set_index("Team")

    # keeps roughly the last 3 seasons plus the current one
    train = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=1170)]
    cols = [c for c in data.columns
            if c.startswith("Home_") or c.startswith("Away_") or c == "Elo_Diff"]

    home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
    away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
    home_model.fit(train[cols], train["FTHG"])
    away_model.fit(train[cols], train["FTAG"])

    loaded[name] = {"state": state, "cols": cols, "home": home_model, "away": away_model}
    print(name, "trained on", len(train), "matches")

app = FastAPI(title="Eyconic AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def get_league(league):
    if league not in loaded:
        raise HTTPException(status_code=404, detail="League not available yet.")
    return loaded[league]


def build_row(lg, home, away):
    state = lg["state"]
    row = {}
    for c in lg["cols"]:
        if c == "Elo_Diff":
            row[c] = state.loc[home, "Elo"] - state.loc[away, "Elo"]
        elif c.startswith("Home_"):
            row[c] = state.loc[home, c[5:]]
        else:
            row[c] = state.loc[away, c[5:]]
    return pd.DataFrame([row])[lg["cols"]]


@app.get("/leagues")
def get_leagues():
    return list(loaded.keys())


@app.get("/teams")
def get_teams(league: str):
    state = get_league(league)["state"]
    latest = state["LastMatch"].max()
    current = state[state["LastMatch"] >= latest - pd.Timedelta(days=30)]
    return sorted(current.index.tolist())


@app.get("/predict")
def get_prediction(league: str, home: str, away: str):
    lg = get_league(league)
    state = lg["state"]
    if home not in state.index or away not in state.index:
        raise HTTPException(status_code=404, detail="Team not found. Check the spelling.")
    if home == away:
        raise HTTPException(status_code=400, detail="Pick two different teams.")
    row = build_row(lg, home, away)
    if row.isna().values.any():
        raise HTTPException(status_code=422,
                            detail="Not enough matches yet for one of these teams.")

    h_xg = float(lg["home"].predict(row)[0])
    a_xg = float(lg["away"].predict(row)[0])
    grid = np.outer(poisson.pmf(goals, h_xg), poisson.pmf(goals, a_xg))
    grid = grid / grid.sum()
    best = np.unravel_index(grid.argmax(), grid.shape)

    return {
        "home_xg": h_xg,
        "away_xg": a_xg,
        "p_home": float(np.tril(grid, -1).sum()),
        "p_draw": float(np.trace(grid)),
        "p_away": float(np.triu(grid, 1).sum()),
        "over": {str(line): float(grid[total > line].sum()) for line in lines},
        "best_score": [int(best[0]), int(best[1])],
    }
