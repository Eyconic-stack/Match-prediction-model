# ⚽ Eyconic AI: Football Match Prediction API

> Pick two teams. Get the **expected goals, win/draw/loss chances, over/under probabilities and the most likely score**, all from a machine learning model trained on past matches.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Leagues](https://img.shields.io/badge/Leagues-Premier%20League%20%7C%20La%20Liga-6C3BAA)

---

## 🎯 The idea

A football score is not random noise. Goals follow patterns: how strong each team is, how well they have been playing, and who has home advantage. Eyconic AI learns those patterns from past matches and turns them into probabilities for a match it has never seen.

---

## ✨ What you get from one request

For any fixture, the API returns:

| Output | Meaning |
|---|---|
| ⚽ `home_xg`, `away_xg` | Expected goals for each team |
| 🏠 `p_home` | Chance of a home win |
| 🤝 `p_draw` | Chance of a draw |
| ✈️ `p_away` | Chance of an away win |
| 📈 `over` | Chance of over 1.5, 2.5, 3.5 and 4.5 total goals |
| 🎯 `best_score` | The single most likely scoreline |

**Example request**

```text
GET /predict?league=Premier League&home=Arsenal&away=Chelsea
```

**Example response** (the shape is real, the numbers are illustrative)

```json
{
  "home_xg": 1.8,
  "away_xg": 1.1,
  "p_home": 0.52,
  "p_draw": 0.24,
  "p_away": 0.24,
  "over": {"1.5": 0.78, "2.5": 0.52, "3.5": 0.28, "4.5": 0.12},
  "best_score": [2, 1]
}
```

---

## 🧠 How it works

1. **Features.** Each team has a stored profile (Elo rating, recent form and more). For a fixture, the app builds one row of home-team features, away-team features and the **Elo difference** between them.
2. **Two models.** A **Poisson regression** (scikit-learn) predicts the home team's goals, and a second one predicts the away team's goals. Poisson regression is a natural fit because goals are small whole numbers that happen at some average rate.
3. **Scaling and regularisation.** Inputs go through a `StandardScaler`, and the models use regularisation (`alpha=1.0`) so they do not overfit.
4. **Recent data only.** Training uses roughly the last three seasons plus the current one, so older football does not drag the model backwards.
5. **A grid of scorelines.** From the two expected-goal values, the app uses the Poisson distribution (SciPy) to build a grid of probabilities for every score from 0-0 up to 10-10.
6. **Everything comes from that grid.** Add the cells where home goals are higher for `p_home`, the diagonal for `p_draw`, the cells where away goals are higher for `p_away`, sum by total goals for the over/under lines, and pick the biggest cell for `best_score`.

---

## 🔌 API endpoints

| Endpoint | What it does |
|---|---|
| `GET /leagues` | Lists the leagues that loaded successfully |
| `GET /teams?league=...` | Lists teams that played in the last 30 days of that league's data |
| `GET /predict?league=...&home=...&away=...` | Returns the prediction |

**Friendly errors**, so the website can show clear messages:

- `404` league not available, or team not found
- `400` the same team was picked twice
- `422` not enough match history for one of the teams

---

## 🌍 Adding a new league

The app is built to grow. To add a league:

1. Create a folder such as `leagues/bundesliga/` containing `features_full.csv` and `team_state.csv`
2. Add one line to the `LEAGUES` dictionary:

```python
LEAGUES = {
    "Premier League": "leagues/epl",
    "La Liga": "leagues/laliga",
    "Bundesliga": "leagues/bundesliga",
}
```

If a league's files are missing, the app skips it and keeps running.

---

## 🚀 Run it

```bash
git clone https://github.com/YOUR-USERNAME/eyconic-ai.git
cd eyconic-ai
python -m pip install fastapi uvicorn pandas numpy scipy scikit-learn
python -m uvicorn server:app --reload
```

> Replace `server` with the name of your Python file if it is different. Then open **http://127.0.0.1:8000/docs** to try every endpoint in your browser.

---

## 🗂️ Project structure

```text
eyconic-ai/
├── server.py                 # the API and the models
├── leagues/
│   ├── epl/
│   │   ├── features_full.csv
│   │   └── team_state.csv
│   └── laliga/
│       ├── features_full.csv
│       └── team_state.csv
└── README.md
```

---

## 📊 Model performance

> ✏️ **Add your own numbers here before sharing.** Interviewers will ask.

| | |
|---|---|
| **Data source** | [FILL IN: where the match data comes from] |
| **Matches trained on** | [FILL IN: printed when the server starts] |
| **Evaluation method** | [FILL IN: e.g. tested on the most recent season] |
| **Result** | [FILL IN: e.g. accuracy or log loss compared with a simple baseline] |

---

## 📚 What I learned building this

- Why **Poisson regression** suits count data like goals
- Turning two expected-goal numbers into a full probability table
- Serving a trained model through a **FastAPI** backend with clear error messages
- Designing code that scales to new leagues by changing one line
- Handling bad requests safely instead of crashing

---

## 🔭 What's next

- [ ] Build a web frontend where anyone can pick a fixture
- [ ] Add user accounts and login
- [ ] Add caching so repeated predictions are instant under heavy traffic
- [ ] Restrict CORS to my own website before going public
- [ ] Add more leagues

---

> ⚠️ Predictions are probabilities, not promises. This project is for learning and entertainment, not betting advice.

👤 **Godwin** · Lagos, Nigeria
🔗 [GitHub](https://github.com/Eyconic-stack) · [LinkedIn](https://linkedin.com/in/godwin-godfrey-289428355?utm_source=share_via&utm_content=profile&utm_medium=member_ios)
