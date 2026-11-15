# cs2-bot-lab-models

<!--toc:start-->
- [cs2-bot-lab-models](#cs2-bot-lab-models)
  - [Stack](#stack)
  - [Setup](#setup)
  - [Data](#data)
  - [Project layout](#project-layout)
  - [Loading data](#loading-data)
  - [Example: train and save a model](#example-train-and-save-a-model)
  - [Comparing models](#comparing-models)
  - [Tuning](#tuning)
  - [Round and match predictions](#round-and-match-predictions)
<!--toc:end-->

Machine-learning models on CS2 data stored as Parquet files.

## Stack

| Library | Purpose |
|---|---|
| pandas + pyarrow | Read Parquet data into DataFrames that models train on |
| scikit-learn | Classic models, preprocessing, train/test splits, cross-validation, metrics |
| lightgbm / xgboost | Gradient boosting for tabular data |
| optuna | Hyperparameter tuning |
| joblib | Save and load trained models |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Data

Each table is a folder under `data/raw/parquet/`, split by map. Git ignores Parquet files.

```
data/raw/parquet/
├── rounds/map=de_dust2/*.parquet      ← one row per round
├── round_state_samples/map=.../       ← snapshots every 32 ticks (the training data)
├── ticks/  kills/  damage/  grenades/  bomb_events/
models/                                ← trained models, scores, Optuna study
```

## Project layout

| Path | What it does |
|---|---|
| `src/cs2_bot_lab_models/data.py` | Load Parquet tables (`load`, `load_all`, `list_tables`) |
| `src/cs2_bot_lab_models/training.py` | Training-data preparation and model definitions shared by the scripts |
| `src/cs2_bot_lab_models/match.py` | Match calculator: P(win / draw / loss) from round win probabilities |
| `src/cs2_bot_lab_models/predict.py` | `predict_round()` / `predict_match()` using the saved models |
| `scripts/model_selection.py` | Compare 6 classifiers on the in-round data |
| `scripts/tune.py` | Tune LightGBM with Optuna → live model |
| `scripts/train_round_models.py` | Train the matchup and round-start models |
| `scripts/evaluate_match.py` | Score match predictions on held-out matches |

Typical order: `model_selection.py` → `tune.py` → `train_round_models.py` → `evaluate_match.py` → use `predict.py`.

## Loading data

Each table is a folder under `data/raw/parquet/`, split into `map=<name>` subfolders.

```python
import pyarrow.dataset as ds

from cs2_bot_lab_models.data import list_tables, load, load_all

list_tables()            # ['bomb_events', 'damage', 'grenades', 'kills', 'round_state_samples', 'rounds', 'ticks']

# Whole table in memory, ready for scikit-learn
df = load_all("rounds")
df = load_all("rounds", columns=["match_id", "round", "winner_side"])
df = load_all("rounds", filter=ds.field("map") == "de_dust2")

# Large tables (e.g. ticks, ~117M rows): read in batches instead
for batch in load("ticks", batch_size=100_000, columns=["match_id", "tick", "hp"]):
    ...
```

`load()` and `load_all()` take the same `columns` and `filter` arguments.
`load()` yields pandas DataFrames with Arrow-backed column types; `load_all()` returns standard NumPy types.

## Example: train and save a model

```python
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from cs2_bot_lab_models.data import MODELS_DIR
from cs2_bot_lab_models.training import load_training_data

X, y, groups = load_training_data(match_fraction=0.25)   # round_state_samples → features, ct_won, match_id

# Split by match, so near-identical snapshots of one match never land on both sides
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(X, y, groups))

model = LGBMClassifier(verbose=-1).fit(X.iloc[train_idx], y.iloc[train_idx])
print("test log loss:", log_loss(y.iloc[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1]))

joblib.dump(model, MODELS_DIR / "lgbm_ct_won.joblib")
```

Run your scripts with `uv run python your_script.py`.

## Comparing models

```bash
uv run python scripts/model_selection.py
```

Compares a base-rate dummy, logistic regression, random forest, hist gradient boosting, LightGBM and XGBoost on predicting `ct_won` from in-round snapshots. It uses 25% of matches and 5-fold cross-validation split by `match_id`, so snapshots of one match never land on both sides. The results table is sorted by log loss (lower is better). Set `MATCH_FRACTION` in the script to `1.0` to use all matches.

## Tuning

Tune LightGBM for predicting `ct_won` with Optuna:

```bash
uv run python scripts/tune.py --trials 50 --match-fraction 0.25
```

- 20% of matches are held out as a test set that tuning never sees.
- Each trial is scored by match-grouped 3-fold cross-validated log loss. Clearly bad trials are stopped early.
- The study is saved in `models/optuna.db`. Running again with the same `--study-name` adds trials to it.
- The best model is retrained, compared with default LightGBM on the test set, and saved as `models/lgbm_tuned.joblib`. Its settings and scores go in `models/lgbm_tuned.json`.

## Round and match predictions

How likely team A is to win a round, and the match (win / draw / loss), against team B.
Matches follow the rules in the data: sides switch after round 12, first to 13 wins, 12–12 is a draw.

| Model | Answers | File |
|---|---|---|
| Matchup | P(A wins a round) as CT and as T on a map, before anything else is known | `models/matchup.joblib` |
| Round start | P(A wins this round) from the first snapshot of the round (freeze end, after buying) | `models/round_start.joblib` |
| Live | P(A wins this round) from any snapshot during the round | `models/lgbm_tuned.joblib` |
| Match calculator | P(A wins / draw / B wins) from any score, computed exactly over the remaining rounds | `src/cs2_bot_lab_models/match.py` |

### 1. Train the models

```bash
uv run python scripts/train_round_models.py         # matchup + round start (a few minutes)
uv run python scripts/tune.py --trials 50           # live model (only needed for live=True)
uv run python scripts/evaluate_match.py             # optional: score match predictions on held-out matches
```

`train_round_models.py` prints cross-validated scores and saves them in `models/round_models.json`.

### 2. Predict a match

```python
from cs2_bot_lab_models.predict import predict_match

# Before the match; A starts on CT
predict_match("SNIPER_NEST", "ROOKIES", "de_nuke")
# {'a_wins': 0.569, 'draw': 0.155, 'b_wins': 0.276}

# During the match: A leads 8-5 and is on T for round 14
predict_match("SNIPER_NEST", "ROOKIES", "de_nuke", score_a=8, score_b=5, next_round=14, a_side="T")
# {'a_wins': 0.958, 'draw': 0.032, 'b_wins': 0.01}
```

| Argument | Meaning |
|---|---|
| `team_a`, `team_b` | Team names as in the data, e.g. `SNIPER_NEST`, `ROOKIES` |
| `map_name` | e.g. `de_nuke`, `de_dust2` |
| `score_a`, `score_b` | Rounds won so far (must add up to `next_round - 1`) |
| `next_round` | The round about to be played, 1–24 |
| `a_side` | A's side (`"CT"` or `"T"`) **in `next_round`**. The sides switch from round 13 on |
| `state`, `live` | Optional snapshot for the current round; see below |

### 3. Predict a round

```python
from cs2_bot_lab_models.predict import predict_round

predict_round("SNIPER_NEST", "ROOKIES", "de_nuke", a_side="CT")   # 0.414, from map and teams only
```

### 4. Sharpen with a snapshot

A snapshot is one `round_state_samples` row (a dict or one-row DataFrame). Use the **first** snapshot of a round for the round-start model, or any later one with `live=True`:

```python
import pyarrow.dataset as ds
from cs2_bot_lab_models.data import load_all
from cs2_bot_lab_models.predict import predict_match, predict_round

round_snaps = load_all(
    "round_state_samples",
    filter=(ds.field("match_id") == "0758eafd-f574-4910-948c-56a00e704b9d") & (ds.field("round") == 14),
).sort_values("tick")

start, mid = round_snaps.head(1), round_snaps.iloc[[len(round_snaps) // 2]]

predict_round("RUSHERS", "SLOW_ELITE", "de_nuke", a_side="CT", state=start)             # 0.347
predict_round("RUSHERS", "SLOW_ELITE", "de_nuke", a_side="CT", state=mid, live=True)  # 0.327

# The match, with the current round sharpened by the snapshot (RUSHERS lead 9-4)
predict_match("RUSHERS", "SLOW_ELITE", "de_nuke", score_a=9, score_b=4, next_round=14, a_side="CT", state=start)
```

### Notes
- In this data, which team is playing barely changes round outcomes: the matchup model scores no better than map alone. Side (T wins 72% of rounds), map, economy and score matter far more, so predictions are least certain before the match and sharpen as it goes.
- The match calculator treats rounds as independent, so it may predict somewhat more draws than happen (about 18% vs 13.6% in the data). `scripts/evaluate_match.py` shows how much this matters.
- Many calls in a row? Create one `Predictor()` from `cs2_bot_lab_models.predict` and reuse it. The module-level functions already reuse one behind the scenes.
