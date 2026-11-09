# cs2-bot-lab-models

<!--toc:start-->
- [cs2-bot-lab-models](#cs2-bot-lab-models)
  - [Stack](#stack)
  - [Setup](#setup)
  - [Data](#data)
  - [Loading data](#loading-data)
  - [Example: train and save a model](#example-train-and-save-a-model)
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

Put your `.parquet` files in `data/raw/`. Subfolders are fine.

```
data/raw/      ← input .parquet files
models/        ← trained models
```

## Loading data

```python
from cs2_bot_lab_models.data import load, list_datasets, MODELS_DIR

list_datasets()          # all .parquet files under data/raw
df = load("rounds")      # loads data/raw/rounds.parquet
df = load()              # loads every file in data/raw, combined
```

## Example: train and save a model

```python
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from cs2_bot_lab_models.data import load, MODELS_DIR

df = load("rounds")
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier().fit(X_train, y_train)
print("accuracy:", model.score(X_test, y_test))

joblib.dump(model, MODELS_DIR / "lgbm_rounds.joblib")
```

Run your scripts with `uv run python your_script.py`.
