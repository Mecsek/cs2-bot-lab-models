"""Shared training-data preparation and model definitions for the round and match models."""

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from cs2_bot_lab_models.data import load_all

TABLE = "round_state_samples"
TARGET = "ct_won"
NON_FEATURES = ["match_id", "round", "tick", "team_ct", "team_t", "map"]
MATCHUP_FEATURES = ["map", "team_ct", "team_t"]
SEED = 42


def round_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from round_state_samples rows: everything except ids and the target."""
    return df.drop(columns=[TARGET, *NON_FEATURES], errors="ignore").astype("float32")


def sample_matches(fraction: float, min_matches: int = 5) -> list[str]:
    """A seeded random subset of match ids (fraction 1.0 = all matches)."""
    ids = load_all(TABLE, columns=["match_id"])["match_id"].unique()
    size = min(len(ids), max(min_matches, int(len(ids) * fraction)))
    rng = np.random.default_rng(SEED)
    return rng.choice(ids, size=size, replace=False).tolist()


def load_training_data(
    match_fraction: float = 1.0, min_matches: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features X, target y and match_id groups for a sample of matches (all snapshots)."""
    matches = sample_matches(match_fraction, min_matches)
    df = load_all(TABLE, filter=ds.field("match_id").isin(matches))

    X = round_features(df)
    y = df[TARGET].astype(int)
    groups = df["match_id"]
    print(
        f"{len(df):,} rows, {len(matches)} matches, {X.shape[1]} features, "
        f"CT win rate {y.mean():.3f}\n"
    )
    return X, y, groups


def load_round_start_data() -> pd.DataFrame:
    """The first snapshot of every round: taken at freeze end, after both teams bought.

    Uses head(1), not groupby().first(): first() takes the first non-empty value of
    each column separately and leaks later information (e.g. bomb_time_left).
    """
    df = load_all(TABLE)
    return (
        df.sort_values(["match_id", "round", "tick"])
        .groupby(["match_id", "round"])
        .head(1)
        .reset_index(drop=True)
    )


def load_round_results() -> pd.DataFrame:
    """The rounds table (one row per round, sorted) with a boolean ct_won column."""
    rounds = (
        load_all("rounds").sort_values(["match_id", "round"]).reset_index(drop=True)
    )
    rounds[TARGET] = rounds["winner_side"] == "CT"
    return rounds


def make_matchup_model(C: float = 0.1):
    """P(CT wins a round) from map + CT team + T team (one-hot, L2-regularized)."""
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(C=C, max_iter=1000),
    )


def fit_matchup_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5
):
    """Pick the regularization strength by match-grouped CV log loss, then fit on all data."""
    search = GridSearchCV(
        make_matchup_model(),
        {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=n_splits),
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search


def make_round_start_model() -> LGBMClassifier:
    """P(CT wins a round) from the round-start snapshot (economy, score, map, teams)."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        verbose=-1,
        random_state=SEED,
    )
