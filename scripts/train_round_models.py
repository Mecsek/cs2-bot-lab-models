"""Train the matchup and round-start models on all matches and save them to models/.

- matchup model:     P(CT wins a round) from map + CT team + T team (rounds table)
- round-start model: P(CT wins a round) from the first snapshot of the round (freeze end)

Scores are match-grouped 5-fold cross-validated, against base-rate and map-only baselines.
"""

import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from cs2_bot_lab_models.data import MODELS_DIR
from cs2_bot_lab_models.predict import MATCHUP_FILE, ROUND_START_FILE
from cs2_bot_lab_models.training import (
    MATCHUP_FEATURES,
    TARGET,
    fit_matchup_model,
    load_round_results,
    load_round_start_data,
    make_round_start_model,
    round_features,
)

N_SPLITS = 5


def scores(y, p) -> dict:
    return {
        "log_loss": float(log_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.ptp(p) > 0 else 0.5,
        "brier": float(brier_score_loss(y, p)),
    }


def cv_proba(model, X, y, groups):
    return cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=GroupKFold(n_splits=N_SPLITS),
        method="predict_proba",
    )[:, 1]


def main() -> None:
    results = {}

    # Baselines and matchup model on the rounds table.
    rounds = load_round_results()
    y, groups = rounds[TARGET].astype(int), rounds["match_id"]
    print(
        f"{len(rounds):,} rounds, {groups.nunique()} matches, CT win rate {y.mean():.3f}\n"
    )

    results["base rate"] = scores(y, np.full(len(y), y.mean()))
    map_only = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=1000)
    )
    results["map only"] = scores(y, cv_proba(map_only, rounds[["map"]], y, groups))

    print("fitting matchup model ...", flush=True)
    matchup, search = fit_matchup_model(rounds[MATCHUP_FEATURES], y, groups, N_SPLITS)
    best_c = search.best_params_["logisticregression__C"]
    results["matchup (map + teams)"] = scores(
        y, cv_proba(search.best_estimator_, rounds[MATCHUP_FEATURES], y, groups)
    )

    # Round-start model on the first snapshot of each round.
    print("fitting round-start model ...", flush=True)
    first = load_round_start_data()
    X_first, y_first, g_first = (
        round_features(first),
        first[TARGET].astype(int),
        first["match_id"],
    )
    round_start = make_round_start_model()
    results["round start (freeze end)"] = scores(
        y_first, cv_proba(round_start, X_first, y_first, g_first)
    )
    round_start.fit(X_first, y_first)

    print(f"\nmatchup regularization C = {best_c}")
    print(f"\n{'model':<26} {'log_loss':>9} {'roc_auc':>8} {'brier':>7}")
    for name, s in results.items():
        print(
            f"{name:<26} {s['log_loss']:>9.4f} {s['roc_auc']:>8.4f} {s['brier']:>7.4f}"
        )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(matchup, MODELS_DIR / MATCHUP_FILE)
    joblib.dump(round_start, MODELS_DIR / ROUND_START_FILE)
    (MODELS_DIR / "round_models.json").write_text(
        json.dumps(
            {"matchup_C": best_c, "cv_folds": N_SPLITS, "cv_scores": results}, indent=2
        )
    )
    print(
        f"\nsaved {MATCHUP_FILE}, {ROUND_START_FILE} and round_models.json in {MODELS_DIR}"
    )


if __name__ == "__main__":
    main()
