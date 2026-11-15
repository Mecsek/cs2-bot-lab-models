"""Compare classifiers for predicting ct_won with match-grouped cross-validation."""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cs2_bot_lab_models.training import SEED, load_training_data

MATCH_FRACTION = 0.25  # share of matches used for comparison; 1.0 = all
N_SPLITS = 5

MODELS = {
    "dummy (base rate)": DummyClassifier(strategy="prior"),
    "logistic regression": make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ),
    "random forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED
    ),
    "hist gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    "lightgbm": LGBMClassifier(
        n_estimators=500, learning_rate=0.05, verbose=-1, random_state=SEED
    ),
    "xgboost": XGBClassifier(
        n_estimators=500, learning_rate=0.05, tree_method="hist", random_state=SEED
    ),
}

SCORING = {
    "roc_auc": "roc_auc",
    "log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
}


def main() -> None:
    X, y, groups = load_training_data(MATCH_FRACTION, min_matches=N_SPLITS)

    cv = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for name, model in MODELS.items():
        print(f"evaluating {name} ...", flush=True)
        res = cross_validate(model, X, y, groups=groups, cv=cv, scoring=SCORING)
        rows.append(
            {
                "model": name,
                "roc_auc": res["test_roc_auc"].mean(),
                "log_loss": -res["test_log_loss"].mean(),
                "brier": -res["test_brier"].mean(),
                "fit_seconds": res["fit_time"].mean(),
            }
        )

    results = pd.DataFrame(rows).sort_values("log_loss")
    print("\n" + results.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
