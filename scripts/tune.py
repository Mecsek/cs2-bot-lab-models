"""Tune LightGBM hyperparameters for predicting ct_won with Optuna.

- 20% of matches are held out as a final test set that tuning never sees.
- Each trial is scored by match-grouped 3-fold cross-validated log loss.
- Trials stop early (pruned) when clearly worse than the median trial.
- The study is saved in models/optuna.db, so re-running continues it.
"""

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from cs2_bot_lab_models.data import MODELS_DIR
from cs2_bot_lab_models.training import SEED, load_training_data

TEST_SIZE = 0.2
N_FOLDS = 3
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 100

FIXED_PARAMS = {
    "objective": "binary",
    "subsample_freq": 1,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        # high minimum: snapshots within a round are near-duplicates
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),  # feature_fraction
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # bagging_fraction
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),  # lambda_l1
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True
        ),  # lambda_l2
    }


def make_objective(X, y, groups):
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {**FIXED_PARAMS, **suggest_params(trial), "n_estimators": MAX_TREES}
        losses, best_iterations = [], []
        for step, (train_idx, valid_idx) in enumerate(folds):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            losses.append(log_loss(y_valid, model.predict_proba(X_valid)[:, 1]))
            best_iterations.append(int(model.best_iteration_))

            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_iterations", best_iterations)
        return float(np.mean(losses))

    return objective


def evaluate(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {
        "log_loss": float(log_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trials", type=int, default=50, help="trials to add in this run"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="stop after this many seconds"
    )
    parser.add_argument(
        "--match-fraction", type=float, default=0.25, help="share of matches to use"
    )
    parser.add_argument("--study-name", default="lgbm_ct_won")
    args = parser.parse_args()

    X, y, groups = load_training_data(args.match_fraction, min_matches=N_FOLDS + 2)

    # 1. Held-out test set (by match) that tuning never sees.
    tune_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(
            X, y, groups
        )
    )
    X_tune, y_tune, g_tune = X.iloc[tune_idx], y.iloc[tune_idx], groups.iloc[tune_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(
        f"tuning on {g_tune.nunique()} matches, testing on {groups.iloc[test_idx].nunique()}\n"
    )

    # 2-5. Optuna study, saved to SQLite so it can be resumed.
    MODELS_DIR.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{MODELS_DIR / 'optuna.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(
        make_objective(X_tune, y_tune, g_tune),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    best = study.best_trial
    n_trees = max(1, int(np.mean(best.user_attrs["best_iterations"])))
    print(f"\nbest trial #{best.number}: cv log loss {best.value:.4f}, {n_trees} trees")
    for name, value in best.params.items():
        print(f"  {name}: {value}")

    # 6. Retrain on all tuning matches, compare with default LightGBM on the test set.
    tuned = LGBMClassifier(**FIXED_PARAMS, **best.params, n_estimators=n_trees).fit(
        X_tune, y_tune
    )
    default = LGBMClassifier(verbose=-1, random_state=SEED).fit(X_tune, y_tune)
    scores = {
        "tuned": evaluate(tuned, X_test, y_test),
        "default": evaluate(default, X_test, y_test),
    }

    print("\ntest set (held-out matches):")
    print(f"  {'model':<8} {'log_loss':>9} {'roc_auc':>8} {'brier':>7}")
    for name, s in scores.items():
        print(
            f"  {name:<8} {s['log_loss']:>9.4f} {s['roc_auc']:>8.4f} {s['brier']:>7.4f}"
        )

    joblib.dump(tuned, MODELS_DIR / "lgbm_tuned.joblib")
    (MODELS_DIR / "lgbm_tuned.json").write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "best_trial": best.number,
                "trials_in_study": len(study.trials),
                "match_fraction": args.match_fraction,
                "params": {**FIXED_PARAMS, **best.params, "n_estimators": n_trees},
                "cv_log_loss": best.value,
                "test_scores": scores,
            },
            indent=2,
        )
    )
    print(f"\nsaved {MODELS_DIR / 'lgbm_tuned.joblib'} and lgbm_tuned.json")


if __name__ == "__main__":
    main()
