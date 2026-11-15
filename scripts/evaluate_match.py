"""Evaluate match-outcome predictions on held-out matches.

20% of matches are held out; the matchup and round-start models are trained on the rest.
At the start of every round of every held-out match, P(A wins / draw / B wins) is
predicted (A = the team starting on CT) and scored against the real result.

Methods compared:
- score only:            score + side, with overall CT/T round win rates
- matchup:               score + side, with map/team round win rates (matchup model)
- matchup + round start: as above, plus the round-start model for the current round
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from cs2_bot_lab_models.match import match_outcome_probs
from cs2_bot_lab_models.training import (
    MATCHUP_FEATURES,
    SEED,
    TARGET,
    fit_matchup_model,
    load_round_results,
    load_round_start_data,
    make_round_start_model,
    round_features,
)

TEST_SIZE = 0.2
OUTCOMES = ["a_wins", "draw", "b_wins"]


def p_ct(
    model, map_name: pd.Series, team_ct: pd.Series, team_t: pd.Series
) -> np.ndarray:
    X = pd.DataFrame(
        {"map": map_name.values, "team_ct": team_ct.values, "team_t": team_t.values}
    )
    return model.predict_proba(X[MATCHUP_FEATURES])[:, 1]


def main() -> None:
    rounds = load_round_results()
    first = load_round_start_data()

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(
            rounds, groups=rounds["match_id"]
        )
    )
    train, test = rounds.iloc[train_idx], rounds.iloc[test_idx].copy()
    train_ids = set(train["match_id"])
    print(
        f"training on {len(train_ids)} matches, evaluating on {test['match_id'].nunique()}\n"
    )

    # Models trained on training matches only.
    matchup, _ = fit_matchup_model(
        train[MATCHUP_FEATURES], train[TARGET].astype(int), train["match_id"]
    )
    first_train = first[first["match_id"].isin(train_ids)]
    round_start = make_round_start_model().fit(
        round_features(first_train), first_train[TARGET].astype(int)
    )
    ct_rate = train[TARGET].mean()

    # Per-round inputs for the held-out matches. A = the team on CT in round 1.
    first_test = first[~first["match_id"].isin(train_ids)]
    start_p = pd.Series(
        round_start.predict_proba(round_features(first_test))[:, 1],
        index=pd.MultiIndex.from_frame(first_test[["match_id", "round"]]),
    )
    test["p_ct_start"] = start_p.reindex(
        pd.MultiIndex.from_frame(test[["match_id", "round"]])
    ).values

    matches = test[test["round"] == 1].set_index("match_id")
    team_a, team_b = matches["team_ct"], matches["team_t"]
    p_a_ct = pd.Series(
        p_ct(matchup, matches["map"], team_a, team_b), index=matches.index
    )
    p_a_t = pd.Series(
        1 - p_ct(matchup, matches["map"], team_b, team_a), index=matches.index
    )

    wins_a = test.groupby("match_id").apply(
        lambda d: (d["winner_team"] == team_a[d.name]).sum(), include_groups=False
    )
    wins_b = test.groupby("match_id").size() - wins_a
    outcome = np.select([wins_a > wins_b, wins_a == wins_b], [0, 1], default=2)
    outcome = pd.Series(outcome, index=wins_a.index)

    rows = []
    for r in test.itertuples(index=False):
        a_side = "CT" if r.round <= 12 else "T"
        score_a, score_b = (
            (r.score_ct, r.score_t) if a_side == "CT" else (r.score_t, r.score_ct)
        )
        p_now = r.p_ct_start if a_side == "CT" else 1 - r.p_ct_start
        args = (score_a, score_b, r.round, a_side)
        rows.append(
            {
                "match_id": r.match_id,
                "round": r.round,
                "outcome": outcome[r.match_id],
                "score only": match_outcome_probs(*args, ct_rate, 1 - ct_rate),
                "matchup": match_outcome_probs(
                    *args, p_a_ct[r.match_id], p_a_t[r.match_id]
                ),
                "matchup + round start": match_outcome_probs(
                    *args, p_a_ct[r.match_id], p_a_t[r.match_id], p_now
                ),
            }
        )
    preds = pd.DataFrame(rows)
    methods = ["score only", "matchup", "matchup + round start"]

    def ll(df: pd.DataFrame, method: str) -> float:
        return log_loss(df["outcome"], np.array(df[method].tolist()), labels=[0, 1, 2])

    moments = {
        "every round": preds,
        "before the match (round 1)": preds[preds["round"] == 1],
        "at half time (round 13)": preds[preds["round"] == 13],
        "late (round 19+)": preds[preds["round"] >= 19],
    }
    print("match outcome log loss (3 outcomes, lower is better):")
    print(f"  {'moment':<28}" + "".join(f"{m:>24}" for m in methods))
    for name, df in moments.items():
        print(f"  {name:<28}" + "".join(f"{ll(df, m):>24.4f}" for m in methods))

    base = outcome.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    print(
        "\nactual outcomes in held-out matches: "
        + ", ".join(f"{o} {base[i]:.1%}" for i, o in enumerate(OUTCOMES))
    )

    # Calibration of P(A wins) for the full method.
    p_win = preds["matchup + round start"].map(lambda t: t[0])
    calib = pd.DataFrame({"predicted": p_win, "actual": preds["outcome"] == 0})
    calib["bin"] = pd.cut(
        calib["predicted"], np.linspace(0, 1, 11), include_lowest=True
    )
    table = calib.groupby("bin", observed=True).agg(
        rows=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    )
    print("\ncalibration of P(A wins), matchup + round start (every round):")
    print(table.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
