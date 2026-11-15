"""Round and match predictions for team A vs team B, using the saved models.

Train the models first: `uv run python scripts/train_round_models.py`
(the live model is models/lgbm_tuned.joblib from scripts/tune.py).

`state` is one round_state_samples row (dict or one-row DataFrame):
- the round's first snapshot (freeze end)  -> round-start model
- any snapshot during the round, live=True -> live model
"""

from functools import cache
from pathlib import Path

import joblib
import pandas as pd

from cs2_bot_lab_models.data import MODELS_DIR
from cs2_bot_lab_models.match import match_outcome_probs
from cs2_bot_lab_models.training import MATCHUP_FEATURES, round_features

MATCHUP_FILE = "matchup.joblib"
ROUND_START_FILE = "round_start.joblib"
LIVE_FILE = "lgbm_tuned.joblib"


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.matchup = joblib.load(models_dir / MATCHUP_FILE)
        self.round_start = joblib.load(models_dir / ROUND_START_FILE)
        live_path = models_dir / LIVE_FILE
        self.live = joblib.load(live_path) if live_path.exists() else None

    def _p_ct_matchup(self, map_name: str, team_ct: str, team_t: str) -> float:
        row = pd.DataFrame([[map_name, team_ct, team_t]], columns=MATCHUP_FEATURES)
        return float(self.matchup.predict_proba(row)[0, 1])

    def matchup_probs(
        self, team_a: str, team_b: str, map_name: str
    ) -> tuple[float, float]:
        """P(A wins a round) as CT and as T, before anything is known about the round."""
        p_a_ct = self._p_ct_matchup(map_name, team_a, team_b)
        p_a_t = 1.0 - self._p_ct_matchup(map_name, team_b, team_a)
        return p_a_ct, p_a_t

    def _p_ct_state(self, state: dict | pd.DataFrame, live: bool) -> float:
        model = self.live if live else self.round_start
        if model is None:
            raise FileNotFoundError(
                f"live model {LIVE_FILE} not found; run scripts/tune.py"
            )
        df = pd.DataFrame([state]) if isinstance(state, dict) else state
        X = round_features(df)[list(model.feature_names_in_)]
        return float(model.predict_proba(X)[0, 1])

    def predict_round(
        self,
        team_a: str,
        team_b: str,
        map_name: str,
        a_side: str,
        state: dict | pd.DataFrame | None = None,
        live: bool = False,
    ) -> float:
        """P(team A wins this round)."""
        if state is None:
            p_a_ct, p_a_t = self.matchup_probs(team_a, team_b, map_name)
            return p_a_ct if a_side == "CT" else p_a_t
        p_ct = self._p_ct_state(state, live)
        return p_ct if a_side == "CT" else 1.0 - p_ct

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        map_name: str,
        score_a: int = 0,
        score_b: int = 0,
        next_round: int = 1,
        a_side: str = "CT",
        state: dict | pd.DataFrame | None = None,
        live: bool = False,
    ) -> dict[str, float]:
        """P(A wins / draw / B wins) the match, from the score before next_round.

        a_side is A's side in next_round. If state is given, it sharpens the current round.
        """
        p_a_ct, p_a_t = self.matchup_probs(team_a, team_b, map_name)
        p_current = (
            self.predict_round(team_a, team_b, map_name, a_side, state, live)
            if state is not None
            else None
        )
        win, draw, loss = match_outcome_probs(
            score_a, score_b, next_round, a_side, p_a_ct, p_a_t, p_current
        )
        return {"a_wins": win, "draw": draw, "b_wins": loss}


@cache
def default_predictor() -> Predictor:
    return Predictor()


def predict_round(team_a, team_b, map_name, a_side, state=None, live=False) -> float:
    """P(team A wins this round), using the models in models/."""
    return default_predictor().predict_round(
        team_a, team_b, map_name, a_side, state, live
    )


def predict_match(
    team_a,
    team_b,
    map_name,
    score_a=0,
    score_b=0,
    next_round=1,
    a_side="CT",
    state=None,
    live=False,
) -> dict[str, float]:
    """P(A wins / draw / B wins) the match, using the models in models/."""
    return default_predictor().predict_match(
        team_a, team_b, map_name, score_a, score_b, next_round, a_side, state, live
    )
