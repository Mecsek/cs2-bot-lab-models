"""Exact match-outcome probabilities from per-round win probabilities.

Match rules (verified on all 816 matches in the data):
- rounds 1-12 are the first half; sides switch from round 13
- the first team to 13 round wins takes the match
- there is no overtime: 12-12 after 24 rounds is a draw
"""

from functools import lru_cache

ROUNDS_PER_HALF = 12
ROUNDS_TO_WIN = 13
MAX_ROUNDS = 2 * ROUNDS_PER_HALF


def other_side(side: str) -> str:
    return "T" if side == "CT" else "CT"


def side_in_round(start_side: str, round_num: int) -> str:
    """A team's side in a given round, from its first-half side."""
    return start_side if round_num <= ROUNDS_PER_HALF else other_side(start_side)


def match_outcome_probs(
    score_a: int,
    score_b: int,
    next_round: int,
    a_side: str,
    p_a_ct: float,
    p_a_t: float,
    p_current: float | None = None,
) -> tuple[float, float, float]:
    """P(A wins, draw, B wins) of the match.

    score_a, score_b: rounds won so far; next_round: the round about to be played (1-24);
    a_side: A's side ("CT" or "T") in next_round;
    p_a_ct, p_a_t: P(A wins a round) when A is CT / T, used for future rounds;
    p_current: P(A wins next_round) if known (round-start or live model), else from the side.
    """
    if a_side not in ("CT", "T"):
        raise ValueError(f"a_side must be 'CT' or 'T', got {a_side!r}")
    if score_a + score_b != next_round - 1:
        raise ValueError(
            f"score {score_a}-{score_b} does not fit next_round={next_round} "
            f"(expected {score_a + score_b + 1})"
        )
    if (
        score_a >= ROUNDS_TO_WIN
        or score_b >= ROUNDS_TO_WIN
        or next_round > MAX_ROUNDS + 1
    ):
        raise ValueError("the match is already over")

    a_start = a_side if next_round <= ROUNDS_PER_HALF else other_side(a_side)

    @lru_cache(maxsize=None)
    def solve(sa: int, sb: int, rnd: int) -> tuple[float, float, float]:
        if sa == ROUNDS_TO_WIN:
            return (1.0, 0.0, 0.0)
        if sb == ROUNDS_TO_WIN:
            return (0.0, 0.0, 1.0)
        if rnd > MAX_ROUNDS:
            return (0.0, 1.0, 0.0)
        if rnd == next_round and p_current is not None:
            p = p_current
        else:
            p = p_a_ct if side_in_round(a_start, rnd) == "CT" else p_a_t
        win_a = solve(sa + 1, sb, rnd + 1)
        win_b = solve(sa, sb + 1, rnd + 1)
        return tuple(p * x + (1 - p) * y for x, y in zip(win_a, win_b))

    return solve(score_a, score_b, next_round)
