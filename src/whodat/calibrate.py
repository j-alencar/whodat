"""Fits thresholds on the `calibrate` videos."""

from whodat.match import Thresholds

Matches = dict[tuple[float, str, str], float]
"""Seconds keyed by top score, top-scoring person, and labeled identity."""


def _passing(matches: Matches, max_error: float, kept, wrong) -> list[float]:
    """Returns thresholds with at most `max_error` of kept time marked `wrong`."""
    passing = []
    for t in sorted({score for score, _, _ in matches}):
        total = bad = 0.0
        for (score, hint, ref), seconds in matches.items():
            if kept(score, t):
                total += seconds
                bad += seconds * wrong(hint, ref)
        if total and bad / total <= max_error:
            passing.append(t)
    return passing


def fit_thresholds(matches: Matches, max_error: float = 0.05) -> Thresholds:
    """Fits thresholds to cap accept and reject error at `max_error` when possible.

    `high` is the lowest score at or above which the wrong person accounts for
    at most `max_error` of accepted time. `low` starts at the highest score
    below which the right person accounts for at most `max_error` of rejected time.
    If no score qualifies, `high` falls back to the maximum score plus `1e-6`
    and `low` to the minimum score. The final `low` is capped at `high`.
    """
    accepting = _passing(matches, max_error, lambda s, t: s >= t, lambda h, r: h != r)
    rejecting = _passing(matches, max_error, lambda s, t: s < t, lambda h, r: h == r)
    high = min(accepting) if accepting else max(s for s, _, _ in matches) + 1e-6
    low = max(rejecting) if rejecting else min(s for s, _, _ in matches)
    return Thresholds(high=high, low=min(low, high))
