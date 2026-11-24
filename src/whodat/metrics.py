"""Scores predictions against label files within scored regions.

The scores count fixed-length frames so score sheets can be added across
videos.
"""

from collections import Counter

import numpy as np

from whodat.labels import REVIEW, UNKNOWN, UNSURE, LabelFile
from whodat.match import Prediction

FRAME = 0.1
NOT_PERSON = (REVIEW, UNKNOWN)


def _covering(spans, times) -> list[list[int]]:
    """Returns the indices of the spans that cover each sampled time."""
    covering = [[] for _ in times]
    for index, (start, end) in enumerate(spans):
        lo, hi = np.searchsorted(times, [start, end])
        for t in range(lo, hi):
            covering[t].append(index)
    return covering


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


class ScoreSheet:
    def __init__(self) -> None:
        self.identification = Counter()
        """Frame counts by reference identity, predicted identity, and `hint`."""
        self.speech = Counter()
        """Frame counts for `speech`, `missed`, and `false_alarm`."""
        self._matches = Counter()

    def add(self, label_file: LabelFile, predictions: list[Prediction]) -> None:
        times = np.concatenate(
            [np.arange(r.start + FRAME / 2, r.end, FRAME) for r in label_file.scored]
            or [np.array([])]
        )
        refs = _covering([(lb.start, lb.end) for lb in label_file.labels], times)
        hyps = _covering([(p.turn.start, p.turn.end) for p in predictions], times)
        for ref, hyp in zip(refs, hyps):
            self.speech["speech"] += bool(ref)
            self.speech["missed"] += bool(ref) and not hyp
            self.speech["false_alarm"] += bool(hyp) and not ref
            if len(ref) == 1 and len(hyp) == 1:
                decision = predictions[hyp[0]].decision
                reference = label_file.labels[ref[0]].identity
                if reference == UNSURE:
                    continue
                self.identification[reference, decision.identity, decision.hint] += 1
                self._matches[round(decision.score, 3), decision.hint, reference] += 1

    @property
    def matches(self) -> dict[tuple[float, str, str], float]:
        """Reports seconds by top score, top-scoring person, and labeled identity.

        Calibration uses these totals.
        """
        return {key: round(n * FRAME, 3) for key, n in self._matches.items()}

    def confusion(self) -> dict[tuple[str, str], float]:
        """Reports seconds for each pair of labeled and predicted identities."""
        result = Counter()
        for (reference, identity, _), n in self.identification.items():
            result[reference, identity] += n * FRAME
        return dict(result)

    def _seconds(self, keep) -> float:
        return sum(
            seconds
            for (reference, identity, hint), seconds in self.identification.items()
            if keep(reference, identity, hint)
        )

    def report(self) -> dict:
        total = self._seconds(lambda r, i, h: True)
        accepted = self._seconds(lambda r, i, h: i != REVIEW)
        speech = self.speech["speech"]
        return {
            "missed_speech_rate": _ratio(self.speech["missed"], speech),
            "false_alarm_rate": _ratio(self.speech["false_alarm"], speech),
            "abstain_rate": _ratio(total - accepted, total),
            "accepted_accuracy": _ratio(
                self._seconds(lambda r, i, h: i != REVIEW and i == r), accepted
            ),
            "unknown_false_accept_rate": _ratio(
                self._seconds(lambda r, i, h: r == UNKNOWN and i not in NOT_PERSON),
                self._seconds(lambda r, i, h: r == UNKNOWN),
            ),
            "forced_accuracy": _ratio(
                self._seconds(lambda r, i, h: (h if i == REVIEW else i) == r), total
            ),
        }
