"""Labels accepted and rejected matches automatically when enabled.
Clusters that need review wait for a human answer.
"""

from dataclasses import dataclass, field

from whodat.labels import REVIEW, Label, Region
from whodat.match import Prediction


@dataclass
class Pending:
    """Holds the turns in a cluster that needs a human answer."""

    cluster: str
    hint: str
    score: float
    turns: list[Region] = field(default_factory=list)

    def answer(self, identity: str) -> list[Label]:
        return [Label(t.start, t.end, identity) for t in self.turns]


def _clip(start: float, end: float, region: Region) -> Region | None:
    start, end = max(start, region.start), min(end, region.end)
    return Region(start, end) if end > start else None


def auto_label(
    region: Region, predictions: list[Prediction], accept: bool = True
) -> tuple[list[Label], dict[str, Pending]]:
    """Labels accepted turns with a name and rejected turns as `unknown`.

    Turns marked `review` wait for a human answer by cluster. With
    `accept=False`, every cluster in the region goes to a human so held-out
    videos are graded against human labels.
    """
    labels, pending = [], {}
    for p in predictions:
        span = _clip(p.turn.start, p.turn.end, region)
        if span is None:
            continue
        if not accept or p.decision.identity == REVIEW:
            entry = pending.setdefault(
                p.turn.cluster,
                Pending(p.turn.cluster, p.decision.hint, p.decision.score),
            )
            entry.turns.append(span)
        else:
            labels.append(Label(span.start, span.end, p.decision.identity))
    return labels, pending
