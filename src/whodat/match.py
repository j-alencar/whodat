"""Matches embeddings to voiceprints.

Score thresholds determine whether to accept, reject, or review a match.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from whodat.labels import REVIEW, UNKNOWN


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    cluster: str
    """The diarizer's anonymous voice, local to one window."""

    @property
    def duration(self) -> float:
        return self.end - self.start


STRATEGIES = ("cluster", "cluster+turn-override")


@dataclass(frozen=True)
class Thresholds:
    high: float
    low: float


@dataclass(frozen=True)
class Decision:
    identity: str
    score: float
    hint: str
    """The top-scoring person in any band, if there's a score."""


def decide(scores: dict[str, float], thresholds: Thresholds) -> Decision:
    hint = max(scores, key=scores.__getitem__)
    score = scores[hint]
    if score >= thresholds.high:
        identity = hint
    elif score < thresholds.low:
        identity = UNKNOWN
    else:
        identity = REVIEW
    return Decision(identity, score, hint)


@dataclass(frozen=True)
class Prediction:
    turn: Turn
    decision: Decision


def _unit(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def match_scores(embedding: np.ndarray, voiceprints: dict[str, np.ndarray]) -> dict:
    return {
        person: float(np.dot(_unit(embedding), _unit(print_)))
        for person, print_ in voiceprints.items()
    }


def identify(
    turns: list[Turn],
    exclusive: list[Turn],
    embeddings,
    voiceprints: dict,
    thresholds: Thresholds,
    strategy: str = "cluster",
) -> list[Prediction]:
    """Predicts an identity for each turn. Turns overlap where people talk at once.

    Identities come from `exclusive`, the turns of one voice at a time:
    crosstalk blurs an embedding. `embeddings[i]` belongs to `exclusive[i]`.
    A NaN row, such as audio too short to embed, counts for nothing. A turn
    with no embedded exclusive speech takes its cluster's identity, and a
    cluster with none is `unknown`.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown Match Strategy {strategy!r}")
    units = _unit(embeddings)
    embedded = ~np.isnan(units).any(axis=1)
    members = defaultdict(list)
    for index, piece in enumerate(exclusive):
        if embedded[index]:
            members[piece.cluster].append(index)

    def decide_mean(indices, weights) -> Decision:
        mean = (units[indices] * np.asarray(weights)[:, None]).sum(axis=0)
        return decide(match_scores(mean, voiceprints), thresholds)

    cluster_decision = defaultdict(lambda: Decision(UNKNOWN, 0.0, ""))
    for cluster, indices in members.items():
        cluster_decision[cluster] = decide_mean(
            indices, [exclusive[i].duration for i in indices]
        )

    predictions = []
    for turn in turns:
        decision = cluster_decision[turn.cluster]
        if strategy == "cluster+turn-override":
            inside = {
                i: min(turn.end, exclusive[i].end) - max(turn.start, exclusive[i].start)
                for i in members[turn.cluster]
            }
            inside = {i: seconds for i, seconds in inside.items() if seconds > 0}
            if inside:
                own = decide_mean(list(inside), list(inside.values()))
                if own.identity not in (REVIEW, UNKNOWN, decision.identity):
                    decision = own
        predictions.append(Prediction(turn, decision))
    return predictions
