"""Proposes enrollment clips from turns in a roster member's prominent videos."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from whodat.labels import Region
from whodat.match import Turn


def _subtract(span: Region, cuts: list[Region]) -> list[Region]:
    pieces = [span]
    for cut in cuts:
        next_pieces = []
        for piece in pieces:
            if cut.end <= piece.start or cut.start >= piece.end:
                next_pieces.append(piece)
                continue
            if cut.start > piece.start:
                next_pieces.append(Region(piece.start, cut.start))
            if cut.end < piece.end:
                next_pieces.append(Region(cut.end, piece.end))
        pieces = next_pieces
    return pieces


def propose_clips(
    turns: list[Turn], min_duration: float = 3.0, margin: float = 0.25
) -> list[Region]:
    """Returns the longest clips first from the cluster with the most speech.

    Excludes other clusters' speech plus a `margin` on either side.
    """
    if not turns:
        return []
    talk = Counter()
    for turn in turns:
        talk[turn.cluster] += turn.duration
    dominant = talk.most_common(1)[0][0]
    others = [
        Region(t.start - margin, t.end + margin) for t in turns if t.cluster != dominant
    ]
    pieces = [
        piece
        for turn in turns
        if turn.cluster == dominant
        for piece in _subtract(Region(turn.start, turn.end), others)
        if piece.end - piece.start >= min_duration
    ]
    return sorted(pieces, key=lambda r: (-(r.end - r.start), r.start))


def _leave_one_out(units: np.ndarray) -> np.ndarray:
    """Computes the cosine similarity of each unit row to the mean of the other rows."""
    others = units.sum(axis=0) - units
    others /= np.linalg.norm(others, axis=1, keepdims=True)
    return (units * others).sum(axis=1)


def rank_clips[T](
    clips: list[T], rows, min_similarity: float = 0.6
) -> list[tuple[T, float]]:
    """Drops outliers and ranks clips by similarity to their peers, most similar first.

    `rows[i]` is the embedding for `clips[i]`. A clip can be any per-clip record.

    The second pass ranks the remaining clips so outliers cannot skew the mean.
    """
    rows = np.asarray(rows, dtype=float)
    keep = ~np.isnan(rows).any(axis=1)
    units = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    keep &= _leave_one_out(np.where(keep[:, None], units, 0.0)) >= min_similarity
    kept = np.flatnonzero(keep)
    if len(kept) < 2:
        return [(clips[i], 1.0) for i in kept]
    similarity = _leave_one_out(units[kept])
    order = np.argsort(-similarity)
    return [(clips[kept[i]], round(float(similarity[i]), 3)) for i in order]


SAME = "same"
DIFFERENT = "different"


@dataclass(frozen=True)
class Merge:
    """Two groups of candidate voices that average linkage joins into one."""

    similarity: float
    left: list[int]
    right: list[int]


def merges(rows) -> list[Merge]:
    """Lists the merges of average linkage, most similar first.

    The voices change only at these merges, so a cut is a choice of how many
    of them to keep.
    """
    rows = np.asarray(rows, dtype=float)
    members = {i: [i] for i in range(len(rows))}
    result = []
    if len(rows) < 2:
        return result
    for k, (a, b, distance, _) in enumerate(
        linkage(rows, method="average", metric="cosine")
    ):
        left, right = members.pop(int(a)), members.pop(int(b))
        members[len(rows) + k] = left + right
        result.append(Merge(round(1.0 - float(distance), 4), left, right))
    return result


def suggested_merge(similarities: list[float]) -> int:
    """The first merge below the gap that stands out most from the merges above.

    Late merges join large mixed groups, so their gaps are wide anyway.
    Measuring each gap against the spread above it finds where alike
    voices stop.
    """
    if len(similarities) < 2:
        return 0

    def surprise(i: int) -> float:
        spread = max(float(np.std(similarities[: i + 1])), 0.01)
        return (similarities[i] - similarities[i + 1]) / spread

    return max(range(len(similarities) - 1), key=surprise) + 1


def _bounds(count: int, answers: dict[int, str]) -> tuple[int, int]:
    low = max((i + 1 for i, a in answers.items() if a == SAME), default=0)
    high = min((i for i, a in answers.items() if a == DIFFERENT), default=count)
    return low, high


def next_merge(count: int, answers: dict[int, str], suggested: int) -> int | None:
    """The merge to ask about next, or None once the answers settle the cut.

    Merges are indexed most similar first. A merge that is the same person
    implies every merge above it is too, so each answer halves the open range.
    The first questions go to both sides of the suggested merge.
    """
    low, high = _bounds(count, answers)
    open_ = [i for i in range(low, high) if i not in answers]
    if not open_:
        return None
    for i in (suggested, suggested - 1):
        if i in open_:
            return i
    middle = (low + high - 1) / 2
    return min(open_, key=lambda i: (abs(i - middle), i))


def kept_merges(count: int, answers: dict[int, str]) -> int:
    """How many merges to keep, splitting voices where the answers are unsure.

    Two voices of one person cost an extra question later, whereas one voice
    of two people mixes their clips.
    """
    return min(_bounds(count, answers))


def cut_between(similarities: list[float], kept: int) -> float:
    """A similarity cut that keeps the first `kept` merges and no others."""
    above = similarities[kept - 1] if kept > 0 else 1.0
    below = similarities[kept] if kept < len(similarities) else above - 0.1
    return round((above + below) / 2, 4)


def group_voices(rows, seconds, min_similarity: float) -> list[list[int]]:
    """Groups indices of similar candidate voices, most speech first."""
    rows = np.asarray(rows, dtype=float)
    if len(rows) == 1:
        return [[0]]
    tree = linkage(rows, method="average", metric="cosine")
    labels = fcluster(tree, t=1.0 - min_similarity, criterion="distance")
    groups = [list(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    total = [sum(seconds[i] for i in g) for g in groups]
    order = sorted(range(len(groups)), key=lambda k: (-total[k], groups[k][0]))
    return [[int(i) for i in groups[k]] for k in order]
