"""Finds turns in a video, embeds them, and matches them to produce predictions."""

import hashlib
import json

import numpy as np

from whodat import media
from whodat.corpus import Corpus
from whodat.diarize import Diarizer
from whodat.embed import Embedder, embed_all
from whodat.labels import Region
from whodat.match import Decision, Prediction, Thresholds, Turn, identify
from whodat.paths import cache_dir

WINDOW_SECONDS = 1200.0
"""Maximum diarization window in seconds.

The limit bounds RAM use because clustering costs more in longer windows.
"""
PIECE_SECONDS = 10.0
"""Maximum clip piece length in seconds.

The pipeline averages the piece embeddings by duration.
"""


def windows(regions: list[Region], limit: float = WINDOW_SECONDS) -> list[Region]:
    result = []
    for region in regions:
        count = max(1, int(np.ceil((region.end - region.start) / limit)))
        step = (region.end - region.start) / count
        result += [
            Region(
                round(region.start + i * step, 3),
                round(region.start + (i + 1) * step, 3),
            )
            for i in range(count)
        ]
    return result


def mean_unit(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    keep = ~np.isnan(rows).any(axis=1)
    rows, weights = rows[keep], weights[keep]
    units = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    mean = (units * weights[:, None]).sum(axis=0)
    return mean / np.linalg.norm(mean)


def build_voiceprints(corpus: Corpus, embedder: Embedder) -> dict[str, np.ndarray]:
    """Builds a voiceprint for each person with enrollment clips.

    The cache key includes the corpus, embedder, and clip list.
    """
    clips = {p.id: p.clips for p in corpus.persons.values() if p.clips}
    key = hashlib.sha1(
        json.dumps({p: [c.format() for c in cs] for p, cs in clips.items()}).encode()
    ).hexdigest()[:12]
    path = cache_dir() / "voiceprints" / corpus.name / f"{embedder.name}-{key}.npz"
    if path.exists():
        return dict(np.load(path))

    voiceprints = {}
    for person, person_clips in clips.items():
        pieces = [
            (clip.video, Region(t, min(t + PIECE_SECONDS, clip.end)))
            for clip in person_clips
            for t in np.arange(clip.start, clip.end, PIECE_SECONDS)
        ]
        rows = embed_all(embedder, [media.read(v, r.start, r.end) for v, r in pieces])
        weights = np.array([r.end - r.start for _, r in pieces])
        voiceprints[person] = mean_unit(rows, weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **voiceprints)
    return voiceprints


def embed_turns(
    video: str, window: Region, turns: list[Turn], embedder: Embedder
) -> np.ndarray:
    key = hashlib.sha1(json.dumps([[t.start, t.end] for t in turns]).encode())
    path = (
        cache_dir()
        / "embeddings"
        / embedder.name
        / video
        / f"{window.start:.1f}-{window.end:.1f}-{key.hexdigest()[:12]}.npy"
    )
    if path.exists():
        return np.load(path)
    audio = media.read(video, window.start, window.end)
    rate = media.SAMPLE_RATE
    rows = embed_all(
        embedder,
        [
            audio[
                int((t.start - window.start) * rate) : int(
                    (t.end - window.start) * rate
                )
            ]
            for t in turns
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, rows)
    return rows


def predict(
    video: str,
    regions: list[Region],
    diarizer: Diarizer,
    embedder: Embedder,
    voiceprints: dict[str, np.ndarray],
    thresholds: Thresholds,
    strategy: str = "cluster",
) -> list[Prediction]:
    predictions = []
    for window in windows(regions):
        found = diarizer.diarize(video, window)
        if not found.turns:
            continue
        rows = embed_turns(video, window, found.exclusive, embedder)
        predictions += identify(
            found.turns, found.exclusive, rows, voiceprints, thresholds, strategy
        )
    return predictions


def save_predictions(path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                [
                    p.turn.start,
                    p.turn.end,
                    p.turn.cluster,
                    p.decision.identity,
                    round(p.decision.score, 4),
                    p.decision.hint,
                ]
                for p in predictions
            ]
        )
    )


def load_predictions(path) -> list[Prediction]:
    return [
        Prediction(Turn(start, end, cluster), Decision(identity, score, hint))
        for start, end, cluster, identity, score, hint in json.loads(path.read_text())
    ]
