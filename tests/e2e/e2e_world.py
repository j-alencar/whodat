"""A synthetic corpus whose speakers are pure tones, and an Embedder that hears them.

Real code paths run end to end: media reads real WAV files, Voices group
by embedding, Enrollment ranks clips. Only the models are fake.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from whodat.corpus import Clip
from whodat.media import SAMPLE_RATE

CACHE = Path(tempfile.mkdtemp(prefix="whodat-e2e-cache-"))
os.environ["XDG_CACHE_HOME"] = str(CACHE)

TONES = {"chris": 150.0, "lyle": 220.0, "zach": 300.0, "stranger": 410.0}


class ToneEmbedder:
    """Energy at each speaker's tone: alike speakers embed alike."""

    name = "tone"

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), 1 / SAMPLE_RATE)
        bins = [spectrum[np.argmin(np.abs(freqs - f))] for f in TONES.values()]
        return np.array(bins) + 1e-3


EMBEDDERS = {"tone": ToneEmbedder}
VOICE_CUT = 0.65
"""Splits the tones apart: same-tone candidates score near 1, others near 0."""


def write_audio(video: str, seconds: float, speech: list[tuple[float, float, str]]):
    path = CACHE / "whodat" / "media" / video / "audio.16k.wav"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(video))
    audio = 0.001 * rng.standard_normal(int(seconds * SAMPLE_RATE))
    for start, end, speaker in speech:
        t = np.arange(int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)) / SAMPLE_RATE
        audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] += 0.3 * np.sin(
            2 * np.pi * TONES[speaker] * t
        )
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)


ROSTER = {
    "persons": {
        "chris": {"name": "Chris", "color": "#2a78d6", "prominent": ["CCC"]},
        "lyle": {"name": "Lyle", "color": "#eb6834", "prominent": ["LLL"]},
        "zach": {"name": "Zach", "color": "#1baf7a", "prominent": ["ZZZ"]},
    }
}

# Who dominates each sampled Window; zach/g1 is a compilation of Chris,
# lyle/g2 a guest.
CANDIDATES = {
    "chris": [("CCC", 0, "chris"), ("CCC", 100, "chris")],
    "lyle": [("LLL", 0, "lyle"), ("LLL", 100, "stranger")],
    "zach": [("ZZZ", 0, "chris"), ("ZZZ", 100, "zach"), ("ZZZ", 200, "zach")],
}

# Review Clusters on the eval Video, by true speaker.
CLUSTERS = {
    "0s/SPEAKER_00": ("chris", [(0, 3), (3, 6), (24, 27)]),
    "0s/SPEAKER_01": ("lyle", [(6, 9), (9, 12)]),
    "0s/SPEAKER_02": ("zach", [(12, 18)]),
    "0s/SPEAKER_03": ("stranger", [(18, 21), (21, 24)]),
}


def _proposals() -> dict[str, list[dict]]:
    result = {}
    for person, windows in CANDIDATES.items():
        result[person] = [
            {
                "id": f"g{i + 1}",
                "video": video,
                "window": {"start": start, "end": start + 100},
                "seconds": 30.0,
                "clips": [
                    {"start": start + k * 10.0, "end": start + (k + 1) * 10.0}
                    for k in range(3)
                ],
            }
            for i, (video, start, _) in enumerate(windows)
        ]
    return result


def _audio() -> None:
    speech = {"CCC": [], "LLL": [], "ZZZ": []}
    for windows in CANDIDATES.values():
        for video, start, speaker in windows:
            speech[video].append((start, start + 30, speaker))
    for video, spans in speech.items():
        write_audio(video, 300, spans)
    write_audio(
        "EVA",
        30,
        [(s, e, who) for who, turns in CLUSTERS.values() for s, e in turns],
    )


def build(root: Path, enrolled: bool) -> Path:
    """A Corpus at `root/corpus`, with state under `root/state`."""
    _audio()
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "manifest.yaml").write_text(
        "videos:\n  CCC: enroll\n  LLL: enroll\n  ZZZ: enroll\n  EVA: eval\n"
    )
    roster = json.loads(json.dumps(ROSTER))
    if enrolled:
        for person, video in (("chris", "CCC"), ("lyle", "LLL"), ("zach", "ZZZ")):
            start = 100 if person == "zach" else 0
            roster["persons"][person]["clips"] = [
                Clip(video, start, start + 4).format()
            ]
    (corpus / "roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    (corpus / "thresholds.yaml").write_text(
        yaml.safe_dump({"tone": {"voice": VOICE_CUT}})
    )
    state = root / "state" / "whodat" / "corpus"
    (state / "proposals").mkdir(parents=True)
    for person, proposals in _proposals().items():
        (state / "proposals" / f"{person}.json").write_text(json.dumps(proposals))
    pending = {
        key: {
            "cluster": key,
            "hint": "",
            "score": 0.0,
            "turns": [{"start": float(s), "end": float(e)} for s, e in turns],
        }
        for key, (_, turns) in CLUSTERS.items()
    }
    (state / "pending").mkdir()
    (state / "pending" / "EVA.json").write_text(json.dumps(pending))
    (corpus / "labels").mkdir()
    (corpus / "labels" / "EVA.tsv").write_text("# scored 00:00:00.0-00:00:30.0\n")
    return corpus


def fresh_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="whodat-e2e-"))


def remove(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)
