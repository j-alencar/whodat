"""Provides interchangeable models for voice embeddings."""

from typing import Protocol

import numpy as np
import torch

from whodat.media import SAMPLE_RATE
from whodat.paths import cache_dir
from whodat.runtime import device

MIN_SECONDS = 0.5
"""Minimum duration in seconds for reliable embeddings in `embed_all`.

Shorter audio gets a NaN row.
"""
MAX_SECONDS = 20.0
"""Maximum duration in seconds. Center-cropping longer audio limits memory use."""


class Embedder(Protocol):
    name: str

    def embed(self, waveform: np.ndarray) -> np.ndarray: ...


def _crop(waveform: np.ndarray) -> np.ndarray:
    limit = int(MAX_SECONDS * SAMPLE_RATE)
    if len(waveform) <= limit:
        return waveform
    start = (len(waveform) - limit) // 2
    return waveform[start : start + limit]


class Ecapa:
    name = "ecapa"

    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cache_dir() / "models" / "ecapa"),
            run_opts={"device": str(device())},
        )

    @torch.inference_mode()
    def embed(self, waveform: np.ndarray) -> np.ndarray:
        signal = torch.from_numpy(_crop(waveform)).unsqueeze(0).to(device())
        return self.model.encode_batch(signal).squeeze().cpu().numpy()


class WeSpeaker:
    """Loads `pyannote/wespeaker-voxceleb-resnet34-LM`, whodat's default embedder."""

    name = "wespeaker"

    def __init__(self) -> None:
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self.inference = Inference(model, window="whole", device=device())

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        signal = torch.from_numpy(_crop(waveform)).unsqueeze(0)
        return np.asarray(
            self.inference({"waveform": signal, "sample_rate": SAMPLE_RATE})
        )


EMBEDDERS = {"ecapa": Ecapa, "wespeaker": WeSpeaker}


def embed_all(embedder: Embedder, waveforms: list[np.ndarray]) -> np.ndarray:
    """Returns one row per waveform, filled with NaNs below `MIN_SECONDS`."""
    rows = [
        embedder.embed(w) if len(w) >= MIN_SECONDS * SAMPLE_RATE else None
        for w in waveforms
    ]
    width = next((len(r) for r in rows if r is not None), 1)
    return np.array([r if r is not None else np.full(width, np.nan) for r in rows])
