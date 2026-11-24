"""Uses pyannote community-1 to find voice turns in bounded audio windows."""

import json
from dataclasses import dataclass

import torch

from whodat import media
from whodat.labels import Region
from whodat.match import Turn
from whodat.paths import cache_dir
from whodat.runtime import device

MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class Diarization:
    turns: list[Turn]
    """Turns of every voice, overlapping where people talk at once."""
    exclusive: list[Turn]
    """Turns of one voice at a time: the likeliest one during crosstalk."""


class Diarizer:
    def __init__(self) -> None:
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(MODEL).to(device())
        return self._pipeline

    def diarize(self, video: str, window: Region) -> Diarization:
        """Finds turns with cluster labels local to `window`."""
        path = (
            cache_dir()
            / "diarization"
            / video
            / f"{window.start:.1f}-{window.end:.1f}.turns.json"
        )
        if path.exists():
            data = json.loads(path.read_text())
            return Diarization(
                *([Turn(*row) for row in data[key]] for key in ("turns", "exclusive"))
            )

        waveform = torch.from_numpy(media.read(video, window.start, window.end))
        output = self.pipeline(
            {"waveform": waveform.unsqueeze(0), "sample_rate": media.SAMPLE_RATE}
        )
        prefix = f"{window.start:.0f}s/"

        def turns(annotation) -> list[Turn]:
            return [
                Turn(
                    round(window.start + segment.start, 3),
                    round(window.start + segment.end, 3),
                    prefix + cluster,
                )
                for segment, _, cluster in annotation.itertracks(yield_label=True)
            ]

        result = Diarization(
            turns(output.speaker_diarization),
            turns(output.exclusive_speaker_diarization),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    key: [[t.start, t.end, t.cluster] for t in getattr(result, key)]
                    for key in ("turns", "exclusive")
                }
            )
        )
        return result
