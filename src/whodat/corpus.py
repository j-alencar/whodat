"""Manages the manifest, roster, thresholds, and label files in a corpus directory."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from whodat.labels import LabelFile, format_time, parse_time
from whodat.match import Thresholds

ROLES = ("enroll", "calibrate", "eval")


@dataclass(frozen=True)
class Clip:
    """A span of an `enroll` video that holds one person's clean speech."""

    video: str
    start: float
    end: float

    def format(self) -> str:
        return f"{self.video} {format_time(self.start)}-{format_time(self.end)}"

    @classmethod
    def parse(cls, text: str) -> "Clip":
        video, span = text.rsplit(" ", 1)
        start, end = span.split("-")
        return cls(video, parse_time(start), parse_time(end))


@dataclass
class Person:
    id: str
    name: str
    color: str
    prominent: list[str] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)


class Corpus:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        manifest = yaml.safe_load((self.root / "manifest.yaml").read_text())
        self.roles: dict[str, str] = {
            str(video): role for video, role in manifest["videos"].items()
        }
        roster = yaml.safe_load((self.root / "roster.yaml").read_text())
        self.persons = {
            pid: Person(
                id=pid,
                name=entry["name"],
                color=entry["color"],
                prominent=[str(v) for v in entry.get("prominent", [])],
                clips=[Clip.parse(c) for c in entry.get("clips", [])],
            )
            for pid, entry in roster["persons"].items()
        }

    @property
    def name(self) -> str:
        return self.root.name

    def videos(self, role: str) -> list[str]:
        return [video for video, r in self.roles.items() if r == role]

    def add_clip(self, person: str, clip: Clip) -> None:
        if self.roles.get(clip.video) != "enroll":
            raise ValueError(f"{clip.video} is not an enroll Video")
        self.persons[person].clips.append(clip)
        self._save_roster()

    def remove_clips(self, person: str, clips: list[Clip]) -> None:
        gone = {c.format() for c in clips}
        self.persons[person].clips = [
            c for c in self.persons[person].clips if c.format() not in gone
        ]
        self._save_roster()

    def clear_clips(self, person: str) -> None:
        self.persons[person].clips = []
        self._save_roster()

    def _save_roster(self) -> None:
        persons = {
            p.id: {
                "name": p.name,
                "color": p.color,
                "prominent": p.prominent,
                "clips": [c.format() for c in sorted(p.clips, key=_clip_order)],
            }
            for p in self.persons.values()
        }
        text = yaml.safe_dump({"persons": persons}, sort_keys=False, width=88)
        (self.root / "roster.yaml").write_text(text)

    # Thresholds, per Embedder

    def _thresholds_file(self) -> dict:
        path = self.root / "thresholds.yaml"
        return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}

    def _save_thresholds_entry(self, embedder: str, **values) -> None:
        data = self._thresholds_file()
        entry = data.setdefault(embedder, {})
        for key, value in values.items():
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        if not entry:
            del data[embedder]
        path = self.root / "thresholds.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=True))

    def thresholds(self, embedder: str) -> Thresholds | None:
        entry = self._thresholds_file().get(embedder) or {}
        return Thresholds(entry["high"], entry["low"]) if "high" in entry else None

    def save_thresholds(self, embedder: str, thresholds: Thresholds) -> None:
        self._save_thresholds_entry(embedder, high=thresholds.high, low=thresholds.low)

    def voice_cut(self, embedder: str) -> float | None:
        """The similarity at or above which candidates form one voice."""
        return (self._thresholds_file().get(embedder) or {}).get("voice")

    def save_voice_cut(self, embedder: str, cut: float | None) -> None:
        self._save_thresholds_entry(embedder, voice=cut)

    # Label Files

    def label_path(self, video: str) -> Path:
        return self.root / "labels" / f"{video}.tsv"

    def label_file(self, video: str) -> LabelFile:
        path = self.label_path(video)
        return LabelFile.parse(path.read_text()) if path.exists() else LabelFile()

    def save_label_file(self, video: str, label_file: LabelFile) -> None:
        path = self.label_path(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label_file.format())


def _clip_order(clip: Clip) -> tuple:
    return (clip.video, clip.start)
