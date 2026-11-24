"""Stores ground-truth labels in TSV files that you can edit by hand."""

from dataclasses import dataclass, field

SCORED_MARK = "# scored"
UNKNOWN = "unknown"
UNSURE = "?"
REVIEW = "review"


@dataclass(frozen=True, order=True)
class Region:
    start: float
    end: float


@dataclass(frozen=True, order=True)
class Label:
    start: float
    end: float
    identity: str


def parse_time(text: str) -> float:
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return round(seconds, 3)


def format_time(seconds: float) -> str:
    tenths = round(seconds * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    return f"{hours:02d}:{minutes:02d}:{rest // 10:02d}.{rest % 10}"


def clock(seconds: float) -> str:
    """Formats time like YouTube, as `H:MM:SS` or as `M:SS` under an hour."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass
class LabelFile:
    scored: list[Region] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)

    def discard(self, labels: list[Label]) -> None:
        """Removes one matching label per entry using the file's timestamp rounding."""
        for label in labels:
            key = _rounded(label)
            for index, kept in enumerate(self.labels):
                if _rounded(kept) == key:
                    del self.labels[index]
                    break

    @classmethod
    def parse(cls, text: str) -> "LabelFile":
        label_file = cls()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# scored "):
                start, end = line.removeprefix("# scored ").split("-")
                label_file.scored.append(Region(parse_time(start), parse_time(end)))
            elif line and not line.startswith("#"):
                start, end, identity = line.split("\t")
                label_file.labels.append(
                    Label(parse_time(start), parse_time(end), identity.strip())
                )
        return label_file

    def format(self) -> str:
        lines = [
            f"# scored {format_time(r.start)}-{format_time(r.end)}"
            for r in sorted(self.scored)
        ]
        lines += [
            f"{format_time(lb.start)}\t{format_time(lb.end)}\t{lb.identity}"
            for lb in sorted(self.labels)
        ]
        return "\n".join(lines) + "\n"

    @classmethod
    def parse_audacity(cls, text: str) -> "LabelFile":
        """Reads an Audacity track where `# scored` labels mark scored regions."""
        label_file = cls()
        for line in text.splitlines():
            if not line.strip() or line.startswith("\\"):
                continue  # spectral-selection lines start with a backslash
            start, end, identity = line.split("\t")
            start, end, identity = (
                round(float(start), 3),
                round(float(end), 3),
                identity.strip(),
            )
            if identity == SCORED_MARK:
                label_file.scored.append(Region(start, end))
            else:
                label_file.labels.append(Label(start, end, identity))
        return label_file

    def format_audacity(self) -> str:
        rows = [(r.start, r.end, SCORED_MARK) for r in self.scored]
        rows += [(lb.start, lb.end, lb.identity) for lb in self.labels]
        return "".join(f"{a:.3f}\t{b:.3f}\t{name}\n" for a, b, name in sorted(rows))

    def without(self, person: str) -> "LabelFile":
        """Copies the file, marking `person` as `unknown` as if off the roster."""
        return LabelFile(
            scored=list(self.scored),
            labels=[
                Label(lb.start, lb.end, UNKNOWN) if lb.identity == person else lb
                for lb in self.labels
            ],
        )


def _rounded(label: Label) -> tuple[str, str, str]:
    return format_time(label.start), format_time(label.end), label.identity
