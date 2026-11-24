"""Writes evaluation metrics as JSON and plots as PNG files."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from whodat.labels import REVIEW, UNKNOWN
from whodat.match import Thresholds
from whodat.metrics import ScoreSheet

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {
    "right Person": "#2a78d6",
    "wrong Person": "#eb6834",
    "unknown speech": "#1baf7a",
}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.title.set_color(INK)


def plot_scores(matches: dict, thresholds: Thresholds, path: Path) -> None:
    groups = {name: ([], []) for name in SERIES}
    for (score, hint, reference), seconds in matches.items():
        name = (
            "unknown speech"
            if reference == UNKNOWN
            else "right Person"
            if hint == reference
            else "wrong Person"
        )
        groups[name][0].append(score)
        groups[name][1].append(seconds)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bins = np.linspace(-0.2, 1.0, 49)
    for name, (scores, weights) in groups.items():
        if scores:
            ax.hist(
                scores,
                bins=bins,
                weights=np.array(weights) / 60,
                histtype="step",
                linewidth=2,
                color=SERIES[name],
                label=name,
            )
    marks = (
        [(thresholds.low, f"high = low {thresholds.low:.2f} ", "right")]
        if thresholds.low == thresholds.high
        else [
            (thresholds.low, f"low {thresholds.low:.2f} ", "right"),
            (thresholds.high, f" high {thresholds.high:.2f}", "left"),
        ]
    )
    for value, text, align in marks:
        ax.axvline(value, color=MUTED, linewidth=1, linestyle="--")
        ax.text(
            value, ax.get_ylim()[1], text, color=MUTED, ha=align, va="top", fontsize=8
        )
    ax.set_xlabel("Top Match cosine score", color=MUTED)
    ax.set_ylabel("Minutes", color=MUTED)
    ax.set_title("Top Match score by labeled identity", loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(sheet: ScoreSheet, persons: list[str], path: Path) -> None:
    confusion = sheet.confusion()
    rows = [p for p in persons + [UNKNOWN] if any(r == p for r, _ in confusion)]
    cols = persons + [UNKNOWN, REVIEW]
    matrix = np.array([[confusion.get((r, c), 0.0) for c in cols] for r in rows])
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(
        figsize=(1.1 * len(cols) + 2, 0.6 * len(rows) + 1.6), dpi=150
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)  # empty cells stay blank
    ax.imshow(
        np.where(matrix > 0, share, np.nan), cmap=cmap, vmin=0, vmax=1, aspect="auto"
    )
    for i in range(len(rows)):
        for j in range(len(cols)):
            if matrix[i, j]:
                text = f"{share[i, j]:.0%}" if share[i, j] >= 0.005 else "<1%"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if share[i, j] > 0.5 else INK,
                )
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Labeled", color=MUTED)
    ax.set_title("Share of labeled speech time by prediction", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_run(
    folder: Path,
    sheets: dict[str, ScoreSheet],
    total: ScoreSheet,
    impostor: ScoreSheet,
    thresholds: Thresholds,
    persons: list[str],
    settings: dict,
) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "settings": settings,
        "thresholds": {"high": thresholds.high, "low": thresholds.low},
        "overall": total.report(),
        "impostor": impostor.report(),
        "videos": {video: sheet.report() for video, sheet in sheets.items()},
        "confusion_seconds": {
            f"{r} -> {p}": round(s, 1)
            for (r, p), s in sorted(total.confusion().items())
        },
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    plot_scores(
        Counter(total.matches) + Counter(impostor.matches),
        thresholds,
        folder / "scores.png",
    )
    plot_confusion(total, persons, folder / "confusion.png")
    return result
