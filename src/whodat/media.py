"""Caches YouTube audio and captions for each video."""

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf

from whodat.paths import cache_dir

SAMPLE_RATE = 16_000


def media_dir(video: str) -> Path:
    return cache_dir() / "media" / video


def youtube_link(video: str, seconds: float = 0.0) -> str:
    return f"https://youtu.be/{video}?t={int(seconds)}"


def _yt_dlp(*args: str, check: bool = True) -> None:
    import deno

    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--quiet",
            "--no-progress",
            "--js-runtimes",
            f"deno:{deno.find_deno_bin()}",
            *args,
        ],
        check=check,
    )


def fetch(video: str) -> None:
    """Downloads uncached audio and tries to fetch uncached English auto-captions."""
    folder = media_dir(video)
    folder.mkdir(parents=True, exist_ok=True)
    if not list(folder.glob("audio.*")):
        _yt_dlp("-f", "bestaudio", "-o", str(folder / "audio.%(ext)s"), "--", video)
    if not (folder / "captions.en.json3").exists():
        # Best-effort: captions only feed Acoustic Hints; YouTube rate-limits.
        _yt_dlp(
            "--skip-download",
            "--write-auto-subs",
            "--sub-langs",
            "en",
            "--sub-format",
            "json3",
            "-o",
            str(folder / "captions"),
            "--",
            video,
            check=False,
        )


def wav_path(video: str) -> Path:
    """Returns the 16 kHz mono PCM path, converting the downloaded audio once."""
    folder = media_dir(video)
    wav = folder / "audio.16k.wav"
    if not wav.exists():
        fetch(video)
        source = next(p for p in folder.glob("audio.*") if p.suffix != ".wav")
        partial = wav.with_suffix(".partial.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(partial),
            ],
            check=True,
        )
        partial.rename(wav)
    return wav


def duration(video: str) -> float:
    return sf.info(wav_path(video)).duration


def read(video: str, start: float, end: float) -> np.ndarray:
    """Reads mono `float32` samples from `start` up to `end`, in seconds."""
    with sf.SoundFile(wav_path(video)) as audio:
        audio.seek(int(start * SAMPLE_RATE))
        frames = int((end - start) * SAMPLE_RATE)
        return audio.read(frames, dtype="float32")


@cache
def words(video: str) -> list[tuple[float, str]]:
    """Returns timestamped YouTube auto-caption segments, or `[]` if absent."""
    path = media_dir(video) / "captions.en.json3"
    if not path.exists():
        return []
    result = []
    for event in json.loads(path.read_text()).get("events", []):
        base = event.get("tStartMs", 0)
        for seg in event.get("segs") or []:
            word = seg.get("utf8", "").strip()
            if word:
                result.append(((base + seg.get("tOffsetMs", 0)) / 1000, word))
    return result
