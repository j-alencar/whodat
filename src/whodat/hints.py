"""Computes acoustic hints for review questions without changing the matches."""

import math
from dataclasses import dataclass

import numpy as np

from whodat.media import SAMPLE_RATE


@dataclass(frozen=True)
class Profile:
    mean: dict[str, float]
    std: dict[str, float]


def features(waveform: np.ndarray, word_count: int | None = None) -> dict[str, float]:
    """Measures median and range of F0, formants, loudness, and speaking rate."""
    import parselmouth

    seconds = len(waveform) / SAMPLE_RATE
    sound = parselmouth.Sound(waveform.astype(np.float64), SAMPLE_RATE)
    f0 = sound.to_pitch(pitch_floor=60, pitch_ceiling=400).selected_array["frequency"]
    f0 = f0[f0 > 0]
    result = {
        "f0_median": float(np.median(f0)) if len(f0) else math.nan,
        "f0_range": float(np.percentile(f0, 90) - np.percentile(f0, 10))
        if len(f0)
        else math.nan,
        "loudness_db": float(20 * np.log10(np.sqrt(np.mean(waveform**2)) + 1e-9)),
        "speaking_rate": word_count / seconds if word_count and seconds else math.nan,
    }
    formants = sound.to_formant_burg(maximum_formant=5000)
    times = np.arange(formants.start_time, formants.end_time, 0.01)
    for n in (1, 2, 3):
        values = np.array([formants.get_value_at_time(n, t) for t in times])
        values = values[~np.isnan(values)]
        result[f"f{n}_median"] = float(np.median(values)) if len(values) else math.nan
    return result


def build_profile(samples: list[dict[str, float]]) -> Profile:
    keys = samples[0].keys()
    mean, std = {}, {}
    for key in keys:
        values = np.array([s[key] for s in samples], dtype=float)
        values = values[~np.isnan(values)]
        mean[key] = float(values.mean()) if len(values) else math.nan
        std[key] = float(values.std()) if len(values) > 1 else math.nan
    return Profile(mean, std)


def rank_profiles(
    turn: dict[str, float], profiles: dict[str, Profile]
) -> list[tuple[str, float]]:
    """Ranks people by comparing the turn's features with their profiles.

    The ranking uses the mean absolute z-score for each person.
    """
    ranked = []
    for person, profile in profiles.items():
        distances = [
            abs(value - profile.mean[key]) / profile.std[key]
            for key, value in turn.items()
            if key in profile.mean
            and not math.isnan(value)
            and not math.isnan(profile.mean[key])
            and profile.std.get(key, 0) > 0  # NaN compares False
        ]
        if distances:
            ranked.append((person, round(float(np.mean(distances)), 2)))
    return sorted(ranked, key=lambda pair: pair[1])
