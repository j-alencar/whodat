"""Locates whodat's files using the freedesktop XDG Base Directory spec.

The cache stores files that can be retrieved or rebuilt, such as audio,
models, diarization results, and embeddings. The state directory holds work
in progress, such as pending review questions and proposed enrollment clips.
"""

import os
from collections.abc import Mapping
from pathlib import Path

APP = "whodat"


def _base(env: Mapping[str, str], variable: str, default: str) -> Path:
    value = env.get(variable)
    if value and Path(value).is_absolute():  # the spec ignores relative paths
        return Path(value)
    return Path(env.get("HOME", Path.home())) / default


def cache_dir(env: Mapping[str, str] = os.environ) -> Path:
    return _base(env, "XDG_CACHE_HOME", ".cache") / APP


def state_dir(env: Mapping[str, str] = os.environ) -> Path:
    return _base(env, "XDG_STATE_HOME", ".local/state") / APP
