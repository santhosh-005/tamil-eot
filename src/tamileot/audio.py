"""Thin wav I/O. Every file in this corpus is 16 kHz mono 16-bit PCM; we assert
that rather than resampling silently, so a format surprise is loud."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000


def duration(path: Path) -> float:
    info = sf.info(str(path))
    assert info.samplerate == SR, f"{path}: {info.samplerate} Hz, expected {SR}"
    assert info.channels == 1, f"{path}: {info.channels} channels, expected mono"
    return info.frames / info.samplerate


def read(path: Path) -> np.ndarray:
    """Whole file as float32 in [-1, 1]."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == SR, f"{path}: {sr} Hz, expected {SR}"
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x)


def read_window(path: Path, start_s: float, end_s: float) -> np.ndarray:
    """Samples in [start_s, end_s), clamped to the file and zero-padded on the
    left if start_s < 0 so the caller always gets the length it asked for."""
    n = sf.info(str(path)).frames
    a, b = int(round(start_s * SR)), int(round(end_s * SR))
    lead = max(0, -a)
    a2, b2 = max(0, a), min(n, b)
    if b2 <= a2:
        return np.zeros(max(0, b - a), dtype=np.float32)
    with sf.SoundFile(str(path)) as f:
        f.seek(a2)
        x = f.read(b2 - a2, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    tail = max(0, b - b2)
    if lead or tail:
        x = np.concatenate([np.zeros(lead, np.float32), x, np.zeros(tail, np.float32)])
    return np.ascontiguousarray(x, dtype=np.float32)


def write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")
