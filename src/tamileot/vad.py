"""Silero VAD v5 over onnxruntime, plus probability -> interval decoding.

Design note: we cache the raw per-frame speech probability, not decoded
intervals. The probabilities are ~10 MB for the whole stereo-call family, and
every boundary decision downstream (threshold, hysteresis, minimum silence)
becomes a cheap re-derivation instead of a 30-minute re-run. Turn-boundary
precision is the whole point of this project, so those knobs get swept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import audio
from .paths import SILERO_ONNX, VAD

WINDOW = 512                      # samples the v5 model requires at 16 kHz
HOP_S = WINDOW / audio.SR         # 0.032 s per frame
CONTEXT = 64                      # v5 prepends the previous 64 samples


class SileroVAD:
    """One onnxruntime session. Not thread-safe; make one per process."""

    def __init__(self, model: Path = SILERO_ONNX, threads: int = 1):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.sr = np.array(audio.SR, dtype=np.int64)

    def probs(self, x: np.ndarray) -> np.ndarray:
        """Per-frame speech probability, one frame per 512 samples.

        Strictly sequential: the model is recurrent and the LSTM state must
        carry frame to frame. Batching several chunks of one file through the
        batch dimension was tried and rejected -- it is compute-bound, not
        overhead-bound, so it returned only 1.8x while flipping ~0.1% of
        threshold decisions through numerical drift. Measured 194x realtime
        per core, and we parallelise across files instead.
        """
        n = len(x) // WINDOW
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = x[: n * WINDOW].reshape(n, WINDOW)
        ctx = np.zeros((n, CONTEXT), dtype=np.float32)
        ctx[1:] = frames[:-1, -CONTEXT:]
        chunks = np.ascontiguousarray(np.concatenate([ctx, frames], axis=1))

        out = np.empty(n, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        run = self.sess.run
        sr = self.sr
        for i in range(n):
            p, state = run(None, {"input": chunks[i : i + 1], "state": state, "sr": sr})
            out[i] = p[0, 0]
        return out

    def probs_file(self, path: Path) -> np.ndarray:
        return self.probs(audio.read(path))


def cache_path(reco: str) -> Path:
    return VAD / f"{reco}.npy"


def save(reco: str, p: np.ndarray) -> None:
    np.save(cache_path(reco), p.astype(np.float16))


def load(reco: str) -> np.ndarray:
    return np.load(cache_path(reco)).astype(np.float32)


# --------------------------------------------------------------------------
# probability -> intervals
# --------------------------------------------------------------------------

def intervals(
    p: np.ndarray,
    thr_on: float = 0.50,
    thr_off: float = 0.35,
    min_speech_s: float = 0.10,
    min_silence_s: float = 0.10,
) -> list[tuple[float, float]]:
    """Hysteresis decode. Returns speech spans in seconds, no padding applied.

    `min_silence_s` only closes a span once that much sub-threshold audio has
    accumulated, so a single dipped frame inside a word does not split it.
    Kept deliberately short: merging real pauses is the caller's decision, and
    an over-merged VAD silently destroys the negative class.
    """
    spans: list[tuple[int, int]] = []
    speaking = False
    start = 0
    silence = 0
    min_sil_f = max(1, int(round(min_silence_s / HOP_S)))
    for i, v in enumerate(p):
        if not speaking:
            if v >= thr_on:
                speaking, start, silence = True, i, 0
        else:
            if v < thr_off:
                silence += 1
                if silence >= min_sil_f:
                    spans.append((start, i - silence + 1))
                    speaking = False
            else:
                silence = 0
    if speaking:
        spans.append((start, len(p)))

    min_sp_f = max(1, int(round(min_speech_s / HOP_S)))
    return [(a * HOP_S, b * HOP_S) for a, b in spans if b - a >= min_sp_f]


def clip_intervals(iv: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    """Speech spans intersected with [a, b]."""
    out = []
    for s, e in iv:
        s2, e2 = max(s, a), min(e, b)
        if e2 > s2:
            out.append((s2, e2))
    return out


def speech_between(iv: list[tuple[float, float]], a: float, b: float) -> float:
    """Total speech seconds inside [a, b]. Used to prove a channel is silent."""
    return sum(e - s for s, e in clip_intervals(iv, a, b))
