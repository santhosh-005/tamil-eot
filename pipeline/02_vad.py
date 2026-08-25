#!/usr/bin/env python3
"""Step 1a.2 -- Silero VAD over both legs of every stereo call.

Caches raw per-frame speech probabilities to data/vad/<reco>.npy (float16).
Boundary decoding happens downstream so thresholds can be swept without
re-running the model.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import audio, corpus, vad  # noqa: E402
from tamileot.paths import MANIFEST  # noqa: E402

_VAD: vad.SileroVAD | None = None


def _init() -> None:
    global _VAD
    _VAD = vad.SileroVAD(threads=1)


def _run(job: tuple[str, str]) -> tuple[str, int, float]:
    reco, wav = job
    if vad.cache_path(reco).exists():
        p = vad.load(reco)
        return reco, len(p), -1.0
    t0 = time.time()
    p = _VAD.probs_file(Path(wav))
    vad.save(reco, p)
    return reco, len(p), time.time() - t0


def main() -> int:
    rows = json.loads((MANIFEST / "stereo_calls.json").read_text(encoding="utf-8"))
    jobs = [
        (corpus.leg_reco(r["base"], side), r["wav"][side])
        for r in rows
        for side in corpus.SIDES
    ]
    total_audio = sum(r["wav_dur"] for r in rows) * 2
    print(f"{len(jobs)} legs, {total_audio / 3600:.2f} h of audio")

    t0 = time.time()
    done, cached = [], 0
    with Pool(12, initializer=_init) as pool:
        for i, (reco, n, dt) in enumerate(pool.imap_unordered(_run, jobs, chunksize=1), 1):
            if dt < 0:
                cached += 1
            done.append(n)
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i:3d}/{len(jobs)}  {el:6.1f}s elapsed, eta {el / i * (len(jobs) - i):5.1f}s")

    wall = time.time() - t0
    frames = sum(done)
    print(f"\n{frames:,} frames ({frames * vad.HOP_S / 3600:.2f} h) in {wall:.1f}s")
    if cached < len(jobs):
        print(f"  {total_audio / wall:.0f}x realtime wall-clock across 12 workers")
    print(f"  cache hits: {cached}/{len(jobs)}")

    p = vad.load(jobs[0][0])
    iv = vad.intervals(p)
    print(f"\nsanity, {jobs[0][0]}: {len(iv)} speech spans, "
          f"{sum(e - s for s, e in iv) / (len(p) * vad.HOP_S) * 100:.1f}% of the leg is speech")
    print(f"  prob distribution: <0.1 {np.mean(p < 0.1) * 100:.1f}%  "
          f">0.9 {np.mean(p > 0.9) * 100:.1f}%  in-between {np.mean((p >= 0.1) & (p <= 0.9)) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
