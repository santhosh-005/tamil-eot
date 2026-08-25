#!/usr/bin/env python3
"""Step 1a.3d -- cut the clips `07_relax_funnel.py` harvested.

Same job and same care as `06_cut_heldback.py`: read a recorded row set, cut
wavs, touch nothing else. The split is already on every row, assigned from
`data/manifest/split.json` by call, so a clip from a test call cannot land in
train no matter how the harvest is re-run.

Two checks that have to pass, because both classes' geometry is the label's
only hiding place:

  no sid collides with a clip that already exists -- a collision would
  overwrite a clip the 197 human labels or the cached verdicts point at;

  the final TRAIL_S of every clip is silence -- if the speaker's resumed
  speech leaked into the trailing window the model reads the answer instead of
  the prosody, which is the exact failure `NEG_MIN_PAUSE` exists to prevent.

    python pipeline/08_cut_relaxed.py --dry-run
    python pipeline/08_cut_relaxed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import audio, rules  # noqa: E402
from tamileot.paths import CLIPS, GOLD, MANIFEST  # noqa: E402


def _cut(job: tuple[str, str, str, float, float]) -> tuple[str, int, float]:
    sid, wav, split, a, b = job
    x = audio.read_window(Path(wav), a, b)
    audio.write(CLIPS / split / f"{sid}.wav", x)
    return sid, len(x), float(np.abs(x[-int(rules.TRAIL_S * audio.SR):]).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="samples_relaxed.jsonl")
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    rows = [json.loads(l) for l in
            (GOLD / a.rows).read_text(encoding="utf-8").splitlines()]
    manifest = {r["base"]: r for r in json.loads((MANIFEST / "stereo_calls.json").read_text())}

    # A harvested sid must never point at an existing clip. `sid` is
    # base+side+millisecond, so a collision means two different rows claim the
    # same boundary -- either a bug in the harvest or a stale file.
    prior = {p.stem for split in ("train", "dev", "test")
             for p in (CLIPS / split).glob("*.wav")}
    clash = sorted({r["sid"] for r in rows} & prior)
    print(f"{len(rows):,} harvested rows, {len(prior):,} clips already on disk")
    if clash and not a.force:
        print(f"  {len(clash):,} sids already have a clip: {clash[:3]}")
        print("  refusing to overwrite. Pass --force only if you know why they exist.")
        return 1

    by = Counter((r["source"], r["split"]) for r in rows)
    print(f"\ncutting {len(rows):,} clips")
    for k in sorted(by):
        print(f"  {k[0]:14s} {k[1]:6s} {by[k]:6,d}")
    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    jobs = [(r["sid"], manifest[r["base"]]["wav"][r["side"]], r["split"],
             r["clip_start"], r["clip_end"]) for r in rows]
    lens, tails = [], []
    with Pool(a.workers) as pool:
        for _sid, n, tail in pool.imap_unordered(_cut, jobs, chunksize=32):
            lens.append(n)
            tails.append(tail)
    lens, tails = np.array(lens), np.array(tails)

    full = int((lens == int(rules.CLIP_S * audio.SR)).sum())
    print(f"\n  {len(lens):,} clips, {full:,} at the full {rules.CLIP_S:.0f}s "
          f"({100 * full / len(lens):.0f}%), median {np.median(lens) / audio.SR:.2f}s")
    print(f"  peak amplitude in the final {rules.TRAIL_S * 1000:.0f} ms: "
          f"median {np.median(tails):.4f}, p95 {np.percentile(tails, 95):.4f} "
          f"(low = the trailing window really is silence)")
    print(f"  -> {CLIPS}")
    print("\n  samples.jsonl, samples_llm.jsonl and split.json are untouched, on purpose.")
    print("  next: 11_label.py over these rows (needs GEMINI_API_KEYS inline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
