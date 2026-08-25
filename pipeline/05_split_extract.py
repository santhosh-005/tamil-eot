#!/usr/bin/env python3
"""Step 1a.4 -- split by call, then cut the audio clips.

The split is by *call*, never by sample: both SPRING_INX releases ship
utterance-level splits that put the same recording on both sides (511 of 545
R2 recordings appear in train and eval alike), and a turn detector trained
across such a split memorises voices instead of learning prosody.

The held-out test calls are the deliverable that outlives the model: exact
channel-derived labels, real Tamil BPO calls, native telephony band. They must
never be trained on.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import audio, corpus, rules  # noqa: E402
from tamileot.paths import CLIPS, GOLD, MANIFEST  # noqa: E402

N_TEST, N_DEV, SEED = 30, 16, 20260819


def choose_split(per_call: dict[str, int]) -> dict[str, str]:
    """Serpentine assignment over calls ordered by yield, so test and dev get a
    representative mix of heavy and light calls rather than a random draw that
    could hand test only the quiet ones."""
    order = sorted(per_call, key=lambda b: (-per_call[b], b))
    rng = random.Random(SEED)
    buckets: list[list[str]] = [order[i : i + 8] for i in range(0, len(order), 8)]
    for b in buckets:
        rng.shuffle(b)
    flat = [b for bucket in buckets for b in bucket]
    split = {}
    for i, base in enumerate(flat):
        m = i % 8
        split[base] = "test" if m < 2 else ("dev" if m == 2 else "train")
    # trim to the requested sizes, moving the excess into train
    for name, want in (("test", N_TEST), ("dev", N_DEV)):
        have = [b for b in flat if split[b] == name]
        for b in have[want:]:
            split[b] = "train"
    return split


def _cut(job: tuple[str, str, str, float, float]) -> tuple[str, int, float]:
    sid, wav, split, a, b = job
    x = audio.read_window(Path(wav), a, b)
    out = CLIPS / split / f"{sid}.wav"
    audio.write(out, x)
    return sid, len(x), float(np.abs(x[-int(0.2 * audio.SR):]).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true", help="split only, skip wav cutting")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (GOLD / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = {r["base"]: r for r in json.loads((MANIFEST / "stereo_calls.json").read_text())}

    usable = [r for r in rows if rules.is_core(r)]
    per_call = defaultdict(int)
    for r in usable:
        per_call[r["base"]] += 1
    for b in manifest:
        per_call.setdefault(b, 0)

    split = choose_split(per_call)
    for r in rows:
        r["split"] = split[r["base"]]

    print(f"{'split':6s} {'calls':>6s} {'complete':>9s} {'incomplete':>11s} {'held-back':>10s}")
    for name in ("train", "dev", "test"):
        sel = [r for r in rows if r["split"] == name]
        nc = len({b for b in split if split[b] == name})
        p = sum(1 for r in sel if r["source"] == "change")
        n = sum(1 for r in sel if r["source"] == "hold_intra")
        h = sum(1 for r in sel if r["source"] in rules.HELD_BACK)
        print(f"{name:6s} {nc:6d} {p:9,d} {n:11,d} {h:10,d}")

    with (GOLD / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (MANIFEST / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"\nwrote {MANIFEST / 'split.json'}")

    if args.no_audio:
        return 0

    jobs = []
    for r in usable:
        side = r["side"]
        jobs.append((r["sid"], manifest[r["base"]]["wav"][side], split[r["base"]],
                     r["clip_start"], r["clip_end"]))
    print(f"\ncutting {len(jobs):,} clips ...")
    lens, tails = [], []
    with Pool(12) as pool:
        for sid, n, tail in pool.imap_unordered(_cut, jobs, chunksize=32):
            lens.append(n)
            tails.append(tail)
    lens, tails = np.array(lens), np.array(tails)
    full = int((lens == int(rules.CLIP_S * audio.SR)).sum())
    print(f"  {len(lens):,} clips, {full:,} at the full {rules.CLIP_S:.0f}s "
          f"({100 * full / len(lens):.0f}%), median {np.median(lens) / audio.SR:.2f}s")
    print(f"  peak amplitude in the final 200 ms: median {np.median(tails):.4f} "
          f"(low = the trailing window really is silence)")
    print(f"  -> {CLIPS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
