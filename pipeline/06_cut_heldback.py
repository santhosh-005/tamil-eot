#!/usr/bin/env python3
"""Step 1a.4b -- cut the 2,909 held-back clips, without disturbing anything else.

`05_split_extract.py` cuts `rules.CORE` only. The other two sources were kept,
labelled and tagged, but never rendered to audio, because they are the cases
where the two witnesses disagree:

  change_midseg  the floor changed, but the transcriber had the segment still
                 running -- somebody being interrupted mid-utterance.
  hold_inter     the same speaker resumed, but the transcriber closed the
                 segment -- possibly a finished sentence followed by another.

`docs/archive/PHASE1A_RESULTS.md` §1 says promoting `hold_inter` "needs a Tamil sentence
-final filter that does not exist yet". That filter now exists: a labeller that
agrees with a human 97.5% of the time, measured over 197 clips in
`reports/model_bakeoff.md`. So these 2,909 stop being unlabellable and start
being 2,909 samples nobody has -- but they need audio before they can be asked
about, and that is all this script does.

It deliberately does **not** re-derive the split or rewrite `samples.jsonl`.
`04` does both, and re-running it would re-draw the sample set and invalidate
the 197 human labels and every cached verdict keyed by `sid`. This reads the
split that is already recorded and only writes wavs.

    python pipeline/06_cut_heldback.py
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
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (GOLD / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = {r["base"]: r for r in json.loads((MANIFEST / "stereo_calls.json").read_text())}

    sel = [r for r in rows if r["source"] in rules.HELD_BACK]
    if not a.force:
        sel = [r for r in sel if not (CLIPS / r["split"] / f"{r['sid']}.wav").exists()]
    if not sel:
        print("nothing to cut -- every held-back clip already exists")
        return 0

    by = Counter((r["source"], r["split"]) for r in sel)
    print(f"cutting {len(sel):,} held-back clips")
    for k in sorted(by):
        print(f"  {k[0]:14s} {k[1]:6s} {by[k]:6,d}")

    jobs = [(r["sid"], manifest[r["base"]]["wav"][r["side"]], r["split"],
             r["clip_start"], r["clip_end"]) for r in sel]
    lens, tails = [], []
    with Pool(a.workers) as pool:
        for _sid, n, tail in pool.imap_unordered(_cut, jobs, chunksize=32):
            lens.append(n)
            tails.append(tail)
    lens, tails = np.array(lens), np.array(tails)
    full = int((lens == int(rules.CLIP_S * audio.SR)).sum())
    print(f"\n  {len(lens):,} clips, {full:,} at the full {rules.CLIP_S:.0f}s "
          f"({100 * full / len(lens):.0f}%), median {np.median(lens) / audio.SR:.2f}s")
    # Same check `04` prints: if the trailing window is not silence the clip
    # geometry is wrong and the model would be reading the answer.
    print(f"  peak amplitude in the final {rules.TRAIL_S * 1000:.0f} ms: "
          f"median {np.median(tails):.4f} (low = the trailing window really is silence)")
    print(f"  -> {CLIPS}")
    print("\n  samples.jsonl and split.json are untouched, on purpose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
