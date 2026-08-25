#!/usr/bin/env python3
"""Refresh the `text` / `next_text` on existing samples, in place.

Why this is not just "re-run 03". `_apportion()` feeds two things: the text we
display, and `Span.words`, which the >=3-word backchannel gate consumes. Fixing
the apportionment therefore changes *which boundaries become samples* -- so a
full rebuild silently re-draws the sample set, and with it the 198 QA clips,
the LLM labels keyed by `sid`, the per-call yields the split is derived from,
and the 3.2 GB of cut audio.

That rebuild is worth doing. It is not worth doing by accident, in the middle
of a validation run. So this script does the surgical half: it recomputes the
boundaries with the corrected apportionment, matches them to existing samples
by `sid` -- which is a pure function of (base, side, t) and therefore stable --
and rewrites only the two text fields. Sample set, labels, splits and clips are
untouched.

    python pipeline/04_refresh_text.py [--dry-run]

When you are ready for the real thing, run 03 -> 04 -> 05 and re-draw the QA
sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot import corpus, rules, turns  # noqa: E402
from tamileot.paths import GOLD, R1  # noqa: E402

_CALLS: dict[str, corpus.StereoCall] = {}


def _init() -> None:
    global _CALLS
    _CALLS = corpus.load_stereo_calls(R1)


def _one(base: str) -> dict[str, tuple[str, str]]:
    """sid -> (prev_text, next_text) for every boundary in this call."""
    tl = turns.build(_CALLS[base])
    out = {}
    for b in turns.boundaries(tl):
        sid = f"{b.base}_{b.side[0]}_{int(round(b.t * 1000)):08d}"
        out[sid] = (b.prev_text, b.next_text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = GOLD / "samples.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    bases = sorted({r["base"] for r in rows})
    print(f"{len(rows):,} samples across {len(bases)} calls")

    fresh: dict[str, tuple[str, str]] = {}
    with Pool(12, initializer=_init) as pool:
        for i, d in enumerate(pool.imap_unordered(_one, bases, chunksize=2), 1):
            fresh.update(d)
            if i % 20 == 0 or i == len(bases):
                print(f"  {i}/{len(bases)} calls")

    stat = Counter()
    for r in rows:
        got = fresh.get(r["sid"])
        if got is None:
            stat["missing"] += 1
            continue
        pt, nt = got
        stat["changed_text"] += pt != r.get("text", "")
        stat["changed_next"] += nt != r.get("next_text", "")
        last_old = (r.get("text") or "").split()[-3:]
        last_new = pt.split()[-3:]
        stat["changed_last3"] += last_old != last_new
        if not args.dry_run:
            r["text"], r["next_text"] = pt, nt

    n = len(rows)
    print(f"\n  boundary not re-found for {stat['missing']} samples"
          + ("   <- expected 0" if stat["missing"] else "   <- clean"))
    for k, lab in (("changed_text", "text changed"),
                   ("changed_last3", "LAST THREE WORDS changed"),
                   ("changed_next", "next_text changed")):
        print(f"  {lab:26s} {stat[k]:6,d}  ({100 * stat[k] / n:5.1f}%)")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  rewrote {path}")
    print("  sample set, labels, splits and clips are unchanged by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
