#!/usr/bin/env python3
"""Step 1a.3 -- build the labelled gold set from the 116 stereo calls.

Writes data/gold/samples.jsonl and prints the full funnel plus the
measurements that justify each gate.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import corpus, rules, turns  # noqa: E402
from tamileot.paths import GOLD, R1, REPORTS  # noqa: E402

_CALLS: dict[str, corpus.StereoCall] = {}


def _init() -> None:
    global _CALLS
    _CALLS = corpus.load_stereo_calls(R1)


def _one(base: str) -> dict:
    tl = turns.build(_CALLS[base])
    bs = turns.boundaries(tl)
    samples, funnel = rules.classify(bs)
    return {
        "base": base,
        "samples": [rules.as_row(s) for s in samples],
        "funnel": funnel,
        "stats": tl.stats,
        "dropped_bleed": tl.dropped_bleed,
        "dropped_bleed_time": tl.dropped_bleed_time,
        "change_gaps": [round(b.gap, 3) for b in bs if b.kind == turns.CHANGE],
        "hold_gaps": [round(b.gap, 3) for b in bs if b.kind == turns.HOLD],
    }


def pct(a: np.ndarray, edges: list[tuple[float, float, str]]) -> None:
    n = len(a)
    for lo, hi, lab in edges:
        c = int(((a >= lo) & (a < hi)).sum())
        print(f"    {lab:26s} {c:7,d}  ({100 * c / n:5.1f}%)")


def main() -> int:
    bases = sorted(corpus.load_stereo_calls(R1))
    with Pool(12, initializer=_init) as pool:
        res = pool.map(_one, bases)

    funnel: Counter[str] = Counter()
    samples: list[dict] = []
    cg, hg = [], []
    bleed_n, bleed_t = 0, 0.0
    seg_sp, vad_sp, kept_sp, ovl = 0.0, 0.0, 0.0, 0.0
    for r in res:
        funnel.update(r["funnel"])
        samples.extend(r["samples"])
        cg.extend(r["change_gaps"])
        hg.extend(r["hold_gaps"])
        bleed_n += r["dropped_bleed"]
        bleed_t += r["dropped_bleed_time"]
        st = r["stats"]
        seg_sp += sum(st["seg_speech"].values())
        vad_sp += sum(st["vad_speech"].values())
        kept_sp += sum(st["kept_speech"].values())
        ovl += st["overlap_s"]

    print("=" * 72)
    print("SPEECH ACCOUNTING (both legs, 116 calls)")
    print("=" * 72)
    print(f"  transcript segments span   : {seg_sp / 3600:7.2f} h")
    print(f"  VAD speech                 : {vad_sp / 3600:7.2f} h  "
          f"({100 * vad_sp / seg_sp:.1f}% of segment span)")
    print(f"  ... backed by a transcript : {kept_sp / 3600:7.2f} h")
    print(f"  dropped as crosstalk bleed : {bleed_t / 3600:7.2f} h in {bleed_n:,} spans "
          f"({100 * bleed_t / vad_sp:.1f}% of VAD speech)")
    print(f"\n  simultaneous speech, VAD level: {100 * ovl / kept_sp:.1f}% of speech")
    print("    (the R1 audit measured 28.1% at the transcript-segment level;")
    print("     the difference is the padding those segments carry)")

    cg, hg = np.array(cg), np.array(hg)
    print("\n" + "=" * 72)
    print(f"BOUNDARIES: {len(cg) + len(hg):,} total  "
          f"({len(cg):,} floor changes, {len(hg):,} same-speaker holds)")
    print("=" * 72)
    print("  gap at floor changes (VAD offset -> other leg onset):")
    pct(cg, [(0, .2, "0-200ms"), (.2, .5, "200-500ms"), (.5, .75, "500-750ms"),
             (.75, 1.5, "0.75-1.5s"), (1.5, 3.0, "1.5-3s"), (3.0, 1e9, ">3s")])
    print(f"    median {np.median(cg):.3f}s")
    print("  pause at same-speaker holds:")
    pct(hg, [(0, .25, "<250ms"), (.25, .5, "250-500ms"), (.5, 1.0, "0.5-1s"),
             (1.0, 2.0, "1-2s"), (2.0, 1e9, ">2s")])
    print(f"    median {np.median(hg):.3f}s")

    print("\n" + "=" * 72)
    print("FUNNEL")
    print("=" * 72)
    order = ["total", "drop_short_prev", "drop_no_text", "drop_straddle", "drop_bargein",
             "cand_change", "drop_pos_gap_short", "drop_pos_gap_long",
             "drop_pos_backchannel", "change_midseg", "keep_pos",
             "cand_hold", "drop_neg_pause_short", "drop_neg_pause_long",
             "drop_neg_other_spoke", "drop_neg_resume_tiny", "hold_inter", "keep_neg"]
    for k in order:
        if k in funnel:
            mark = "  <=" if k.startswith("keep") else ""
            print(f"  {k:24s} {funnel[k]:7,d}{mark}")

    by_src = Counter(s["source"] for s in samples)
    core = sum(by_src[s] for s in rules.CORE)
    print("\n" + "=" * 72)
    print("GOLD SET")
    print("=" * 72)
    print(f"  change      label 1  {by_src['change']:6,d}   floor changed at a segment end")
    print(f"  hold_intra  label 0  {by_src['hold_intra']:6,d}   pause inside one segment")
    print(f"  {'-' * 62}")
    print(f"  CORE                 {core:6,d}   balance "
          f"{by_src['change'] / max(core, 1):.2f} positive")
    print("\n  held back (labelled and kept, out of the core set):")
    print(f"  change_midseg        {by_src['change_midseg']:6,d}   other side cut in mid-utterance")
    print(f"  hold_inter           {by_src['hold_inter']:6,d}   pause straddles a segment edge")
    print(f"\n  calls contributing   {len({s['base'] for s in samples})}/116")

    out = GOLD / "samples.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (REPORTS / "funnel.json").write_text(json.dumps(dict(funnel), indent=1), encoding="utf-8")
    print(f"\nwrote {out}  ({len(samples):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
