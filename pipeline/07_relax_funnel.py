#!/usr/bin/env python3
"""Step 1a.3c -- harvest boundaries the first pass rejected.

`03_gold.py` found 50,532 boundaries in the 116 calls and kept 16,216 (32%).
The other 34,316 were rejected by gates written when the *pipeline* had to
decide the label on its own. Two of those gates existed for that reason alone,
and a labeller measured at 97.5% against a human listener now decides instead
-- the same argument that unlocked the two held-back sources in
`06_cut_heldback.py`.

Moved:

  min_prev_dur  1.00 -> 0.50   The gate wanted "enough speech to read prosody".
                               Half a second of Tamil is two or three syllables
                               with a pitch contour; the real reason for 1.00
                               was that `prev_dur` was the one known structural
                               confound (`docs/archive/PHASE1A_RESULTS.md` §1), and
                               relabelling by listening already severed that
                               link (Cohen's d -0.33 -> -0.09). Below 0.5 s the
                               spans are mostly VAD fragments, so the floor
                               moves rather than disappears.

  neg_max_pause 2.00 -> 5.00   A speaker who pauses 3 s and then resumes is the
                               single hardest `incomplete` there is, and the
                               gate threw all 377 away because a long pause
                               *might* mean they had finished. That is a label
                               question, and it now has a labeller. Capped at
                               5 s because the tail runs to 80 s, and an 80 s
                               pause is not a pause -- it is a new turn.

NOT moved, and why:

  drop_neg_pause_short  5,475  The largest single pool, and unusable as-is. The
                               speaker who paused is the one who resumes, on
                               this same leg, so a pause shorter than TRAIL_S
                               puts their resumed speech *inside* the clip's
                               trailing window and hands the model the answer.
                               Recovering these needs a geometry change applied
                               to both classes at once, not a gate move.
  drop_straddle         6,174  Genuine collision. `gap` is measured to a later
                               span and means nothing (see `Boundary`).
  drop_bargein            311  Same.
  drop_pos_backchannel  3,778  Would yield ~3.8k more `complete`. The set is
                               already 65% complete; this is the wrong class.
  drop_pos_gap_long     2,132  Also `complete`. Same reason.

Nothing here rewrites `samples.jsonl` or re-derives the split. `03`/`04` do
both, and re-running them would re-draw the sample set and invalidate the 197
human labels and every cached verdict keyed by `sid`. This reads the recorded
split, dedupes against the recorded sids, and writes a new file.

    python pipeline/07_relax_funnel.py --dry-run
    python pipeline/07_relax_funnel.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import corpus, rules, turns  # noqa: E402
from tamileot.paths import GOLD, MANIFEST, R1, REPORTS  # noqa: E402

_CALLS: dict[str, corpus.StereoCall] = {}


def _init() -> None:
    global _CALLS
    _CALLS = corpus.load_stereo_calls(R1)


def _one(base: str) -> dict:
    """Both passes over the same boundaries, so the difference between them is
    attributable to the gates and to nothing else."""
    bs = turns.boundaries(turns.build(_CALLS[base]))
    strict, _ = rules.classify(bs, rules.DEFAULT)
    relaxed, funnel = rules.classify(bs, rules.RELAXED, harvest="relaxed")
    return {
        "base": base,
        "strict_sids": [s.sid for s in strict],
        "relaxed": [rules.as_row(s) for s in relaxed],
        "funnel": funnel,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="samples_relaxed.jsonl")
    a = ap.parse_args()

    have = {json.loads(l)["sid"]
            for l in (GOLD / "samples.jsonl").read_text(encoding="utf-8").splitlines()}
    split = json.loads((MANIFEST / "split.json").read_text(encoding="utf-8"))
    print(f"recorded set: {len(have):,} sids across {len(split)} calls "
          f"({Counter(split.values())})")

    bases = sorted(corpus.load_stereo_calls(R1))
    with Pool(a.workers, initializer=_init) as pool:
        res = pool.map(_one, bases)

    funnel: Counter[str] = Counter()
    strict_now: set[str] = set()
    rows: list[dict] = []
    for r in res:
        funnel.update(r["funnel"])
        strict_now.update(r["strict_sids"])
        rows.extend(r["relaxed"])

    # Three-way split of what the relaxed pass produced. `drift` is the honest
    # part: `04_refresh_text.py` re-apportioned words *after* `samples.jsonl`
    # was written, so the current strict gates keep clips the recorded file
    # does not have. Those are not a relaxation result and are not counted as
    # one -- they are a code/data skew, reported and then excluded.
    known = [r for r in rows if r["sid"] in have]
    drift = [r for r in rows if r["sid"] not in have and r["sid"] in strict_now]
    new = [r for r in rows if r["sid"] not in have and r["sid"] not in strict_now]

    print(f"\nrelaxed pass produced {len(rows):,} samples")
    print(f"  already in samples.jsonl {len(known):7,d}   untouched")
    print(f"  strict-gate drift        {len(drift):7,d}   text refresh, excluded")
    print(f"  NEW from relaxation      {len(new):7,d}   <=")
    assert len(known) + len(drift) + len(new) == len(rows)
    # Every recorded sid must survive a superset of gates, or the relaxation is
    # not a superset and the two sets are no longer comparable.
    missing = have - {r["sid"] for r in rows}
    print(f"\n  recorded sids the relaxed pass does NOT reproduce: {len(missing):,}"
          + ("  (all from the same text refresh)" if missing else ""))

    for r in new:
        r["split"] = split[r["base"]]

    by_src = Counter(r["source"] for r in new)
    by_lab = Counter(r["label"] for r in new)
    print("\nNEW rows by source:")
    for s in ("change", "hold_intra", "change_midseg", "hold_inter"):
        if by_src[s]:
            print(f"  {s:14s} {by_src[s]:6,d}   label {'complete' if s.startswith('change') else 'incomplete'}")
    print(f"  {'-' * 44}")
    print(f"  complete       {by_lab[1]:6,d}")
    print(f"  incomplete     {by_lab[0]:6,d}   <= the class that was short")
    print(f"\nNEW rows by split: {dict(Counter(r['split'] for r in new))}")
    print(f"calls contributing: {len({r['base'] for r in new})}/116")

    # What this does to the balance the whole exercise is about. Gate labels
    # only -- `hold_intra` came back 53% *complete* once a labeller listened, so
    # the real balance is not known until step 07 has run over these rows.
    old_c, old_i = 10_493, 5_720
    print("\nclass balance, gate labels (pre-labelling):")
    print(f"  now      {old_c:6,d} complete / {old_i:6,d} incomplete   "
          f"{old_c / old_i:.2f}:1")
    print(f"  after    {old_c + by_lab[1]:6,d} complete / {old_i + by_lab[0]:6,d} incomplete   "
          f"{(old_c + by_lab[1]) / (old_i + by_lab[0]):.2f}:1")
    print("  (gate labels. 53% of `hold_intra` flipped to complete last time a")
    print("   labeller listened, so treat this as an upper bound on the fix.)")

    # The confound this relaxation could re-create: `min_prev_dur` was the gate
    # holding it shut. Report the shift now; the real test is Cohen's d on the
    # *LLM* labels after step 07, which `09_verify.py` already measures.
    pk = np.array([r["prev_dur"] for r in known])
    pn = np.array([r["prev_dur"] for r in new])
    lk = np.array([r["label"] for r in known])
    ln = np.array([r["label"] for r in new])
    print("\nprev_dur, the one known structural confound:")
    for nm, p, y in (("recorded", pk, lk), ("new", pn, ln)):
        if not len(p):
            continue
        c, i = p[y == 1], p[y == 0]
        d = ((c.mean() - i.mean()) / np.sqrt((c.var() + i.var()) / 2)) if len(c) and len(i) else float("nan")
        print(f"  {nm:9s} median {np.median(p):.2f}s   complete {c.mean():.2f}s  "
              f"incomplete {i.mean():.2f}s   d {d:+.2f}")
    print("  Gate labels, so `d` here is the gates talking to themselves.")
    print("  The number that matters is `09_verify.py` after step 07 relabels these.")

    print("\nfunnel under the relaxed gates:")
    for k in ("total", "drop_short_prev", "drop_straddle", "drop_bargein",
              "cand_change", "drop_pos_gap_long", "drop_pos_backchannel",
              "change_midseg", "keep_pos", "cand_hold", "drop_neg_pause_short",
              "drop_neg_pause_long", "drop_neg_other_spoke",
              "drop_neg_resume_tiny", "hold_inter", "keep_neg"):
        if k in funnel:
            print(f"  {k:24s} {funnel[k]:7,d}")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out = GOLD / a.out
    with out.open("w", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (REPORTS / "funnel_relaxed.json").write_text(
        json.dumps({"funnel": dict(funnel), "new": len(new), "drift": len(drift),
                    "known": len(known), "by_source": dict(by_src),
                    "by_label": {str(k): v for k, v in by_lab.items()},
                    "by_split": dict(Counter(r["split"] for r in new))},
                   indent=1), encoding="utf-8")
    print(f"\nwrote {out}  ({len(new):,} rows)")
    print(f"      {REPORTS / 'funnel_relaxed.json'}")
    print("\nsamples.jsonl, samples_llm.jsonl and split.json are untouched, on purpose.")
    print("next: pipeline/08_cut_relaxed.py, then 11_label.py over the new rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
