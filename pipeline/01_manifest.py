#!/usr/bin/env python3
"""Step 1a.1 -- assemble and re-verify the 116 stereo telephone conversations.

Writes data/manifest/stereo_calls.json. Re-runs the checks from
docs/archive/DATASET_AUDIT_SPRING_INX_R1.md section 2 so the claim that the two legs are
separated speakers is a property this pipeline asserts, not something taken on
trust from an earlier session.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot import audio, corpus  # noqa: E402
from tamileot.paths import MANIFEST, R1  # noqa: E402

MAX_DUR_MISMATCH = 0.001  # legs of one call must be sample-aligned


def main() -> int:
    calls = corpus.load_stereo_calls(R1)
    print(f"stereo calls with both legs present: {len(calls)}")

    rows, bad = [], []
    for base in sorted(calls):
        c = calls[base]
        durs = {s: audio.duration(c.wavs[s]) for s in corpus.SIDES}
        mismatch = abs(durs[corpus.LEFT] - durs[corpus.RIGHT])
        if mismatch > MAX_DUR_MISMATCH:
            bad.append((base, mismatch))
        rows.append(
            {
                "base": base,
                "wav_dur": round(durs[corpus.LEFT], 3),
                "dur_mismatch": round(mismatch, 6),
                "wav": {s: str(c.wavs[s]) for s in corpus.SIDES},
                "n_seg": {s: len(c.legs[s]) for s in corpus.SIDES},
                "speech": {s: round(sum(x.dur for x in c.legs[s]), 2) for s in corpus.SIDES},
                "span": round(c.span, 2),
            }
        )

    wav_h = sum(r["wav_dur"] for r in rows) / 3600
    speech_h = sum(sum(r["speech"].values()) for r in rows) / 3600
    segs = sum(sum(r["n_seg"].values()) for r in rows)
    print(f"  call wall-clock          : {wav_h:6.2f} h  ({2 * wav_h:.2f} h of leg audio)")
    print(f"  transcribed speech, both legs summed: {speech_h:6.2f} h")
    # >100% of wall clock means the two speakers' segments cover more time than
    # the call lasts, which only overlap or segment padding can produce.
    print(f"    = {100 * speech_h / wav_h:.1f}% of wall clock, {100 * speech_h / (2 * wav_h):.1f}% of leg audio")
    print(f"  transcript segments      : {segs:,}")
    print(f"  leg duration mismatch >{MAX_DUR_MISMATCH * 1000:.0f} ms: {len(bad)}", end="")
    print("   <- 0 means the legs are sample-aligned" if not bad else f"  {bad[:5]}")

    balance = [min(r["speech"].values()) / max(max(r["speech"].values()), 1e-9) for r in rows]
    balance.sort()
    print(
        "  speech balance between legs (min/max): "
        f"p10={balance[len(balance) // 10]:.2f} median={balance[len(balance) // 2]:.2f}"
    )

    out = MANIFEST / "stereo_calls.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}  ({len(rows)} calls)")
    return 0 if len(rows) == 116 and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
