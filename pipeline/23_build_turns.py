#!/usr/bin/env python3
"""Step 4a -- build the turn-level table.

Regroups speech already on disk into **turns**: one speaker's continuous hold
of the floor, with their own pauses as mid-turn spans and the silence after the
turn as the real end-of-turn. The 8 s `clips` format cannot express that
distinction; this one exists to carry it.

Nothing is re-derived. The VAD is cached, the transcripts are unchanged, and
the call-level split is read from `manifest/split.json` and carried through
untouched -- the test split has been frozen since before training and the paper
depends on it.

Turns come from floor occupancy, never from `turns.Boundary`. See
`src/tamileot/floor.py` for why, with the measurements.

**No audio is cut here.** 32.9 h of turn audio is ~3.8 GB of PCM16, and
`21_pack_for_hf.py` can slice the leg WAVs straight into parquet. Every row
carries `start_time`, `end_time` and `trail_s`, so the slice is reproducible
from the table alone.

    export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1
    python pipeline/23_build_turns.py        # ~1 min, writes gold/turns.jsonl
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot import audio, corpus, floor, turns  # noqa: E402
from tamileot.paths import GOLD, MANIFEST, R1, REPORTS, require_corpus  # noqa: E402

LANG = "ta"
OUT = GOLD / "turns.jsonl"
REPORT = REPORTS / "turns_build.txt"

# Fields lifted from the gold boundary table where one lands on a turn end.
# `split` is deliberately NOT among them -- it comes from the call, so every
# turn gets one whether or not a label exists.
JOIN = ("sid", "label", "label_pipeline", "llm_verdict", "llm_conf",
        "dispute", "source", "harvest")

_CALLS: dict[str, corpus.StereoCall] = {}


def _init() -> None:
    global _CALLS
    _CALLS = corpus.load_stereo_calls(R1)


def _one(base: str) -> list[dict]:
    call = _CALLS[base]
    tl = turns.build(call)
    wav_dur = {side: audio.duration(call.wavs[side]) for side in call.wavs}
    out = []
    for t in floor.build(tl, wav_dur):
        out.append({
            "base": t.base, "side": t.side, "index": t.index,
            "start": t.start, "end": t.end, "trail": t.trail,
            "eot_gap": t.eot_gap, "silence": t.silence,
            "text": t.text, "words": t.words,
            "n_spans": len(t.spans), "absorbed": t.absorbed,
            "starts_segment": t.starts_segment, "ends_segment": t.ends_segment,
            "pause_crosstalk": t.pause_crosstalk, "clean": t.clean,
            "speech_crosstalk": t.speech_crosstalk,
            "speech": round(t.speech, 3), "word_rate": round(t.word_rate, 2),
        })
    return out


def row(t: dict, split: str, gold: dict | None) -> dict:
    """One published row. Field names here are the ones that ship."""
    sp = t["side"][0]
    out = {
        "id": f"{t['base']}_{sp}_{t['index']:04d}",
        "conversation_id": t["base"],
        "speaker_id": f"{t['base']}_{sp}",
        "turn_index": t["index"],
        "language": LANG,
        "start_time": t["start"],
        "end_time": t["end"],
        "duration": round(t["end"] - t["start"], 3),
        "trail_s": t["trail"],
        "transcript": t["text"],
        "n_words": t["words"],
        "speech_s": t["speech"],
        "word_rate": t["word_rate"],
        "n_spans": t["n_spans"],
        "n_absorbed": t["absorbed"],
        "silence_spans": [{"start": a, "end": b} for a, b in t["silence"]],
        "eot_gap": t["eot_gap"],
        "starts_segment": t["starts_segment"],
        "ends_segment": t["ends_segment"],
        "pause_crosstalk": t["pause_crosstalk"],
        "speech_crosstalk": t["speech_crosstalk"],
        "clean": t["clean"],
        "split": split,
    }
    for k in JOIN:
        out[k] = gold.get(k) if gold else None
    return out


def q(a: list[float], f: float) -> float:
    return sorted(a)[min(int(f * len(a)), len(a) - 1)]


def main() -> int:
    require_corpus(R1)
    bases = sorted(corpus.load_stereo_calls(R1))
    if not bases:
        raise SystemExit(f"no calls under {R1} -- is SPRING_INX_R1 set?")

    split = json.loads((MANIFEST / "split.json").read_text(encoding="utf-8"))
    missing = [b for b in bases if b not in split]
    if missing:
        raise SystemExit(f"{len(missing)} calls absent from split.json: {missing[:3]}")

    gold = {}
    for line in (GOLD / "samples_llm_all.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("llm_ok"):
            gold[(r["base"], r["side"], round(r["t"], 3))] = r

    with Pool(12, initializer=_init) as pool:
        per_call = pool.map(_one, bases)

    rows, joined = [], 0
    for turns_of_call in per_call:
        for t in turns_of_call:
            g = gold.get((t["base"], t["side"], t["end"]))
            joined += g is not None
            rows.append(row(t, split[t["base"]], g))

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = _report(rows, joined, len(bases))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}  ({len(rows):,} rows)")
    print(f"wrote {REPORT}")
    return 0


def _report(rows: list[dict], joined: int, n_calls: int) -> list[str]:
    d = [r["duration"] for r in rows]
    e = [r["eot_gap"] for r in rows if r["eot_gap"] is not None]
    ns = [len(r["silence_spans"]) for r in rows]
    aud = sum(r["duration"] + r["trail_s"] for r in rows)
    lab = Counter(r["label"] for r in rows)
    L = ["=" * 72,
         f"TURNS -- {len(rows):,} from {n_calls} calls, both legs",
         "=" * 72,
         f"  duration        median {st.median(d):6.2f}s  mean {st.mean(d):6.2f}s"
         f"  p90 {q(d, .90):6.2f}s  max {max(d):7.2f}s",
         f"  turn audio + trail            : {aud / 3600:6.2f} h",
         f"  mid-turn pauses   median {st.median(ns):.0f}  mean {st.mean(ns):.2f}"
         f"  max {max(ns)}   >=1 pause: {100 * sum(1 for x in ns if x) / len(ns):.1f}%",
         f"  words/turn        median {st.median([r['n_words'] for r in rows]):5.1f}"
         f"   under 3 words: {100 * sum(1 for r in rows if r['n_words'] < 3) / len(rows):.1f}%",
         f"  backchannels absorbed         : {sum(r['n_absorbed'] for r in rows):,}"
         f"  in {sum(1 for r in rows if r['n_absorbed']):,} turns",
         "",
         "  LONG TAIL  (real monologue, not a segmentation artefact -- one R1 call"
         " is 20 min of two sequential monologues at 0.5% overlap)"]
    for thr in (30, 60, 120, 300):
        sel = [x for x in d if x > thr]
        L.append(f"    > {thr:3d}s : {len(sel):5,d} turns ({100 * len(sel) / len(d):5.2f}%)"
                 f"  holding {sum(sel) / 3600:5.2f} h of the {sum(d) / 3600:.2f} h")
    pa = [s["end"] - s["start"] for r in rows for s in r["silence_spans"]]
    L += [f"    mid-turn pause  median {st.median(pa):.2f}s  p99 {q(pa, .99):.2f}s"
          f"  max {max(pa):.2f}s   over 5 s: {sum(1 for x in pa if x > 5):,}"
          f" of {len(pa):,} ({100 * sum(1 for x in pa if x > 5) / len(pa):.1f}%)",
         "",
         "  END-OF-TURN GAP",
         f"    median {st.median(e):5.2f}s   p10 {q(e, .10):6.2f}s   p90 {q(e, .90):5.2f}s",
         f"    >= 0.2 s (clean EOT) : {sum(1 for x in e if x >= .2):6,d}"
         f"  ({100 * sum(1 for x in e if x >= .2) / len(e):5.1f}%)",
         f"    <  0   (overlap)     : {sum(1 for x in e if x < 0):6,d}"
         f"  ({100 * sum(1 for x in e if x < 0) / len(e):5.1f}%)",
         "",
         "  QUALITY FUNNEL  (`clean` -- usable as an end-of-turn example)"]
    gate = [
        ("all turns", lambda r: True),
        ("opens an utterance, not mid-word", lambda r: r["starts_segment"]),
        ("+ transcriber closed the utterance", lambda r: r["ends_segment"]),
        ("+ other speaker not talking across it", lambda r: r["speech_crosstalk"] <= 0.05),
        ("+ handover not overlapped (eot >= 0)",
         lambda r: r["eot_gap"] is not None and r["eot_gap"] >= 0.0),
        ("+ at least 1.0 s of speech", lambda r: r["duration"] >= 1.0),
        ("+ transcript rate >= 1.0 word/s", lambda r: r["word_rate"] >= 1.0),
    ]
    g = rows
    for name, fn in gate:
        g = [r for r in g if fn(r)]
        L.append(f"    {name:38s} {len(g):6,d} ({100 * len(g) / len(rows):5.1f}%)")
    assert len(g) == sum(1 for r in rows if r["clean"]), "gate disagrees with `clean`"
    cd = [r["duration"] for r in g]
    L += [f"    clean: duration median {st.median(cd):.2f}s  "
          f"words median {st.median([r['n_words'] for r in g]):.1f}  "
          f"labelled {sum(1 for r in g if r['label'] is not None):,}  "
          f"audio {sum(r['duration'] + r['trail_s'] for r in g) / 3600:.2f} h",
          "",
          "  LABEL JOIN (gold boundary landing on a turn end)",
         f"    joined   : {joined:6,d}  ({100 * joined / len(rows):5.1f}%)",
         f"    complete : {lab[1]:6,d}",
         f"    incomplete:{lab[0]:6,d}",
         f"    unlabelled:{lab[None]:6,d}",
         "",
         "  SPLIT (carried through from manifest/split.json, unchanged)",
         f"    {'split':6s} {'turns':>7s} {'calls':>6s} {'labelled':>9s} {'clean':>7s}"]
    for name in ("train", "dev", "test"):
        sel = [r for r in rows if r["split"] == name]
        L.append(f"    {name:6s} {len(sel):7,d} "
                 f"{len({r['conversation_id'] for r in sel}):6d} "
                 f"{sum(1 for r in sel if r['label'] is not None):9,d} "
                 f"{sum(1 for r in sel if r['clean']):7,d}")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
