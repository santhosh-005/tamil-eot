#!/usr/bin/env python3
"""Step 4b -- verify `gold/turns.jsonl` before anything is packed or published.

This exists because the replay-pump bug cost 26 accuracy points while producing
entirely plausible numbers, and every failure mode this format has is of the
same kind: a turn that swallows a floor change, a silence span that is really
the other speaker talking, a split that drifted. None of them raise.

The check that matters most is **totality**. Turns are a partition of the
confirmed span sequence, so every span must land in exactly one turn -- not
zero, not two. A definition that loses or double-counts speech fails here
immediately, which is precisely what the two boundary-table reconstructions
in the first draft of the spec would have done.

Checks are re-derived from the corpus, not from `floor.build`'s intermediates,
so a bug in the builder cannot hide itself.

    export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1
    python pipeline/24_verify_turns.py            # exits non-zero on any failure
    python pipeline/24_verify_turns.py --listen 10  # also cut wavs to spot-listen
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

from tamileot import audio, corpus, floor, turns, vad  # noqa: E402
from tamileot.paths import GOLD, MANIFEST, R1, REPORTS, require_corpus  # noqa: E402

EPS = 1.5e-3          # one 16 kHz frame is 62.5 us; ms rounding dominates
LISTEN = REPORTS / "qa" / "turns"

_CALLS: dict[str, corpus.StereoCall] = {}


def _init() -> None:
    global _CALLS
    _CALLS = corpus.load_stereo_calls(R1)


def _facts(base: str) -> dict:
    """Re-derive what the call actually contains, independent of the built table."""
    call = _CALLS[base]
    tl = turns.build(call)
    return {
        "base": base,
        "spans": [(round(s.start, 3), round(s.end, 3), s.side, s.words, s.segs)
                  for s in sorted(tl.spans, key=lambda x: (x.start, x.end))],
        "wav_dur": {side: audio.duration(call.wavs[side]) for side in call.wavs},
    }


class Check:
    def __init__(self) -> None:
        self.fail: dict[str, list[str]] = defaultdict(list)
        self.n: dict[str, int] = defaultdict(int)

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.n[name] += 1
        if not ok:
            self.fail[name].append(detail)

    def report(self) -> bool:
        width = max(len(k) for k in self.n)
        print("=" * (width + 34))
        print("TURN TABLE VERIFICATION")
        print("=" * (width + 34))
        clean = True
        for name in self.n:
            bad = len(self.fail[name])
            mark = "ok  " if not bad else "FAIL"
            print(f"  [{mark}] {name:<{width}}  {self.n[name] - bad:7,d}/{self.n[name]:,}")
            for d in self.fail[name][:5]:
                print(f"         - {d}")
            if bad > 5:
                print(f"         ... and {bad - 5:,} more")
            clean &= not bad
        print()
        print("PASS -- safe to pack" if clean else "FAILED -- do not pack or publish")
        return clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=0, metavar="N",
                    help="cut N random turns to reports/qa/turns/ for spot-listening")
    ap.add_argument("--acoustic", type=int, default=0, metavar="N",
                    help="measure trailing-window loudness over N sampled turns")
    ap.add_argument("--clean", action="store_true",
                    help="restrict --listen/--acoustic to rows where clean is true")
    args = ap.parse_args()

    require_corpus(R1)
    rows = [json.loads(l) for l in
            (GOLD / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    split = json.loads((MANIFEST / "split.json").read_text(encoding="utf-8"))
    bases = sorted({r["conversation_id"] for r in rows})

    with Pool(12, initializer=_init) as pool:
        facts = {f["base"]: f for f in pool.map(_facts, bases)}

    by_call: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_call[r["conversation_id"]].append(r)

    c = Check()
    seen_ids: set[str] = set()

    for base, trs in by_call.items():
        f = facts[base]
        trs.sort(key=lambda r: r["turn_index"])

        # ---- totality: no speech lost, no speech double-counted --------------
        #
        # A span is accounted for when a turn of its OWN side covers it. The
        # only spans allowed to have no turn are the backchannels `runs()`
        # absorbed -- and they must really be backchannels, which is what makes
        # this catch data loss rather than excuse it.
        #
        # Location is deliberately not asserted for those. A backchannel starts
        # inside the turn that absorbed it, but it can run a fraction past the
        # end, and once `_split_lapses` cuts a turn at a long pause a
        # backchannel sitting in that pause ends up between two turns rather
        # than inside either. It is still a backchannel and still not a turn.
        own: dict[tuple, int] = {}
        for r in trs:
            side = "Left" if r["speaker_id"].endswith("_L") else "Right"
            for sp in f["spans"]:
                if (sp[2] == side and sp[0] >= r["start_time"] - EPS
                        and sp[1] <= r["end_time"] + EPS):
                    own[sp] = own.get(sp, 0) + 1
        for sp in f["spans"]:
            n_own = own.get(sp, 0)
            c("no speech double-counted", n_own <= 1,
              f"{base} {sp[2]} [{sp[0]}, {sp[1]}] in {n_own} same-side turns")
            c("no speech lost -- unclaimed spans are backchannels",
              n_own >= 1 or (sp[3] < floor.BACKCHANNEL_WORDS
                             and (sp[1] - sp[0]) < floor.BACKCHANNEL_DUR),
              f"{base} {sp[2]} [{sp[0]}, {sp[1]}] "
              f"({sp[1] - sp[0]:.2f}s / {sp[3]:.1f}w) in no turn")

        # ---- per-turn invariants --------------------------------------------
        for i, r in enumerate(trs):
            rid = r["id"]
            side = "Left" if r["speaker_id"].endswith("_L") else "Right"

            c("id unique", rid not in seen_ids, rid)
            seen_ids.add(rid)
            c("turn_index contiguous and ordered", r["turn_index"] == i,
              f"{rid}: index {r['turn_index']} at position {i}")
            c("duration == end - start",
              abs(r["duration"] - (r["end_time"] - r["start_time"])) < EPS, rid)
            c("split matches the call's split", r["split"] == split[base],
              f"{rid}: {r['split']} != {split[base]}")

            # silence spans: inside, ordered, disjoint, non-empty
            prev_end = r["start_time"]
            for s in r["silence_spans"]:
                c("silence spans inside the turn",
                  s["start"] >= r["start_time"] - EPS and s["end"] <= r["end_time"] + EPS,
                  f"{rid}: [{s['start']}, {s['end']}] vs [{r['start_time']}, {r['end_time']}]")
                c("silence spans ordered and disjoint", s["start"] >= prev_end - EPS,
                  f"{rid}: {s['start']} < {prev_end}")
                c("silence spans non-empty", s["end"] > s["start"],
                  f"{rid}: [{s['start']}, {s['end']}]")
                prev_end = s["end"]

            # n_absorbed counts the other-side spans inside this turn that no
            # same-side turn of their own claims -- see floor._count_absorbed
            absorbed = [sp for sp in f["spans"]
                        if sp[2] != side and r["start_time"] <= sp[0] < r["end_time"]
                        and own.get(sp, 0) == 0]
            c("n_absorbed counts the folded-in backchannels",
              len(absorbed) == r["n_absorbed"],
              f"{rid}: {len(absorbed)} unclaimed other-side spans, "
              f"n_absorbed={r['n_absorbed']}")

            # eot_gap agrees with the next turn
            if i + 1 < len(trs):
                c("eot_gap == next turn start - this end",
                  r["eot_gap"] is not None
                  and abs(r["eot_gap"] - (trs[i + 1]["start_time"] - r["end_time"])) < EPS,
                  f"{rid}: {r['eot_gap']}")
            else:
                c("last turn of a call has eot_gap null", r["eot_gap"] is None, rid)

            # the audio slice fits inside the leg wav -- no silent zero padding
            c("audio window inside the wav",
              r["start_time"] >= -EPS
              and r["end_time"] + r["trail_s"] <= f["wav_dur"][side] + EPS,
              f"{rid}: {r['end_time'] + r['trail_s']:.3f} > {f['wav_dur'][side]:.3f}")

            # the trailing window must not contain this speaker's own next speech
            own_next = [sp for sp in f["spans"]
                        if sp[2] == side and sp[0] > r["end_time"] + EPS]
            if own_next:
                c("no owner speech inside the trailing window",
                  own_next[0][0] >= r["end_time"] + r["trail_s"] - EPS,
                  f"{rid}: own speech at {own_next[0][0]}, trail ends "
                  f"{r['end_time'] + r['trail_s']:.3f}")

            # ---- quality witnesses, re-derived from the transcript ----------
            mine = [sp for sp in f["spans"] if sp[2] == side]
            held = [sp for sp in mine
                    if sp[0] >= r["start_time"] - EPS and sp[1] <= r["end_time"] + EPS]
            if held:
                a, b = mine.index(held[0]), mine.index(held[-1])
                starts = a == 0 or not (set(mine[a - 1][4]) & set(held[0][4]))
                ends = b == len(mine) - 1 or not (set(mine[b + 1][4]) & set(held[-1][4]))
                c("starts_segment matches the transcript", r["starts_segment"] == starts,
                  f"{rid}: {r['starts_segment']} != {starts}")
                c("ends_segment matches the transcript", r["ends_segment"] == ends,
                  f"{rid}: {r['ends_segment']} != {ends}")

            oiv = [(sp[0], sp[1]) for sp in f["spans"] if sp[2] != side]
            xt = max((vad.speech_between(oiv, s["start"], s["end"])
                      for s in r["silence_spans"]), default=0.0)
            sp_s = r["duration"] - sum(s["end"] - s["start"] for s in r["silence_spans"])
            c("speech_s == duration minus the pauses",
              abs(r["speech_s"] - sp_s) < EPS, f"{rid}: {r['speech_s']} != {sp_s:.3f}")
            c("word_rate == n_words / speech_s",
              abs(r["word_rate"] - (r["n_words"] / r["speech_s"] if r["speech_s"] > 0 else 0)) < 0.01,
              f"{rid}: {r['word_rate']}")

            c("pause_crosstalk matches the other leg",
              abs(r["pause_crosstalk"] - xt) < EPS,
              f"{rid}: {r['pause_crosstalk']} != {xt:.3f}")

            c("clean agrees with its six gates",
              r["clean"] == (r["starts_segment"] and r["ends_segment"]
                             and r["speech_crosstalk"] <= floor.MAX_PAUSE_CROSSTALK
                             and r["eot_gap"] is not None and r["eot_gap"] >= 0.0
                             and r["duration"] >= floor.MIN_CLEAN_DUR
                             and r["word_rate"] >= floor.MIN_WORD_RATE),
              f"{rid}: clean={r['clean']}")
            # The two guarantees `_split_lapses` makes, checked on every row.
            # `held` is the other leg minus the absorbed backchannels -- a
            # backchannel inside a pause is correct and must not fail here.
            held = [(sp[0], sp[1]) for sp in f["spans"]
                    if sp[2] != side and own.get(sp, 0) > 0]
            for s in r["silence_spans"]:
                c("no pause hides a floor change",
                  vad.speech_between(held, s["start"], s["end"]) <= floor.MAX_PAUSE_CROSSTALK,
                  f"{rid}: other leg holds the floor across [{s['start']}, {s['end']}]")
                c("no pause is a lapse",
                  s["end"] - s["start"] <= floor.LAPSE_S + EPS,
                  f"{rid}: {s['end'] - s['start']:.2f}s pause exceeds LAPSE_S")

            c("speech_crosstalk matches the other leg, backchannels excluded",
              abs(r["speech_crosstalk"]
                  - vad.speech_between(held, r["start_time"], r["end_time"])) < EPS,
              f"{rid}: {r['speech_crosstalk']}")

            # a joined label must belong to this exact turn end
            if r["label"] is not None:
                want = f"{base}_{side[0]}_{int(round(r['end_time'] * 1000)):08d}"
                c("joined sid decodes to this turn end", r["sid"] == want,
                  f"{rid}: {r['sid']} != {want}")

    # ---- corpus-level ------------------------------------------------------
    c("all 116 calls present", len(by_call) == len(split),
      f"{len(by_call)} calls in the table, {len(split)} in split.json")
    for name, want in (("train", 71), ("dev", 15), ("test", 30)):
        got = len({r["conversation_id"] for r in rows if r["split"] == name})
        c("split call counts unchanged from the paper", got == want,
          f"{name}: {got} calls, expected {want}")

    ok = c.report()

    pool_rows = [r for r in rows if r["clean"]] if args.clean else rows
    if args.acoustic:
        _acoustic(pool_rows, args.acoustic)
    if args.listen:
        _cut_for_listening(pool_rows, args.listen, args.clean)
    return 0 if ok else 1


def _peak(r: dict) -> tuple[float, float] | None:
    side = "Left" if r["speaker_id"].endswith("_L") else "Right"
    wav = R1 / "Audio" / f"{r['conversation_id']}_{side}.wav"
    if r["trail_s"] <= 0.01:
        return None
    tail = audio.read_window(wav, r["end_time"], r["end_time"] + r["trail_s"])
    body = audio.read_window(wav, r["start_time"], r["end_time"])
    if not len(body) or not len(tail):
        return None
    return float(np.abs(tail).max()), float(np.abs(body).max())


def _acoustic(rows: list[dict], n: int) -> None:
    """How loud is the trailing window, relative to the turn's own speech?

    Reported, not asserted -- there is no correct threshold. A quiet trail is
    the expectation: the owner has stopped and the other leg is a separate
    file, so only ~2% crosstalk bleed should be there. What actually shows up
    is the decay of the owner's last syllable, which Silero cuts at thr_off
    while it is still audible. That is the right thing to ship -- it is what a
    deployed detector hears -- but it means a nonzero trail is normal, and a
    single loud sample is not evidence of a bug.
    """
    picks = random.Random(20260913).sample(rows, min(n, len(rows)))
    with Pool(12) as pool:
        res = [x for x in pool.map(_peak, picks) if x]
    ratio = sorted(t / b for t, b in res if b > 0)
    tp = sorted(t for t, _ in res)
    Q = lambda v, f: v[min(int(f * len(v)), len(v) - 1)]  # noqa: E731
    print(f"\nTRAILING WINDOW, {len(res)} sampled turns")
    print(f"  absolute peak      median {Q(tp, .5):.4f}  p90 {Q(tp, .9):.4f}  max {Q(tp, 1.0):.4f}")
    print(f"  vs the turn's own  median {Q(ratio, .5):.1%}  p90 {Q(ratio, .9):.1%}"
          f"  p99 {Q(ratio, .99):.1%}")
    for thr in (0.05, 0.10, 0.20):
        k = sum(1 for x in ratio if x > thr)
        print(f"  louder than {thr:.0%} of the turn's peak: {k:4d}/{len(ratio)} ({100 * k / len(ratio):4.1f}%)")


def _cut_for_listening(rows: list[dict], n: int, only_clean: bool = False) -> None:
    """Cut N random turns so a human can confirm the thing actually sounds like
    one speaker finishing. Peak amplitude in the trailing window is printed
    next to each: a loud trail means the window caught speech it should not.

    Listen to the `--clean` pool before shipping. A spot-listen of the full
    table will keep turning up fragments and pauses with the other speaker in
    them, because 73.6% of rows are not clean -- that is the point of the flag,
    not a defect."""
    out = LISTEN / ("clean" if only_clean else "all")
    out.mkdir(parents=True, exist_ok=True)
    picks = random.Random(20260913).sample(rows, min(n, len(rows)))
    print(f"\ncutting {len(picks)} turns to {out} ...")
    for r in picks:
        base = r["conversation_id"]
        side = "Left" if r["speaker_id"].endswith("_L") else "Right"
        wav = R1 / "Audio" / f"{base}_{side}.wav"
        x = audio.read_window(wav, r["start_time"], r["end_time"] + r["trail_s"])
        audio.write(out / f"{r['id']}.wav", x)
        tail = x[-int(max(r["trail_s"], 0.01) * audio.SR):]
        peak = float(np.abs(tail).max()) if len(tail) else 0.0
        print(f"  {r['id']}  {r['duration']:6.2f}s  eot {str(r['eot_gap']):>7s}  "
              f"pauses {len(r['silence_spans'])}  xtalk {r['pause_crosstalk']:.2f}  "
              f"{r['transcript'][:44]}")


if __name__ == "__main__":
    raise SystemExit(main())
