#!/usr/bin/env python3
"""Step 5 -- replay labelled test boundaries through the live turn-taking path.

Every accuracy number in `RESULTS.md` comes from scoring a pre-cut clip: the
window ends exactly 0.20 s after the speech offset, because that is how
`rules.py` cut it. **Nothing in production cuts a window that way.** Live, the
window is whatever LiveKit's VAD hands over, at whatever moment it decides
speech has stopped -- and it will not even ask below
`min_silence_duration + 50 ms`.

So the offline number answers "is the model right about this clip", and this
answers the two questions that actually decide whether it works in a call:

  1. **Coverage.** How many real boundaries does the live path ever ask about?
     27% of `incomplete` test cases have a gap under 0.30 s. LiveKit cannot
     query those at all -- the model is never consulted, so they are neither
     right nor wrong, they are simply outside the product.
  2. **Accuracy on what it does ask.** Same labels, same model, but the window
     comes from Silero rather than from the corpus transcript.

What is real here and what is not: this drives the **real** `silero.VAD` at the
**real** `min_silence_duration=0.25`, in **real** 20 ms frames, through the
**real** `SmartTurnDetectorStream`. It does not run STT, an LLM or TTS, because
none of them feed the prediction -- they would add API cost and nondeterminism
to a measurement that needs neither. This is the turn-taking path, not a whole
agent.

    python pipeline/22_replay_live.py --limit 100        # smoke, ~1 min
    python pipeline/22_replay_live.py                    # full test split

Writes `reports/replay_live.md` and `reports/replay_live.jsonl`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tamileot import audio, rules  # noqa: E402
from tamileot.paths import CLIPS, GOLD, MANIFEST, REPORTS  # noqa: E402

SR = audio.SR
FRAME = SR // 50                      # 20 ms, the way a transport delivers audio
LEAD_S = 8.0                          # the model's window
N_SAMPLES = int(LEAD_S * SR)
VAD_MIN_SILENCE = 0.25                # LiveKit's floor; see livekit_agent.py
MAX_TAIL_S = 2.5                      # how long to wait for the VAD to close


def frame(x: np.ndarray):
    """A real `rtc.AudioFrame`.

    A duck-typed stand-in works for our own `push_audio` and is what the
    selftest uses, but Silero's `push_frame` silently accepts it and emits
    **nothing at all** -- not even `INFERENCE_DONE`. A VAD that reports no
    events is indistinguishable from a VAD that heard no speech, which is how
    this script first produced a confident "0% coverage".
    """
    from livekit import rtc
    return rtc.AudioFrame(data=(np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(),
                          sample_rate=SR, num_channels=1, samples_per_channel=len(x))


def load_rows(split: str, core_only: bool) -> list[dict]:
    """The LLM labels, from `samples_llm.jsonl` -- **not** `samples.jsonl`.

    Both files carry a `label` field for the same `sid` and they agree on only
    **62.3%** of the test split. `samples.jsonl` holds the rule-derived label;
    the settled policy is `llm`, which is what the model was trained on and
    what `16_evaluate.py` scores against. Reading the wrong one here
    scored `tiny` at 58% instead of 84% and very nearly got published as a
    finding about live serving.
    """
    src = GOLD / "samples_llm.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("llm_ok") and r["split"] == split]
    return [r for r in rows if rules.is_core(r)] if core_only else rows


async def replay_one(row: dict, wav: Path, vad, detector) -> dict:
    """One boundary, start to finish, the way a session would see it.

    The audio runs from 8 s before the decision point to `MAX_TAIL_S` after it,
    so the VAD gets the same lead-in the model's window needs and enough tail
    to close a pause. Frames go to the VAD and the detector together, because
    that is what `AgentSession` does -- the detector buffers continuously and
    the VAD decides *when* the buffer gets read.
    """
    t = row["t"]
    x = audio.read_window(wav, t - LEAD_S, t + MAX_TAIL_S)

    # ---- pass 1: where does the VAD close? --------------------------------
    vs = vad.stream()
    closes: list[tuple[int, float]] = []

    async def pump() -> None:
        for i in range(0, len(x) - FRAME + 1, FRAME):
            vs.push_frame(frame(x[i:i + FRAME]))
            # Silero runs its inference on its own task. Without a yield this
            # loop starves it and the whole window is pushed before a single
            # frame is scored -- one way to get a silent 0% coverage.
            await asyncio.sleep(0)
        vs.flush()
        vs.end_input()

    async def listen() -> None:
        async for ev in vs:
            if str(getattr(ev.type, "value", ev.type)) == "end_of_speech":
                closes.append((int(ev.samples_index), float(ev.silence_duration or 0)))

    await asyncio.gather(pump(), listen())
    await vs.aclose()

    # ---- pass 2: score the boundary's own close ---------------------------
    # Two passes rather than one, because a single pass has to read the
    # detector's buffer at the moment the VAD event surfaces -- and here the
    # pump is unthrottled, so it has already run *past* the close and the
    # buffer holds audio the session would not have had yet. That is not a
    # subtle error: it dragged accuracy down 26 points by feeding the model
    # the next speaker's audio. Slicing at `samples_index` is exactly what a
    # real session's 8 s buffer would contain at that instant.
    #
    # Several pauses fall in one window and only one is the labelled boundary;
    # earlier closes belong to earlier boundaries, which are their own rows.
    idx = next((i for i, _ in closes if i / SR - LEAD_S >= -0.05), None)
    asked = None
    if idx is not None:
        sil = next(s for i, s in closes if i == idx)
        win = x[max(0, idx - N_SAMPLES):idx]
        ds = detector.stream()
        for i in range(0, len(win) - FRAME + 1, FRAME):
            ds.push_audio(frame(win[i:i + FRAME]))     # the real framing path
        buf = ds._buf.copy()
        await ds.aclose()
        asked = {"at": round(idx / SR - LEAD_S, 3),
                 "silence_s": round(sil, 3),
                 "audio_s": round(len(buf) / SR, 3),
                 "p": round(float(detector.predict_proba(buf)), 6)}

    # The same boundary scored the way `RESULTS.md` scores it: the pre-cut
    # clip, ending 0.20 s past the offset. Same row, same model, same
    # threshold, so the *only* variable left is the window -- which is the
    # whole point. Without this the live number has nothing to be a delta from.
    clip = CLIPS / row["split"] / f"{row['sid']}.wav"
    offline = float(detector.predict_proba(audio.read(clip))) if clip.exists() else None

    return {"sid": row["sid"], "label": row["label"], "gap": row["gap"],
            "source": row["source"], "n_closes": len(closes),
            "offline_p": round(offline, 6) if offline is not None else None,
            "asked": asked}


async def main_async(a: argparse.Namespace) -> int:
    from livekit.plugins import silero

    from smart_turn_livekit import SmartTurnDetector

    manifest = {m["base"]: m for m in json.loads((MANIFEST / "stereo_calls.json").read_text())}
    rows = load_rows(a.split, not a.include_heldback)
    if a.limit:
        # Shuffled, because the file is in call order: the first N rows are the
        # first few calls, and a smoke run on those reports whatever class
        # balance those speakers happened to have rather than the split's.
        import random
        random.Random(20260819).shuffle(rows)
        rows = rows[:a.limit]
    print(f"  {len(rows)} boundaries from '{a.split}'"
          f"{'' if a.include_heldback else ' (core only)'}")

    vad = silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE, sample_rate=SR)
    detector = SmartTurnDetector(model=a.model, operating_point=a.operating_point,
                                 cpu_count=a.threads)
    print(f"  detector {detector.model} @ {detector.threshold}, "
          f"VAD min_silence {VAD_MIN_SILENCE}s\n")

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        wav = Path(manifest[r["base"]]["wav"][r["side"]])
        out.append(await replay_one(r, wav, vad, detector))
        if i % 100 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el:.0f}s elapsed, "
                  f"{el/i*(len(rows)-i):.0f}s left", flush=True)

    await detector.aclose()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "replay_live.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    (REPORTS / "replay_live.md").write_text(
        report(out, detector.threshold, a), encoding="utf-8")
    print(f"\n  -> {REPORTS / 'replay_live.md'}")
    print()
    print(report(out, detector.threshold, a))
    return 0


def report(out: list[dict], thr: float, a: argparse.Namespace) -> str:
    L: list[str] = []
    add = L.append
    n = len(out)
    asked = [r for r in out if r["asked"]]
    add(f"# Live-path replay — `{a.model}` @ {thr}")
    add("")
    add(f"{n} labelled boundaries from `{a.split}`, replayed through the real "
        f"Silero VAD (`min_silence_duration={VAD_MIN_SILENCE}`) and the real "
        f"`SmartTurnDetectorStream` in 20 ms frames. No STT, LLM or TTS — none "
        f"of them feed the prediction.")
    add("")

    # -- coverage ----------------------------------------------------------
    add("## Coverage — what the live path never asks about")
    add("")
    add("| | n | share |")
    add("|---|---|---|")
    add(f"| boundaries | {n} | |")
    add(f"| **VAD closed, model consulted** | **{len(asked)}** | "
        f"{100*len(asked)/n:.1f}% |")
    add(f"| VAD never closed — never asked | {n-len(asked)} | "
        f"{100*(n-len(asked))/n:.1f}% |")
    add("")
    for lab, name in ((1, "complete"), (0, "incomplete")):
        sub = [r for r in out if r["label"] == lab]
        got = [r for r in sub if r["asked"]]
        if sub:
            add(f"- `{name}`: asked on {len(got)} of {len(sub)} "
                f"({100*len(got)/len(sub):.0f}%)")
    add("")
    add("A boundary the VAD never closes on is one the model is **never "
        "consulted about** — LiveKit will not request a prediction below "
        f"`min_silence_duration + 50 ms`. Those cases are not errors; they are "
        "outside the product. The offline test set counts them, which is why "
        "its accuracy and this one are not the same quantity.")
    add("")

    if not asked:
        return "\n".join(L)

    # -- accuracy: same rows, two windows ----------------------------------
    first = [(r["label"], r["asked"], r["offline_p"]) for r in asked]

    def score(get) -> tuple[float, int, int, int, int]:
        tp = sum(1 for l, q, o in first if l == 1 and get(q, o) > thr)
        tn = sum(1 for l, q, o in first if l == 0 and get(q, o) <= thr)
        fp = sum(1 for l, q, o in first if l == 0 and get(q, o) > thr)
        fn = sum(1 for l, q, o in first if l == 1 and get(q, o) <= thr)
        return 100 * (tp + tn) / len(first), tp, tn, fp, fn

    live = score(lambda q, o: q["p"])
    paired = all(o is not None for _, _, o in first)
    off = score(lambda q, o: o) if paired else None

    add("## Accuracy — the same boundaries, two windows")
    add("")
    add(f"Identical rows ({len(first)}), model and threshold. The only "
        "difference is where the window ends: the pre-cut clip stops 0.20 s "
        "after the labelled offset, the live one stops wherever Silero closed.")
    add("")
    add("| | pre-cut clip | **live window** |")
    add("|---|---|---|")
    if off:
        add(f"| accuracy | {off[0]:.2f}% | **{live[0]:.2f}%** |")
        add(f"| true complete | {off[1]} | {live[1]} |")
        add(f"| true incomplete | {off[2]} | {live[2]} |")
        add(f"| **false complete** (talks over the user) | {off[3]} | **{live[3]}** |")
        add(f"| false incomplete (waits too long) | {off[4]} | {live[4]} |")
        add(f"| FP/N | {100*off[3]/len(first):.2f}% | **{100*live[3]/len(first):.2f}%** |")
        add("")
        add(f"**Delta: {live[0]-off[0]:+.2f} points.** This is the cost of "
            "serving, isolated — same audio, same labels, same model, later "
            "window.")
    else:
        add(f"| accuracy | — | **{live[0]:.2f}%** |")
        add("")
        add("Pre-cut clips not on disk, so there is no paired column. Rebuild "
            "with `pipeline/05_split_extract.py` to get the delta.")
    add("")
    add("Neither column is the headline `RESULTS.md` number: that one is over "
        "the whole split, and this is over the subset the VAD surfaced.")
    add("")

    # -- what the window looked like ---------------------------------------
    offs = sorted(q["at"] for _, q, _ in first)
    fills = sorted(q["audio_s"] for _, q, _ in first)
    q50 = lambda xs: xs[len(xs) // 2]                                # noqa: E731
    add("## The window the model was handed")
    add("")
    add("| | p10 | p50 | p90 |")
    add("|---|---|---|---|")
    add(f"| VAD close, relative to the labelled offset | {offs[len(offs)//10]:+.2f} s | "
        f"**{q50(offs):+.2f} s** | {offs[9*len(offs)//10]:+.2f} s |")
    add(f"| audio in the buffer | {fills[len(fills)//10]:.2f} s | "
        f"**{q50(fills):.2f} s** | {fills[9*len(fills)//10]:.2f} s |")
    add("")
    add(f"Offline clips end **+0.20 s** past the labelled offset, by "
        f"construction (`rules.py: TRAIL_S`). The gap between that and the p50 "
        f"above is the train/serve skew this whole script exists to measure.")
    add("")
    short = sum(1 for f in fills if f < 7.9)
    add(f"{short} of {len(fills)} predictions ({100*short/len(fills):.0f}%) ran "
        f"on less than a full 8 s window, zero-padded by `features.py`.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="smart-turn-tamil-tiny")
    ap.add_argument("--operating-point", default="inherited")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--include-heldback", action="store_true",
                    help="also replay `change_midseg` / `hold_inter` rows")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
