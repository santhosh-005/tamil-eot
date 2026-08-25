#!/usr/bin/env python3
"""Turn a recorded call into the numbers, and say plainly what it cannot show.

    python integrations-test/livekit-sarvam/analyze_call.py                      # newest call
    python integrations-test/livekit-sarvam/analyze_call.py integrations-test/livekit-sarvam/calls/*.jsonl
    python integrations-test/livekit-sarvam/analyze_call.py --compare integrations-test/livekit-sarvam/calls/*.jsonl
    python integrations-test/livekit-sarvam/analyze_call.py --md                 # paste into RESULTS.md

**A live call has no labels.** Nobody wrote down, per pause, whether the
speaker had actually finished — so this cannot report accuracy, and any number
here that looked like accuracy would be invented. What it does report is the
set of things the offline test set structurally cannot:

  * the detector ran in the real path (or did not — the commonest failure is a
    VAD `min_silence_duration` above 200 ms, which silently disables it)
  * **what the caller actually waited** — endpointing plus LLM plus TTS, of
    which the detector only moves the first term
  * **how often the agent talked over an unfinished user**, counted by LiveKit
    rather than by listening to a recording
  * latency through the real adapter, on live audio
  * what probabilities look like on live speech vs the sealed test set
  * **how much audio each prediction actually had** — the live path can hand
    the model half a second where every training clip had eight

One call proves the thing runs. It does not prove it helps: for that, record a
control arm with `livekit_agent.py --baseline` and read both with `--compare`.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALLS = HERE / "calls"

# Below this a prediction ran on a zero-padded buffer rather than a full
# window. Not 8.0 exactly: the buffer is assembled from 20 ms frames, so a
# "full" window lands a frame or two short of the nominal length.
FULL_WINDOW_S = 7.9


def load(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "path": path,
        "meta": next((r for r in rows if r["kind"] == "meta"), {}),
        "pred": [r for r in rows if r["kind"] == "prediction"],
        "sess": [r for r in rows if r["kind"] == "session"],
        "note": [r for r in rows if r["kind"] == "note"],
        "text": [r for r in rows if r["kind"] == "transcript"],
        "metric": [r for r in rows if r["kind"] == "metric"],
        "intr": [r for r in rows if r["kind"] in ("interruption", "overlap")],
        "err": [r for r in rows if r["kind"] == "error"],
        "t0": next((r.get("started_at") for r in rows if r["kind"] == "meta"), 0.0),
        "end": next((r for r in rows if r["kind"] == "end"), {}),
    }


def dead_air(d: dict) -> tuple[list[float], int, int]:
    """Gap from each committed user turn to the next time the agent spoke.

    `started_speaking_at` is an epoch on the assistant's metric row, so it has
    to be rebased onto the log's own clock. Worth the trouble: this is the only
    view of a turn that was accepted and then never answered, which every other
    channel records as though nothing happened.
    """
    t0 = d["t0"] or 0.0
    starts = sorted(r["started_speaking_at"] - t0 for r in d["metric"]
                    if r.get("role") == "assistant" and "started_speaking_at" in r)
    users = [t for t in d["text"] if t.get("role") == "user" and t.get("source") == "item"]
    gaps = []
    for u in users:
        nxt = [s for s in starts if s >= u["t"] - 0.5]
        if nxt:
            gaps.append(min(nxt) - u["t"])
    return gaps, len(users), len(starts)


def fragmentation(d: dict) -> tuple[float, int, int]:
    """User turns per uninterrupted stretch of user speech.

    **The working interruption metric.** `agent_false_interruption` reads 0 on
    every arm recorded so far, because it only fires when the agent actually
    started replying — and the commonest damage is quieter than that: the turn
    is committed mid-sentence, the user keeps going, and one utterance arrives
    as four. A control arm at 0.3 s produced runs of 5, 6, 5 and 7.

    1.0 means every stretch of speech became exactly one turn. Higher is worse.

    Confounded by dead air: if the agent is slow, the user keeps talking and the
    block grows. Read it next to the reply count, never alone.
    """
    t0 = d["t0"] or 0.0
    ev = [(t["t"], "u") for t in d["text"]
          if t.get("role") == "user" and t.get("source") == "item"]
    ev += [(r["started_speaking_at"] - t0, "a") for r in d["metric"]
           if r.get("role") == "assistant" and "started_speaking_at" in r]
    ev.sort()
    runs, cur = [], 0
    for _, kind in ev:
        if kind == "u":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    n = sum(runs)
    return (n / len(runs) if runs else 0.0), n, len(runs)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def field(rows: list[dict], name: str) -> list[float]:
    """Every value of one metric field, across whichever metric classes carry it."""
    return [r[name] for r in rows if isinstance(r.get(name), (int, float))]


def arm_of(d: dict) -> str:
    meta = d["meta"]
    if meta.get("arm"):
        return str(meta["arm"])
    return "detector" if d["pred"] else "baseline"


def duration(d: dict) -> float:
    if d["end"].get("t"):
        return float(d["end"]["t"])
    ts = [r["t"] for key in ("pred", "sess", "text", "metric") for r in d[key]
          if isinstance(r.get("t"), (int, float))]
    return max(ts) if ts else 0.0


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_latency(d: dict, add) -> None:
    """What the caller waited. The only number here a listener would notice."""
    m = d["metric"]
    stages = [("endpointing", "end_of_turn_delay", "**ours**"),
              ("transcription", "transcription_delay", "saarika"),
              ("LLM first token", "llm_node_ttft", "sarvam LLM"),
              ("TTS first byte", "tts_node_ttfb", "bulbul")]
    e2e = field(m, "e2e_latency")
    have = [(n, field(m, k), who) for n, k, who in stages if field(m, k)]
    if not (have or e2e):
        add("**No per-turn timings in this log.** `install_session_hooks` was")
        add("not called, so end-to-end response latency is unmeasured — the")
        add("detector latency below is a component of it, not a substitute.")
        add("")
        return

    add("### What the caller waited")
    add("")
    add("| stage | p50 | p95 | n |")
    add("|---|---|---|---|")
    for name, xs, who in have:
        add(f"| {name} — {who} | **{1000*pct(xs,50):.0f}** ms | "
            f"{1000*pct(xs,95):.0f} ms | {len(xs)} |")
    if e2e:
        add(f"| **total, user stops → agent speaks** | "
            f"**{1000*pct(e2e,50):.0f}** ms | {1000*pct(e2e,95):.0f} ms | {len(e2e)} |")
    add("")
    add("`e2e_latency` is LiveKit's own figure, not a sum of the rows above —")
    add("the stages overlap. Quote the total when claiming a user-perceptible")
    add("win and the endpointing row when claiming a model one; they are")
    add("different sizes and mixing them is dishonest.")
    add("")


def section_interruptions(d: dict, add) -> None:
    """The error this model exists to prevent, counted rather than listened for."""
    intr = d["intr"]
    false_int = [r for r in intr if r["kind"] == "interruption"]
    overlap = [r for r in intr if r["kind"] == "overlap"]
    # Committed items only. A user turn also appears as an STT row, and
    # counting both would halve the rate.
    turns = len([t for t in d["text"]
                 if t.get("role") == "user" and t.get("source") == "item"]) \
        or len(d["sess"])

    add("### Talking over the user")
    add("")
    if not d["metric"] and not intr:
        add("Not measured — no session hooks on this log.")
        add("")
        return
    add("| | |")
    add("|---|---|")
    add(f"| agent replied over an unfinished user | "
        f"**{len(false_int)}**{f' of {turns} turns' if turns else ''} |")
    if false_int:
        add(f"| …of which the agent recovered on its own | "
            f"{sum(1 for r in false_int if r.get('resumed'))} |")
    if overlap:
        real = sum(1 for r in overlap if r.get("is_interruption"))
        add(f"| overlapping speech, user cut the agent off | {real} of {len(overlap)} |")
    add("")
    add("First row is LiveKit's `agent_false_interruption` — it started")
    add("replying and the user had not finished. **That is the error this model")
    add("exists to prevent**, counted by the framework rather than by listening")
    add("to a recording afterwards.")
    add("")
    add("This is the number the baseline arm is for. A fast fixed delay will")
    add("score well on latency and badly here; the detector's claim is that it")
    add("holds this down *without* paying the slow arm's wait.")
    add("")


def section_dead_air(d: dict, add) -> None:
    """Turns that were accepted and then never answered.

    `e2e_latency` only exists for turns the agent *did* reply to, so a median
    of 3 s can sit happily on top of a call where a third of the turns got
    nothing at all. This is the section that catches that.
    """
    gaps, n_user, n_reply = dead_air(d)
    if not gaps and not d["err"]:
        return
    add("### Dead air")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| user turns committed | {n_user} |")
    add(f"| times the agent spoke | {n_reply} |")
    if gaps:
        add(f"| longest wait for a reply | **{max(gaps):.1f} s** |")
        add(f"| waits over 10 s | {sum(1 for g in gaps if g > 10)} of {len(gaps)} |")
    if d["err"]:
        add(f"| errors | **{len(d['err'])}** |")
    add("")
    if d["err"]:
        for e in d["err"][:5]:
            add(f"- `{e['t']:6.1f}s` {e.get('source','?')}: {e.get('error','')[:160]}")
        add("")
    elif gaps and max(gaps) > 10:
        add("**No errors recorded** — but that only means the `error` event was")
        add("not subscribed on this log, not that nothing failed. A committed")
        add("turn that never gets a reply is invisible in every other channel:")
        add("`e2e_latency` exists only for turns the agent *did* answer, so a")
        add("healthy median can sit on top of a call full of dropped turns.")
        add("")
    add("> This is not the detector. It hands the session a probability and the")
    add("> session picks a delay; everything after that is STT, LLM and TTS.")
    add("> It is here because a demo dies on dead air regardless of whose fault")
    add("> it is, and because a `total response` median computed only over")
    add("> answered turns will not show it.")
    add("")


def section_short_buffers(d: dict, add) -> None:
    """Predictions the live path fed less than a full window.

    Nothing else in the project can see this: every offline clip is exactly
    8 s, so the failure mode does not exist until the model is in a session.
    """
    pred = d["pred"]
    short = [p for p in pred if p.get("audio_s", 8.0) < FULL_WINDOW_S]
    if not short:
        return
    full = [p for p in pred if p.get("audio_s", 8.0) >= FULL_WINDOW_S]
    thr = pred[0]["threshold"]

    def rate(ps: list[dict]) -> str:
        if not ps:
            return "—"
        c = sum(1 for p in ps if p["probability"] > thr)
        return f"{c}/{len(ps)} ({100*c/len(ps):.0f}%)"

    add("### Short buffers — predictions on a padded window")
    add("")
    add(f"**{len(short)} of {len(pred)} predictions ran on less than a full 8 s "
        f"window** ({min(p['audio_s'] for p in short):.1f}–"
        f"{max(p['audio_s'] for p in short):.1f} s of audio).")
    add("")
    add("| window | n | said `complete` | median p |")
    add("|---|---|---|---|")
    add(f"| full (≥{FULL_WINDOW_S:g} s) | {len(full)} | {rate(full)} | "
        f"{st.median([p['probability'] for p in full]):.3f} |" if full else
        f"| full (≥{FULL_WINDOW_S:g} s) | 0 | — | — |")
    add(f"| **padded** (<{FULL_WINDOW_S:g} s) | {len(short)} | **{rate(short)}** | "
        f"{st.median([p['probability'] for p in short]):.3f} |")
    add("")
    add("**Why this matters, and why it is expected to skew `complete`.**")
    add("`features.py` right-pads a short buffer with zeros to 8 s, so half a")
    add("second of speech becomes half a second of speech followed by seven and")
    add("a half seconds of silence — the strongest 'this person has stopped'")
    add("cue the model has. Training saw almost none of it: **58 of 16,216**")
    add("clips in `data/gold/samples.jsonl` are shorter than 8 s (0.4%), and")
    add("every one of the rest ends 0.20 s after the speech offset.")
    add("")
    add("The buffers are short because `SmartTurnDetectorStream.flush()` clears")
    add("the window at every turn boundary, so **the first predictions of each")
    add("user turn run out of distribution** — and they skew toward `complete`,")
    add("which means toward interrupting a speaker who has only just started.")
    add("Two candidate fixes, both testable offline: keep a rolling window")
    add("across the boundary so the pad is real audio, or suppress predictions")
    add("below a minimum fill and let the session fall back.")
    add("")


def section_counterfactual(d: dict, add) -> None:
    """What a fixed delay would have cost on this same call.

    Derived, not measured — it assumes the conversation would have unfolded
    identically, which it would not. It is a sanity check on whether a control
    arm is worth recording, never a substitute for recording one.
    """
    meta, sess = d["meta"], d["sess"]
    mn, mx = meta.get("min_delay"), meta.get("max_delay")
    if not sess or mn is None or mx is None or mx <= mn:
        return
    delays = [s["endpointing_delay"] for s in sess
              if isinstance(s.get("endpointing_delay"), (int, float))]
    if not delays:
        return
    held = sum(1 for x in delays if x >= mx)
    n = len(delays)
    ours, safe = sum(delays), n * mx

    add("### What a fixed delay would have cost (derived)")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| this call, adaptive | {ours:.1f} s of endpointing over {n} turns |")
    add(f"| fixed at {mx:g} s (safe) | {safe:.1f} s — **{safe-ours:.1f} s slower**, "
        f"{(safe-ours)/n:.2f} s per turn |")
    add(f"| fixed at {mn:g} s (fast) | {n*mn:.1f} s, but **{held} turn(s) "
        f"committed while the user was still talking** |")
    add("")
    add("> Derived from this call's own decisions, assuming the conversation")
    add("> would have gone the same way with a different delay — it would not.")
    add("> Record the control arm (`--baseline`) and use `--compare`; this")
    add("> table only says whether that is worth the trouble.")
    add("")


def section_transcript(d: dict, add) -> None:
    """Probabilities next to the sentences that produced them."""
    rows = [t for t in d["text"] if t.get("text")]
    if not rows:
        return
    # A user turn is recorded twice -- once from the STT event, once from the
    # committed conversation item. Show the items (they carry both roles and
    # the `interrupted` flag) and keep the STT rows only for the language
    # check below, which is the one thing they alone can answer.
    items = [t for t in rows if t.get("source") == "item"]
    text = items or rows
    pred = d["pred"]
    add("### Turns")
    add("")
    add("| t | who | said | p at that moment |")
    add("|---|---|---|---|")
    for t in text[:40]:
        near = [p for p in pred if p["t"] <= t["t"]]
        p = f"{near[-1]['probability']:.3f}" if near else "—"
        body = t["text"].replace("|", "/").strip()
        add(f"| {t['t']:.1f}s | {t.get('role','?')} | {body[:90]} | "
            f"{p if t.get('role') == 'user' else ''} |")
    if len(text) > 40:
        add(f"| … | | {len(text)-40} more | |")
    add("")
    langs = {t.get("language") for t in rows if t.get("language")}
    if langs:
        add(f"STT reported language: {', '.join(sorted(str(l) for l in langs))}. "
            f"Anything but a Tamil code makes `supports_language()` return False "
            f"and silently disables the detector.")
        add("")


# --------------------------------------------------------------------------
def report(d: dict, md: bool) -> list[str]:
    meta, pred, sess, note = d["meta"], d["pred"], d["sess"], d["note"]
    L: list[str] = []
    add = L.append
    arm = arm_of(d)
    dur = duration(d)

    add(f"## {d['path'].name} — `{arm}`")
    add("")

    if arm.startswith("baseline"):
        add(f"**Control arm: no turn detector**, `{meta.get('baseline_mode','?')}` "
            f"endpointing fixed at {meta.get('min_delay','?')} s, "
            f"over {dur/60:.1f} min.")
        add("")
        add("LiveKit only leaves `min_delay` when a detector reports a probability")
        add("below the threshold, so with none attached every turn takes the same")
        add("wait. That is the whole of the baseline.")
        add("")
        section_latency(d, add)
        section_interruptions(d, add)
        section_dead_air(d, add)
        section_transcript(d, add)
        return L

    if not pred:
        add("**NO PREDICTIONS — the detector never ran.** In order of likelihood:")
        add("")
        add("1. `supports_language()` returned False — STT reported a non-Tamil")
        add("   code. This is the silent one, and by far the likeliest.")
        add("2. The detector was not actually passed to `turn_handling`.")
        add("3. Nobody spoke, or the VAD never fired.")
        add("")
        add("Not the VAD silence threshold — LiveKit raises at session start if")
        add("`min_silence_duration` is below 0.25, so that failure is never quiet.")
        add("")
        # Still worth printing: the call happened, it just happened without us,
        # which makes it an unintended baseline arm rather than a dead log.
        section_latency(d, add)
        section_interruptions(d, add)
        section_dead_air(d, add)
        section_transcript(d, add)
        return L

    lat = [p["inference_s"] * 1000 for p in pred]
    prob = [p["probability"] for p in pred]
    thr = pred[0]["threshold"]
    below = sum(1 for p in prob if p <= thr)

    add(f"`{meta.get('variant','?')}` @ threshold **{thr}** "
        f"(`{meta.get('operating_point','?')}`), "
        f"min_delay {meta.get('min_delay','?')} / max_delay {meta.get('max_delay','?')}, "
        f"VAD silence {meta.get('vad_min_silence','?')} s.")
    add("")
    add(f"**{len(pred)} predictions over {dur/60:.1f} min.**")
    add("")

    section_latency(d, add)
    section_interruptions(d, add)
    section_dead_air(d, add)

    add("### The detector itself")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| detector latency p50 / p95 / max | **{pct(lat,50):.0f}** / {pct(lat,95):.0f} / "
        f"{max(lat):.0f} ms |")
    add(f"| probability median | {st.median(prob):.3f} |")
    add(f"| said `complete` | {len(pred)-below} of {len(pred)} "
        f"({100*(len(pred)-below)/len(pred):.0f}%) |")
    add(f"| said `incomplete` | {below} ({100*below/len(pred):.0f}%) |")
    add("")
    add("**Computed, not necessarily used.** The model runs on every VAD pause;")
    add("the session only consumes the prediction that is current when it needs")
    add("a decision, and discards the rest. The `session` block below is what")
    add("actually changed the agent's behaviour.")
    add("")
    add("Latency is **adapter end-to-end** — mel plus ONNX plus the thread")
    add("handoff — not the model-only figure `bench.py` prints. Expect it above")
    add("the 83 ms in `RESULTS.md`; that one excludes the ~12 ms mel and the")
    add("executor. Compare live numbers to live numbers.")
    add("")

    # Confidence shape. A model that is never confident is as much a problem as
    # one that is always wrong, and the offline set cannot show live drift.
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    add("Probability distribution — a live call should look decisive at both")
    add("ends, like the test set. A pile-up in 0.4–0.6 means the model is")
    add("hedging on this audio and the threshold is doing all the work:")
    add("")
    add("| band | n | |")
    add("|---|---|---|")
    for lo, hi in bands:
        n = sum(1 for p in prob if lo <= p < hi or (hi == 1.0 and p == 1.0))
        add(f"| {lo:.1f}–{hi:.1f} | {n} | {'█' * round(24 * n / len(prob))} |")
    add("")

    section_short_buffers(d, add)

    if sess:
        mx, mn = meta.get("max_delay"), meta.get("min_delay")
        delays = [s.get("endpointing_delay") for s in sess if s.get("endpointing_delay") is not None]
        held = sum(1 for x in delays if mx is not None and x >= mx)
        blind = sum(1 for s in sess if s.get("probability") is None)
        cached = sum(1 for s in sess if s.get("from_cache"))
        trig: dict[str, int] = {}
        for s in sess:
            trig[str(s.get("trigger"))] = trig.get(str(s.get("trigger")), 0) + 1

        add(f"**The session used {len(sess)} of the {len(pred)} predictions.** This is")
        add("LiveKit's own view — proof the number changed behaviour, not just that")
        add("it was computed.")
        add("")
        add("| | |")
        add("|---|---|")
        add(f"| held the floor (`max_delay` {mx}) | **{held} of {len(delays)}** |")
        add(f"| replied fast (`min_delay` {mn}) | {len(delays)-held} |")
        add(f"| decided with no prediction | {blind} |")
        add(f"| served from cache | {cached} |")
        add(f"| triggers | {', '.join(f'{k} {v}' for k, v in sorted(trig.items()))} |")
        add("")
        if held == 0:
            add("> **The detector never extended a wait on this call.** Every")
            add("> prediction the session consumed was above the threshold, so the")
            add("> agent always took `min_delay`. That means the model ran and did")
            add("> no harm — but the behaviour it exists for, holding the floor")
            add("> through a mid-sentence pause, was never exercised. To test it,")
            add("> pause deliberately in the middle of a sentence and check the")
            add("> agent waits instead of jumping in.")
            add("")
        section_counterfactual(d, add)
    else:
        add("**No session events.** The predictions were computed but LiveKit's")
        add("`\"eot prediction\"` log line was never seen — either the hook was not")
        add("installed, or `livekit.agents` is not at DEBUG. The model ran; whether")
        add("the session used it is unverified.")
        add("")

    section_transcript(d, add)

    if note:
        add("Notes taken during the call:")
        add("")
        for n in note:
            add(f"- `{n['t']:6.1f}s` {n['text']}")
        add("")

    add("> **No accuracy here, by construction** — a live call carries no")
    add("> per-pause labels, and this one is a handful of turns. Accuracy comes")
    add("> from the 4,168-clip offline set; this says whether the thing behaves")
    add("> in a real session. Do not quote the two as if they were the same kind")
    add("> of number.")
    return L


def compare(ds: list[dict]) -> list[str]:
    """One row per arm. This is the table the demo is actually about."""
    L: list[str] = ["## Arms", ""]
    add = L.append
    add("**Ours:** endpointing. **Not ours:** everything after it. Both are")
    add("here so the second cannot be quietly credited to the first.")
    add("")
    add("| arm | mins | **endpointing p50** | **frag/utterance** | blocks | replies | "
        "LLM ttft p50 | total p50 | false-interrupt | longest wait |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    ttfts: list[float] = []
    for d in ds:
        m = d["metric"]
        eou, tot = field(m, "end_of_turn_delay"), field(m, "e2e_latency")
        ttft = field(m, "llm_node_ttft")
        ttfts += [pct(ttft, 50)] if ttft else []
        fi = sum(1 for r in d["intr"] if r["kind"] == "interruption")
        gaps, n_user, n_reply = dead_air(d)
        frag, _, blocks = fragmentation(d)
        add(f"| `{arm_of(d)}` | {duration(d)/60:.1f} | "
            f"{f'**{1000*pct(eou,50):.0f}** ms' if eou else '—'} | "
            f"{f'**{frag:.1f}**' if blocks else '—'} | {blocks} | {n_reply} | "
            f"{f'{1000*pct(ttft,50):.0f} ms' if ttft else '—'} | "
            f"{f'{1000*pct(tot,50):.0f} ms' if tot else '—'} | "
            f"{fi if (m or d['intr']) else '—'} | "
            f"{f'{max(gaps):.0f} s' if gaps else '—'} |")
    add("")
    add("**frag/utterance** = user turns per uninterrupted stretch of speech.")
    add("1.0 means every stretch became exactly one turn; higher means the agent")
    add("committed mid-sentence and one utterance arrived in pieces. It is here")
    add("because `false-interrupt` — LiveKit's own counter — reads 0 on every arm")
    add("recorded so far: it only fires when the agent actually started talking,")
    add("and the commoner damage is quieter than that. **Compare arms only where")
    add("`blocks` is similar**; it counts things the user tried to say, which is")
    add("the closest these logs get to a common denominator.")
    add("")

    # A comparison is only worth printing if the arms are comparable. These are
    # the three ways these runs stop being, checked rather than assumed.
    warn: list[str] = []
    if ttfts and max(ttfts) > 2 * min(ttfts):
        warn.append(f"**The LLM was {max(ttfts)/min(ttfts):.0f}× slower in one arm than "
                    f"another** ({1000*min(ttfts):.0f} → {1000*max(ttfts):.0f} ms p50). "
                    "`total p50` is mostly that, not the detector. Read the "
                    "endpointing column instead.")
    turns = [dead_air(d)[1] for d in ds]
    if turns and max(turns) > 1.5 * max(1, min(turns)):
        warn.append(f"**The arms are not the same conversation** — {min(turns)} to "
                    f"{max(turns)} user turns. Some of that is the effect being "
                    "measured (a fast arm chops one sentence into several turns), "
                    "and some is just a different chat. The two cannot be "
                    "separated from these logs.")
    # Whether two arms share a commit path is a question about the data, not
    # about the mode string: what matters is whether the transcript had landed
    # before the turn committed. Where it had not, the commit was waiting on
    # STT and the configured delay is not what was actually paid -- which makes
    # the endpointing column a comparison of LiveKit's plumbing.
    tds = [(arm_of(d), pct(field(d["metric"], "transcription_delay"), 50))
           for d in ds if field(d["metric"], "transcription_delay")]
    waiting = [a for a, v in tds if v > 0.1]
    if waiting and len(waiting) < len(tds):
        warn.append(f"**The arms did not commit turns the same way**: "
                    f"{', '.join(f'`{a}`' for a in waiting)} waited on the transcript "
                    f"(`transcription_delay` p50 > 100 ms) while the others already had "
                    "it. The endpointing column is then partly LiveKit's plumbing "
                    "rather than the configured delay.")
    gaps_all = [max(dead_air(d)[0] or [0]) for d in ds]
    if any(g > 10 for g in gaps_all):
        warn.append(f"**Some turns were never answered** — longest wait "
                    f"{max(gaps_all):.0f} s. `total p50` is computed only over turns "
                    "that *did* get a reply, so it looks healthy either way. See the "
                    "dead-air section per arm.")
    if warn:
        add("### Before quoting any of this")
        add("")
        for w in warn:
            add(f"- {w}")
        add("")

    add("Read it as a trade, not a win: a baseline set fast should beat the")
    add("detector on endpointing and lose on interruptions, and one set slow")
    add("should do the reverse. **The claim is the detector sits off that")
    add("line** — near the fast arm's latency at the slow arm's interruption")
    add("rate. If it does not, say so.")
    add("")
    add("Same speaker, same script, same order, or none of this holds. Turn")
    add("counts this small are an illustration, not a measurement.")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--md", action="store_true", help="markdown only, no header")
    ap.add_argument("--compare", action="store_true",
                    help="one row per call — the arms table, for the demo")
    a = ap.parse_args()

    files = a.files or sorted(
        (p for p in CALLS.glob("*.jsonl") if p.name != "selftest.jsonl"),
        key=lambda p: p.stat().st_mtime)[-(99 if a.compare else 1):]
    if not files:
        print(f"no call logs in {CALLS}. Run:\n"
              f"  python integrations-test/livekit-sarvam/livekit_agent.py console")
        return 1

    ds = []
    for f in files:
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            continue
        ds.append(load(f))

    out: list[str] = []
    if a.compare:
        out += compare(ds) + [""]
    for d in ds:
        out += report(d, a.md) + [""]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
