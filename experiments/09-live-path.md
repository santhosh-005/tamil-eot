# 09 — Does it survive the live path? ✅

Three separate questions, three separate measurements. Only the second one is
labelled, and only the second one is quotable as accuracy.

| | question | answer |
|---|---|---|
| 1 | does it run in a real agent at all? | ✅ LiveKit 1.7 + Pipecat 1.7, all-Sarvam stack |
| 2 | what does serving cost in accuracy? | ✅ −2.60 points, 92.2% coverage |
| 3 | does it beat a fixed timeout? | ⚠️ −35% fragmentation for +120 ms — directional; live arms cannot be paired |

---

## 1. ✅ It runs end-to-end

Complete Tamil voice agent, **LiveKit Agents 1.7**:

| | |
|---|---|
| turn detection | **`smart-turn-tamil`** tiny int8 |
| VAD | Silero, `min_silence_duration=0.25` |
| STT | Sarvam `saarika:v2.5` |
| LLM | Sarvam `sarvam-105b` |
| TTS | Sarvam `bulbul:v3` |

Also verified on **Pipecat 1.7** via `LocalSmartTurnAnalyzerV3` — same weights,
no adapter needed.

**Adapter latency ~120–155 ms** on live audio: mel (~12 ms) plus ONNX plus the
thread handoff, against the 83 ms inference-only figure. Different spans, both
right — never mix them.

Rig, selftests and the recording protocol: `integrations-test/livekit-sarvam/`.

---

## 2. ✅ What serving costs, measured against labels

**The problem with every accuracy number in this repo.** They all score a
**pre-cut clip**, ending exactly 0.20 s past the speech offset because
`rules.py` cut it there. **Nothing in production cuts a window that way.** Live, the window is whatever
the VAD hands over, whenever it decides speech stopped.

**Setup.** 500 labelled test boundaries replayed through the **real Silero VAD**
(`min_silence_duration=0.25`) and the **real `SmartTurnDetectorStream`** in 20 ms
frames. tiny int8 @ 0.5. No STT, LLM or TTS — none of them feed the prediction.

**Paired**: the same rows scored both ways, in the same run.

### Result

| | pre-cut clip | **live window** |
|---|---|---|
| accuracy | 86.12% | **83.51%** |
| false complete (talks over the user) | 24 | **30** |
| FP/N | 5.21% | **6.51%** |

**−2.60 points.** The paired view is the honest one:

```
identical verdict on 419 of 461 clips   90.9%
offline right / live wrong   27
live right / offline wrong   15     McNemar p = 0.0896
```

**Not significant at p < 0.05.** But the delta is stable across samples (−2.67
at n=187, −2.60 at n=461) and consistent in direction. Read it as *"about 2–3
points, probably real, small"* — **not as a measured constant.**

### Cause, measured

| | p10 | p50 | p90 |
|---|---|---|---|
| VAD close, relative to labelled offset | +0.26 s | **+0.29 s** | +0.32 s |

Clips stop at **+0.20 s**. A **90 ms window shift** — 90 ms more trailing
silence than training ever showed the model — and it barely moves it.

### Coverage 92.2%

| | n | share |
|---|---|---|
| VAD closed, model consulted | 461 | **92.2%** |
| VAD never closed — never asked | 39 | 7.8% |

`complete` 94%, `incomplete` 89%.

Those 39 are **not errors**. LiveKit will not request a prediction below
`min_silence_duration + 50 ms`, so they are outside the product.

This also explains a number that otherwise looks wrong: **the pre-cut column
reads 86.12% while the full split reads 83.35%.** The boundaries a VAD surfaces
are the easier ones.

### ✅ The paired design caught two harness bugs, both silently wrong

**1. A replay pump that outran the VAD.** Feeding audio to Silero and the
detector concurrently with no throttle let the buffer fill *past* the close
point, so the model saw audio a session would not have had yet — the next
speaker's, usually. **Cost 26 accuracy points** and looked exactly like a
live-serving catastrophe. Now two passes, sliced deterministically at
`samples_index`.

**2. Scoring against the wrong label file.** → [pitfalls](../docs/pitfalls.md).

> **A live-only number is unfalsifiable.** Both bugs were caught by scoring the
> same rows the known-good way *in the same run*: when the paired column failed
> to reproduce 84%, the harness was wrong, not the model.
>
> **Never report a new measurement against a figure remembered from a doc.**

### ⚠️ Short buffers — mechanism real, effect not established

`flush()` clears the window at every turn boundary; `features.py` right-pads to
8 s with zeros; **58 of 16,216** training clips are < 8 s. So padded windows
*are* out of distribution, and half a second of speech followed by 7.5 s of
silence is the strongest *"they have stopped"* cue the model has.

But the live `complete`-rate gap went **+62, +21, −1 points** over three calls.
Two looked damning, the third showed nothing.

**Do not quote it.** Measure offline by truncating test clips to 0.5/1/2/4 s,
where n=4,168 and labels exist. 🔍 Open.

---

## 3. Against fixed-timeout endpointing

Without a detector the baseline is a **dial, not a point**: LiveKit sets
`endpointing_delay = min_delay` and only moves to `max_delay` when a detector
reports below threshold. So both ends of the dial have to be on the chart.

Three arms, one speaker, one topic, 2026-08-25:

| arm | endpointing p50 | frag / utterance | blocks | replies |
|---|---|---|---|---|
| fixed 0.3 s | 301 ms | 3.7 | 7 | 7 |
| fixed 2.5 s | 2,501 ms | 1.0 | 5 | 5 |
| **detector** | **421 ms** | **2.4** | 7 | 8 |

`fixed 0.3` and `detector` produced **7 utterance blocks each** over 3.5 / 3.2
min — the closest pairing the run affords. **+120 ms of endpointing, −35%
fragmentation.** The detector held the floor on 11 of 26 decisions.

**Not a controlled test.** The LLM and TTS are non-deterministic, so each arm is
a different conversation on a different clock. Compare per-utterance rates, not
totals or wall-clock. The paired measurement is §2.

**Directional.** "Fast latency at the safe arm's interruption rate" would need
2.4 to be near 1.0, and it is not.

### ⚠️ Three things that make live numbers hard to read

**`agent_false_interruption` reads 0 on every arm.** It only fires once the
agent has actually started talking. The commoner damage — turn committed
mid-sentence, utterance arriving in five pieces — is invisible to it.
**`frag/utterance` is the working proxy.**

**`total p50` is unusable.** LLM ttft p50 was 2,770 / 5,335 / 3,374 ms across
the three arms and swung the other way on an earlier run. It measures the LLM
API, not the detector.

**Dead air on every arm**, including the fixed ones: 12/25, 0/4, 7/17 turns
waited > 10 s, worst 34 s. Causes named by the `error` channel — a TTS API
error, and LLM ttft p95 at 13.7 s. **Not the detector.** But a demo dies on it
anyway, so it is recorded here rather than left for someone to rediscover live.

**A control arm has to sit on the same commit path.** The first run put the
baselines on `turn_detection="vad"`, which waits on STT, while the detector arm
got a pre-landed transcript on half its turns. Its 446 vs 719 ms "win" was
plumbing, not model — thrown out and re-run with both on `stt`.

---

## What a live call can and cannot tell you

**It cannot report accuracy.** Nobody labelled, per pause, whether the speaker
had finished. Any accuracy number from a live call is invented unless someone
labels the recording afterwards — `analyze_call.py` refuses to print one.

What it does establish, and the offline set structurally cannot: what the caller
actually waited, whether the detector ran in the real path at all, latency
through the real adapter, and the probability distribution on live speech.

---

`reports/replay_live.md`, `integrations-test/livekit-sarvam/` ·
`python pipeline/22_replay_live.py`
