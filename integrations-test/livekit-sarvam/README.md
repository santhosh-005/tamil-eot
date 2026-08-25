# Live call test

Every number in `RESULTS.md` is **offline**, on pre-cut 8-second clips. This is
the rig that closes that gap — a Tamil voice agent you can actually talk to,
with the turn detector instrumented so the call produces numbers instead of
impressions.

Stack is all-Sarvam apart from the detector, which is the configuration their
own LiveKit guide says is unavailable: their per-language thresholds cover 14
languages, Tamil is not one, so the documented advice is `turn_detection="stt"`
— semantic turn detection **off**. This runs it on.

| | |
|---|---|
| `livekit_agent.py` | **primary.** Sarvam STT/LLM/TTS + `SmartTurnDetector` |
| `pipecat_bot.py` | same stack on Pipecat's own `LocalSmartTurnAnalyzerV3` |
| `turnlog.py` | records every prediction to JSONL |
| `analyze_call.py` | JSONL → the numbers |

---

## 1. Install

```bash
.venv/bin/pip install -r integrations-test/livekit-sarvam/requirements.txt
.venv/bin/pip install 'smart-turn-livekit[livekit]'
cp integrations-test/livekit-sarvam/.env.example integrations-test/livekit-sarvam/.env      # then fill in SARVAM_API_KEY
```

Weights download from HuggingFace on first use and are cached — nothing to set.
`SMART_TURN_MODEL=/path/to.onnx` pins a local file instead.

## 2. Check the chain before spending a call

```bash
python integrations-test/livekit-sarvam/livekit_agent.py --selftest
python integrations-test/livekit-sarvam/pipecat_bot.py   --selftest
```

No credentials, no microphone. Pushes real Tamil clips through the real
adapter in 20 ms frames and asserts predictions come out and get recorded.
Cheaper to find a broken model path here than with a Tamil speaker waiting.

## 3. Make the call

```bash
python integrations-test/livekit-sarvam/livekit_agent.py console      # local mic, no LiveKit server
```

`console` is the one to start with — no room, no cloud account, just your
microphone and speakers. When that works:

```bash
python integrations-test/livekit-sarvam/livekit_agent.py dev          # needs LIVEKIT_* in .env
```

then join from <https://agents-playground.livekit.io>. This is the one that
exercises real WebRTC — 48 kHz, network jitter, packet loss — which `console`
does not.

```bash
python integrations-test/livekit-sarvam/pipecat_bot.py                # opens a local WebRTC client
```

## 4. Record the control arm

A call with the detector on proves it **ran**. It does not prove it **helped**.
For that you need the same conversation without it:

```bash
python integrations-test/livekit-sarvam/livekit_agent.py --baseline 0.3  console   # fast, rude
python integrations-test/livekit-sarvam/livekit_agent.py --baseline 2.5  console   # safe, slow
python integrations-test/livekit-sarvam/livekit_agent.py                 console   # ours
```

**Without a detector the baseline is a dial, not a point.** LiveKit sets
`endpointing_delay = min_delay` and only moves it to `max_delay` when a detector
reports below the threshold, so with none attached every turn takes the same
fixed wait. Set it low and the agent talks over every mid-sentence pause; set it
high and every turn pays the full wait.

**The claim to support is that the detector sits off that line** — near the fast
arm's latency at the slow arm's interruption rate. Both ends have to be on the
chart or there is nothing to compare against.

Same speaker, same script, same order, all three times.

## 5. Read the result

```bash
python integrations-test/livekit-sarvam/analyze_call.py               # newest call
python integrations-test/livekit-sarvam/analyze_call.py --compare integrations-test/livekit-sarvam/calls/*.jsonl
```

---

## What a live call can and cannot tell you

**It cannot report accuracy.** Nobody labelled, per pause, whether the speaker
had actually finished. Any accuracy number from a live call is invented unless
someone labels the recording afterwards. `analyze_call.py` refuses to print one.
Accuracy comes from the 4,168-clip offline set; these are different kinds of
number and quoting them together is how a demo stops being credible.

What it does establish, and the offline set structurally cannot:

- **what the caller actually waited** — LiveKit's own `e2e_latency`, plus the
  endpointing / STT / LLM / TTS split. The detector moves **one** of those terms
- **how often the agent talked over an unfinished user** —
  `agent_false_interruption`, counted by the framework rather than by ear
- the detector **ran in the real path** rather than being silently bypassed
- **latency through the real adapter**, on live audio
- **probability distribution on live speech** vs the sealed test set — a
  pile-up around the threshold means the model is hedging on this audio
- **how much audio each prediction actually got** — see below
- how often the session took `max_delay` over `min_delay`

### Short buffers: the failure mode only a live call can show

`SmartTurnDetectorStream.flush()` clears the window at every turn boundary, so
the first predictions of each user turn run on a partial buffer, which
`features.py` right-pads with zeros to 8 s. Half a second of speech followed by
seven and a half seconds of silence is the strongest *"they have stopped"* cue
the model has — and it never saw one in training: **58 of 16,216** clips in
`data/gold/samples.jsonl` are shorter than 8 s (0.4%), and every other clip ends
0.20 s after the speech offset.

On the first recorded call this was not theoretical: **4 of 12 predictions ran
on 0.5–3.6 s of audio, and all 4 said `complete`** (median p 0.761) against 3 of
8 on full windows (median p 0.215). Small n, but the direction is exactly what
the mechanism predicts, and it biases toward interrupting a speaker who has only
just started. `analyze_call.py` reports it as its own section.

### Latency: do not compare these numbers to `RESULTS.md`

`RESULTS.md` quotes **83 ms** for tiny int8. That is model-only, batch 1, one
thread, idle box. The figure `analyze_call.py` prints is adapter end-to-end —
mel (~12 ms) plus ONNX plus the thread handoff — and lands around **120 ms** on
the same machine. Both are correct; they measure different spans. Compare live
to live.

A latency number without thread count and machine load attached is not a
measurement. This project published three wrong ones before adopting that rule.

---

## Two things that stop the detector running

**1. VAD silence threshold — fails loudly.** LiveKit requires
`min_silence_duration >= 0.25` (`MIN_SILENCE_DURATION_MS + 50` ms) and raises
at session start below it. A **floor, not a ceiling**: Silero's 0.55 default is
legal, just slower. `livekit_agent.py` pins 0.25, the fastest legal value.

**2. Language gating — fails silently.** `supports_language()` returns `False`
for anything the chosen model was not benchmarked on — Tamil only, for the Tamil
fine-tunes. If STT reports `en-IN`, the detector is skipped and the
agent quietly falls back to plain VAD timing — the exact behaviour this project
exists to replace. This is the one to watch for.

`analyze_call.py` prints a diagnosis when a log has zero predictions.

---

## Recording notes while you talk

`agent_false_interruption` catches the cases LiveKit itself notices. For
anything it does not — a wait that felt too long, a reply that landed on a
breath — mark it as it happens:

```python
recorder.note("cut me off mid-sentence")
```

`analyze_call.py` prints notes with timestamps against the prediction stream,
so a complaint lines up with the probability that caused it.

---

## Which operating point to test

Run the same script twice and diff the reports. The threshold is the actual
product decision, and it means different things per framework:

| | Pipecat | LiveKit |
|---|---|---|
| what the decision does | **ends the turn** | picks `min_delay` vs `max_delay` |
| cost of a false `complete` | talks over the user | a shorter pause |
| default here | `polite` | `inherited` (0.5) |

That asymmetry is why the two bots ship different defaults.
