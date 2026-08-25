#!/usr/bin/env python3
"""Tamil voice agent on LiveKit — Sarvam STT/LLM/TTS + our Tamil turn detector.

This is the live test. Every number in `RESULTS.md` is offline, on pre-cut 8 s
clips; nothing here has run against a real microphone, real network jitter and
real VAD timing. That gap is what this closes.

The stack is deliberately all-Sarvam apart from the turn detector, because that
is exactly the configuration their own LiveKit guide says is not available:

> "the semantic turn detector ... resolves to a local model whose per-language
>  thresholds cover 14 languages (Hindi included), falling back to an
>  English-calibrated threshold for everything else."

Tamil is not in the 14, so their recommendation is `turn_detection="stt"` —
semantic turn detection off. This runs it on.

    # no LiveKit server, no room, just your microphone:
    python integrations-test/livekit-sarvam/livekit_agent.py console

    # against LiveKit Cloud, joinable from the Agents Playground:
    python integrations-test/livekit-sarvam/livekit_agent.py dev

    # verify the whole chain with no credentials and no audio device:
    python integrations-test/livekit-sarvam/livekit_agent.py --selftest

Predictions land in `integrations-test/livekit-sarvam/calls/<timestamp>.jsonl`. Read them with
`python integrations-test/livekit-sarvam/analyze_call.py`.

## The baseline arm

A call with the detector on proves it ran. It does not prove it helped — for
that you need the same conversation without it, which is what `--baseline`
records:

    python integrations-test/livekit-sarvam/livekit_agent.py --baseline 0.3  console   # fast, rude
    python integrations-test/livekit-sarvam/livekit_agent.py --baseline 2.5  console   # safe, slow
    python integrations-test/livekit-sarvam/livekit_agent.py                 console   # ours
    python integrations-test/livekit-sarvam/analyze_call.py --compare integrations-test/livekit-sarvam/calls/*.jsonl

**Without a turn detector the baseline is a dial, not a point.** LiveKit sets
`endpointing_delay = min_delay` and only moves it to `max_delay` when a
detector reports a probability below the threshold
(`audio_recognition.py`) — so with no detector *every* turn gets the same
fixed wait. Set it low and the agent talks over every mid-sentence pause; set
it high and every turn pays the full wait. The claim this demo has to support
is that the detector gets the interruption rate of the high setting at close
to the latency of the low one, so both ends of the dial have to be on the
chart.

`--baseline` uses `turn_detection="stt"`, which is both what Sarvam's guide
recommends and the only mode that shares a commit path with the detector arm —
see `BASELINE_MODE` for what a `vad` control got wrong. `TAMIL_EOT_BASELINE_MODE`
overrides it.

Env: copy `.env.example` to `.env` and fill it in.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from turnlog import TurnRecorder  # noqa: E402

# Module scope, not inside `entrypoint`. LiveKit runs the entrypoint on a job
# thread and its plugin registry rejects that with
#
#     RuntimeError: Plugins must be registered on the main thread
#
# which surfaces as an unhandled job exception several screens into the log,
# after the worker has already reported itself healthy. Importing here also
# means the ~1.7 s `livekit.agents` import is paid at startup rather than on
# the first turn.
from livekit.plugins import sarvam, silero  # noqa: E402

# Tuning knobs, all env-overridable so a call can be re-run at a different
# operating point without editing code.
MODEL = os.getenv("TAMIL_EOT_MODEL", "smart-turn-tamil-tiny")
OPERATING_POINT = os.getenv("TAMIL_EOT_OPERATING_POINT", "inherited")
MIN_DELAY = float(os.getenv("TAMIL_EOT_MIN_DELAY", "0.3"))
MAX_DELAY = float(os.getenv("TAMIL_EOT_MAX_DELAY", "2.5"))

# LiveKit requires the VAD to report at least MIN_SILENCE_DURATION_MS + 50 ms
# = **0.25 s** before it will request a prediction, and refuses to start below
# that (`audio_recognition._check_vad_silence_requirement`). This is a floor,
# not a ceiling. 0.25 is the lowest legal value and therefore the lowest
# endpointing latency; Silero's 0.55 default is also fine, just slower.
VAD_MIN_SILENCE = 0.25

# Which fixed-delay path the control arm uses. **`stt`, to match the arm it is
# being compared against.**
#
# The first three-arm run used `vad` here, on the reasoning that it works with
# any STT and is the stronger control. That was the wrong axis: with a detector
# attached, 13 of 28 commits came through the STT path, and 9 of 17 turns had a
# transcript already in hand (`transcription_delay` 0 ms) when the turn
# committed. The `vad` control had none -- its `transcription_delay` p50 was
# 718 ms, tracking its endpointing p50 of 719 ms almost exactly, because the
# commit was waiting on saarika rather than on the configured 0.3 s.
#
# So that run's headline -- 446 ms endpointing for the detector against 719 ms
# for the control -- is a comparison between two different commit paths, and
# says nothing about the detector. A control has to sit on the same path.
# `stt` is also what Sarvam's own guide recommends, so it is the more faithful
# baseline anyway.
BASELINE_MODE = os.getenv("TAMIL_EOT_BASELINE_MODE", "stt")

INSTRUCTIONS = (
    "நீங்கள் ஒரு உதவியாளர். தமிழில் மட்டும், சுருக்கமாக பதிலளிக்கவும். "
    "ஒவ்வொரு பதிலும் இரண்டு வாக்கியங்களுக்கு மிகாமல் இருக்க வேண்டும். "
    "(You are a helpful assistant. Reply only in Tamil, briefly — at most two "
    "sentences per reply. Ask a short follow-up question so the conversation "
    "keeps going, because a turn detector can only be tested on turns.)"
)

_recorder: TurnRecorder | None = None

# Set by `--baseline DELAY`: run the control arm with no detector at a fixed
# endpointing delay. None means the detector arm.
BASELINE_DELAY: float | None = None


def _call_log_path(arm: str) -> Path:
    return HERE / "calls" / f"{time.strftime('%Y%m%d-%H%M%S')}-{arm}.jsonl"


def build_detector(on_prediction=None):
    """The one piece of this file that is ours."""
    from smart_turn_livekit import SmartTurnDetector
    return SmartTurnDetector(
        model=MODEL,
        operating_point=OPERATING_POINT,
        on_prediction=on_prediction,
        # One thread: a server carrying concurrent calls cannot hand 12 cores
        # to each of them, and 1 thread is the column RESULTS.md quotes.
        cpu_count=1,
    )


async def entrypoint(ctx) -> None:
    from livekit.agents import Agent, AgentSession

    baseline = BASELINE_DELAY is not None
    arm = f"baseline{BASELINE_DELAY:g}" if baseline else "detector"

    global _recorder
    _recorder = TurnRecorder(_call_log_path(arm), meta={
        "arm": arm,
        "model": None if baseline else MODEL,
        "operating_point": None if baseline else OPERATING_POINT,
        # Both arms report the two delays, so the analyser never has to guess
        # which one a turn took. In the baseline they are equal by
        # construction: with no detector LiveKit never leaves `min_delay`.
        "min_delay": BASELINE_DELAY if baseline else MIN_DELAY,
        "max_delay": BASELINE_DELAY if baseline else MAX_DELAY,
        "baseline_mode": BASELINE_MODE if baseline else None,
        "vad_min_silence": VAD_MIN_SILENCE,
        "stt": os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
        "tts": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
        "llm": os.getenv("SARVAM_LLM_MODEL", "sarvam-105b"),
    })
    _recorder.install_livekit_hook()

    if baseline:
        turn_detection: object = BASELINE_MODE
        endpointing = {"mode": "fixed",
                       "min_delay": BASELINE_DELAY, "max_delay": BASELINE_DELAY}
        print(f"\n  turn detector: NONE — baseline arm, "
              f"`{BASELINE_MODE}` @ fixed {BASELINE_DELAY:g}s")
    else:
        turn_detection = build_detector(on_prediction=_recorder)
        endpointing = {"mode": "fixed", "min_delay": MIN_DELAY, "max_delay": MAX_DELAY}
        print(f"\n  turn detector: {turn_detection.model} "
              f"@ threshold {turn_detection.threshold}")
    print(f"  recording to:  {_recorder.path}\n")

    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE),
        stt=sarvam.STT(language="ta-IN", model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")),
        llm=sarvam.LLM(model=os.getenv("SARVAM_LLM_MODEL", "sarvam-105b")),
        tts=sarvam.TTS(target_language_code="ta-IN",
                       model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")),
        turn_handling={
            "turn_detection": turn_detection,
            # min_delay is what a confident `complete` gets; below the
            # threshold the session waits max_delay instead. That is the whole
            # mechanism -- the detector never ends a turn itself under LiveKit.
            "endpointing": endpointing,
        },
    )

    # Close the log when the job ends, so `analyze_call.py` gets a real call
    # duration instead of inferring it from the last prediction -- which
    # silently truncates every call that ends in silence.
    async def _close_log() -> None:
        if _recorder is not None:
            _recorder.close()

    ctx.add_shutdown_callback(_close_log)

    await ctx.connect()
    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)
    # After start(): the session must exist before its events can be subscribed.
    _recorder.install_session_hooks(session)
    await session.generate_reply(instructions="வணக்கம் சொல்லி, ஒரு கேள்வி கேளுங்கள்.")


# --------------------------------------------------------------------------
# selftest -- the whole chain, no credentials, no audio device
# --------------------------------------------------------------------------
def selftest() -> int:
    """Push real Tamil audio through the detector and check the plumbing.

    This is not an accuracy test — the plugin's own test suite covers
    that. It answers the question that actually blocks a live call: are the
    weights findable, does the adapter produce predictions, and does the
    recorder write a file you can analyse afterwards. Cheaper to find out here
    than with a Tamil speaker waiting on the line.
    """
    import asyncio

    import numpy as np

    clips = sorted((HERE.parent / "data" / "clips" / "test").glob("*.wav"))[:6]
    if clips:
        import soundfile as sf
        waves = [sf.read(str(c), dtype="float32")[0] for c in clips]
        src = f"{len(waves)} real Tamil test clips"
    else:
        rs = np.random.RandomState(0)
        waves = [rs.randn(16_000 * 4).astype(np.float32) * 0.05 for _ in range(6)]
        src = "synthetic noise (research clips not present)"

    # The recorder appends, deliberately — a crash mid-call must not lose the
    # record. That makes a fixed filename wrong here, so the selftest starts
    # clean rather than counting the previous run's rows.
    out = HERE / "calls" / "selftest.jsonl"
    out.unlink(missing_ok=True)
    rec = TurnRecorder(out, meta={"selftest": True, "source": src,
                                  "model": MODEL, "operating_point": OPERATING_POINT,
                                  "min_delay": MIN_DELAY, "max_delay": MAX_DELAY,
                                  "vad_min_silence": VAD_MIN_SILENCE})
    try:
        detector = build_detector(on_prediction=rec)
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"  detector : {detector.model} @ threshold {detector.threshold}")
    print(f"  audio    : {src}")

    class Frame:
        """The three attributes `push_audio` reads off an `rtc.AudioFrame`."""
        def __init__(self, x):
            self.data = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
            self.sample_rate, self.num_channels = 16_000, 1

    async def run():
        stream = detector.stream()
        for w in waves:
            # 20 ms frames, the way a transport actually delivers audio
            for i in range(0, len(w) - 320, 320):
                stream.push_audio(Frame(w[i:i + 320]))
            ev = await stream.predict()
            assert 0.0 <= ev.end_of_turn_probability <= 1.0, ev
            print(f"    p={ev.end_of_turn_probability:.3f}  "
                  f"{'complete' if ev.end_of_turn_probability > detector.threshold else 'incomplete':10s} "
                  f"inference={1000 * (ev.inference_duration or 0):.0f} ms")
            stream.flush()
        await stream.aclose()

    asyncio.run(run())
    rec.close()

    n = sum(1 for line in out.read_text().splitlines() if '"kind": "prediction"' in line)
    print(f"\n  {n} predictions recorded -> {out}")
    if n != len(waves):
        print(f"FAIL: expected {len(waves)} predictions, recorded {n}")
        return 1
    print("  OK — detector, adapter and recorder all work. "
          "Add credentials and run `console` for a real call.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="verify the chain offline, then exit")
    ap.add_argument("--baseline", type=float, metavar="DELAY",
                    help="control arm: no detector, fixed endpointing delay in "
                         "seconds. Run it twice (e.g. 0.3 and 2.5) — with no "
                         "detector the baseline is a dial, not a point.")
    args, rest = ap.parse_known_args()

    if args.selftest:
        return selftest()

    if args.baseline is not None:
        global BASELINE_DELAY
        BASELINE_DELAY = args.baseline

    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass

    missing = [k for k in ("SARVAM_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"missing env: {', '.join(missing)} — see integrations-test/livekit-sarvam/.env.example")
        return 2

    # DEBUG so the recorder's hook can see LiveKit's own eot line.
    logging.getLogger("livekit.agents").setLevel(logging.DEBUG)

    from livekit import agents

    sys.argv = [sys.argv[0], *rest]
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
