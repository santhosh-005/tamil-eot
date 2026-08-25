#!/usr/bin/env python3
"""Tamil voice agent on Pipecat — Sarvam STT/LLM/TTS + our Tamil turn analyzer.

The LiveKit agent (`livekit_agent.py`) is the primary live test; this is the
same stack on the other framework, to show the weights are not LiveKit-specific.

**There is no adapter on this side, deliberately.** `pipecat-ai` already ships
`LocalSmartTurnAnalyzerV3`, which loads any Smart Turn ONNX file — so using our
model here is one `smart_turn_model_path` argument and nothing else.

**The threshold means something different here.** Under Pipecat the analyzer's
decision *ends the turn* — a false `complete` talks over the user; under LiveKit
it only chooses how long to wait. Same model, strictly worse consequences for
being wrong. Pipecat hardcodes 0.5, so that trade is not tunable from here.

    # local WebRTC, opens a browser client, no Daily account:
    python integrations-test/livekit-sarvam/pipecat_bot.py

    # verify the chain with no credentials and no audio device:
    python integrations-test/livekit-sarvam/pipecat_bot.py --selftest

Needs `pipecat-ai[webrtc,runner,sarvam,silero]`. Predictions land in
`integrations-test/livekit-sarvam/calls/pipecat-<timestamp>.jsonl`; read them with `analyze_call.py`.
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

MODEL = os.getenv("TAMIL_EOT_MODEL", "smart-turn-tamil-tiny")

INSTRUCTIONS = (
    "நீங்கள் ஒரு உதவியாளர். தமிழில் மட்டும், சுருக்கமாக பதிலளிக்கவும். "
    "ஒவ்வொரு பதிலும் இரண்டு வாக்கியங்களுக்கு மிகாமல் இருக்க வேண்டும். "
    "(Reply only in Tamil, at most two sentences, and end with a short "
    "question so there are turns to detect.)"
)


def build_analyzer():
    """Pipecat's own analyzer, pointed at our weights.

    There is no adapter for Pipecat and there should not be: `pipecat-ai` ships
    `LocalSmartTurnAnalyzerV3`, which already loads any Smart Turn ONNX file.
    `smart_turn_livekit.resolve_model` fetches ours from HuggingFace and returns
    the cached path; everything after that is stock Pipecat.

    Two things this costs, both accepted:

    * **Pipecat hardcodes the 0.5 threshold.** Under Pipecat the decision *ends
      the turn*, so `polite` would be worth more here than under LiveKit — but
      changing it means subclassing `_predict_endpoint`, and an example is the
      wrong place to ship a second adapter.
    * **No `on_prediction` hook**, so this side records no per-turn log. The
      LiveKit agent is the instrumented one.
    """
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from smart_turn_livekit import resolve_model

    return LocalSmartTurnAnalyzerV3(
        # The ONNX **file**, not its directory. Passing the directory gets
        # you `INVALID_PROTOBUF: Protobuf parsing failed` from onnxruntime,
        # which reads like a corrupt download rather than a wrong argument.
        smart_turn_model_path=str(resolve_model(MODEL)),
    )


async def run_bot(transport, recorder: TurnRecorder) -> None:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.sarvam.llm import SarvamLLMService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.transcriptions.language import Language

    key = os.environ["SARVAM_API_KEY"]
    stt = SarvamSTTService(api_key=key, settings=SarvamSTTService.Settings(
        model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"), language=Language.TA_IN))
    tts = SarvamTTSService(api_key=key, settings=SarvamTTSService.Settings(
        model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"), language=Language.TA_IN))
    llm = SarvamLLMService(api_key=key, settings=SarvamLLMService.Settings(
        model=os.getenv("SARVAM_LLM_MODEL_PIPECAT", "sarvam-105b")))

    ctx = OpenAILLMContext([{"role": "system", "content": INSTRUCTIONS}])
    agg = llm.create_context_aggregator(ctx)

    task = PipelineTask(
        Pipeline([transport.input(), stt, agg.user(), llm, tts,
                  transport.output(), agg.assistant()]),
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connect(_t, _c):
        await task.queue_frames([ctx.get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_t, _c):
        recorder.note("client disconnected")
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        recorder.close()


async def bot(runner_args) -> None:
    """Entrypoint for `pipecat.runner`."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    recorder = TurnRecorder(
        HERE / "calls" / f"pipecat-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
        meta={"framework": "pipecat", "model": MODEL},
    )
    analyzer = build_analyzer()
    print(f"\n  turn analyzer: {MODEL} @ Pipecat's fixed 0.5 threshold")
    print(f"  recording to:  {recorder.path}\n")

    params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        turn_analyzer=analyzer,
    )
    transport = await create_transport(runner_args, {
        "webrtc": lambda: params, "daily": lambda: params,
    })
    await run_bot(transport, recorder)


# --------------------------------------------------------------------------
def selftest() -> int:
    """Drive the analyzer through Pipecat's own buffering, no credentials.

    Deliberately uses `append_audio` in 20 ms frames rather than calling
    `_predict_endpoint` directly: the buffering and speech bookkeeping between
    those two is where a live pipeline actually differs from a test set, and
    calling the inner method would skip exactly that.
    """
    import asyncio

    import numpy as np

    clips = sorted((HERE.parent / "data" / "clips" / "test").glob("*.wav"))[:4]
    if clips:
        import soundfile as sf
        waves = [sf.read(str(c), dtype="int16")[0] for c in clips]
        src = f"{len(waves)} real Tamil test clips"
    else:
        rs = np.random.RandomState(0)
        waves = [(rs.randn(16_000 * 4) * 3000).astype(np.int16) for _ in range(4)]
        src = "synthetic noise (research clips not present)"

    out = HERE / "calls" / "pipecat-selftest.jsonl"
    out.unlink(missing_ok=True)
    rec = TurnRecorder(out, meta={"framework": "pipecat", "selftest": True, "source": src,
                                  "model": MODEL})
    try:
        a = build_analyzer()
    except (OSError, ImportError) as e:
        print(f"FAIL: {e}")
        return 1
    a.set_sample_rate(16_000)
    print(f"  analyzer : {MODEL} @ Pipecat's fixed 0.5 threshold")
    print(f"  audio    : {src}")

    async def run():
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
        for w in waves:
            a.clear()
            for i in range(0, len(w) - 320, 320):
                a.append_audio(w[i:i + 320].tobytes(), is_speech=True)
            state, _ = await a.analyze_end_of_turn()
            assert state in (EndOfTurnState.COMPLETE, EndOfTurnState.INCOMPLETE), state
            print(f"    {state.name}")

    asyncio.run(run())
    rec.close()

    rec.note(f"selftest: {len(waves)} clips through Pipecat's analyzer")
    # No prediction count to check: Pipecat's analyzer has no `on_prediction`
    # hook, so nothing lands in the log. What this proves is that our weights
    # load into *their* analyzer and reach a decision through their buffering,
    # which is the only claim this file makes.
    print(f"\n  OK — {len(waves)} clips decided. Our ONNX loads into Pipecat's")
    print("  own analyzer and reaches a verdict through its buffering.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args, rest = ap.parse_known_args()

    if args.selftest:
        return selftest()

    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass
    if not os.getenv("SARVAM_API_KEY"):
        print("missing env: SARVAM_API_KEY — see integrations-test/livekit-sarvam/.env.example")
        return 2

    from pipecat.runner.run import main as runner_main

    sys.argv = [sys.argv[0], *rest]
    runner_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
