"""Record what the turn detector actually did on a live call.

Every accuracy number in this project is offline, on pre-cut 8-second clips.
A live call cannot reproduce those — there are no labels — but it can answer
the questions the offline set structurally cannot:

  * does the model run in the real path at all, or is it silently bypassed?
  * what is inference latency on live audio, not on a benchmark loop?
  * how long after the user stops speaking does a prediction land?
  * what do the probabilities look like on real speech, versus the test set?

Four sources are recorded, and they are not the same thing:

  `prediction`   ours, from the adapter's `on_prediction` hook. Ground truth
                 for what the model computed.
  `session`      LiveKit's own `"eot prediction"` debug line, scraped from its
                 logger. This is what the *session* did with our number — which
                 endpointing delay it picked, and what triggered the call.
                 Recorded separately because a detector that computes a good
                 probability the session then ignores is a real failure mode,
                 and only the second source can see it.
  `transcript`   what was actually said, per turn. A probability of 0.94 at
                 t=12.3 s is unfalsifiable on its own; next to the sentence it
                 scored, a Tamil speaker can check it in a second.
  `metric`       LiveKit's own timings — `end_of_utterance_delay`, LLM `ttft`,
                 TTS `ttfb`. Their sum is what the user actually waits for, and
                 the detector only moves the first term. Recording all three is
                 what stops an 80 ms model win being reported as if the caller
                 felt it.

Plus `interruption`: `agent_false_interruption` fires when the agent started
replying and the user had not finished. That is the error this model exists to
prevent, counted by the framework rather than by listening to a recording.

`EotPredictionEvent` would be the clean way to get `session`, but in
livekit-agents 1.7.0 it is routed only to a remote session host — the source
carries a TODO to make it public. The log line is the stable surface today.

    from turnlog import TurnRecorder
    rec = TurnRecorder("calls/run1.jsonl", meta={"variant": "tiny"})
    detector = SmartTurnDetector(model="smart-turn-tamil-tiny", on_prediction=rec)
    rec.install_livekit_hook()
    ...                              # after session.start():
    rec.install_session_hooks(session)
    ...
    rec.close()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any


class TurnRecorder:
    """Append-only JSONL of every end-of-turn decision on a call.

    Safe to call from the inference thread: appends are serialised under a lock
    and flushed immediately, so a crash mid-call still leaves a readable file.
    Immediate flush costs a syscall per pause — a few dozen over a whole call,
    against losing the entire record if the agent dies. Worth it.
    """

    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")
        self._n = {"prediction": 0, "session": 0}
        self._t0 = time.time()
        self._handler: logging.Handler | None = None
        self._write("meta", {"started_at": self._t0, **(meta or {})})

    # -- writing ------------------------------------------------------------
    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")
            self._fh.flush()
            if kind in self._n:
                self._n[kind] += 1

    def __call__(self, pred) -> None:
        """The `on_prediction` hook. Signature matches `core.Prediction`."""
        self._write("prediction", {
            "at": pred.at,
            "t": pred.at - self._t0,
            "probability": round(pred.probability, 6),
            "threshold": pred.threshold,
            "complete": pred.complete,
            "inference_s": round(pred.inference_s, 6),
            "audio_s": round(pred.audio_s, 3),
        })

    def note(self, text: str, **kw: Any) -> None:
        """Free-form marker — use it to tag an interruption while on the call."""
        self._write("note", {"t": time.time() - self._t0, "text": text, **kw})

    # -- LiveKit's own view -------------------------------------------------
    def install_livekit_hook(self) -> None:
        """Capture LiveKit's `"eot prediction"` debug line.

        Needs the `livekit.agents` logger at DEBUG. The handler filters on the
        exact message, so it does not depend on parsing formatted output — and
        it attaches nothing to the record beyond what LiveKit already puts in
        `extra`, so a field appearing or vanishing across versions shows up as
        a missing key rather than a crash.
        """
        rec = self
        WANTED = ("probability", "unlikely_threshold", "endpointing_delay",
                  "language", "trigger", "from_cache")

        class _Hook(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.getMessage() != "eot prediction":
                    return
                try:
                    rec._write("session", {
                        "t": time.time() - rec._t0,
                        **{k: getattr(record, k, None) for k in WANTED},
                    })
                except Exception:      # a logging handler must never raise
                    pass

        lg = logging.getLogger("livekit.agents")
        lg.setLevel(logging.DEBUG)
        self._handler = _Hook()
        self._handler.setLevel(logging.DEBUG)
        lg.addHandler(self._handler)

    # -- the rest of the call ------------------------------------------------
    # Per-turn timings, off `ChatMessage.metrics`. The `metrics_collected`
    # event carries the same numbers but is deprecated in livekit-agents 1.7.0
    # ("use ChatMessage.metrics for per-turn latency"), and this source is the
    # better one anyway: it arrives already attached to the turn it describes,
    # so a latency never has to be matched back to a sentence by timestamp.
    #
    # A whitelist rather than the whole dict: the report also carries provider
    # request ids and per-provider metadata that would triple the log, and a
    # field renamed in a version bump should surface as an absent key rather
    # than a crash.
    _METRIC_FIELDS = (
        # user message -- the endpointing term, which is the one we move
        "end_of_turn_delay",
        "transcription_delay",
        "on_user_turn_completed_delay",
        # assistant message -- what the caller actually experienced
        "e2e_latency",          # user stopped speaking -> agent started
        "llm_node_ttft",
        "llm_node_ttfs",
        "tts_node_ttfb",
        "playback_latency",
        "started_speaking_at",
        "stopped_speaking_at",
    )

    def install_session_hooks(self, session: Any) -> None:
        """Subscribe to the session events the JSONL cannot get any other way.

        Call it **after** `session.start()`. Everything here is optional: each
        subscription is attempted separately and a name LiveKit has renamed is
        skipped rather than fatal, because losing one channel of instrumentation
        must not lose the call.

        What each one is for:

        `user_input_transcribed`  the STT `language` code. `supports_language()`
            returning False on a non-Tamil code is the one failure that disables
            the detector *silently* — this is the only place the code is visible.
        `conversation_item_added` the turn text, so a probability can be read
            next to the sentence that produced it.
        `agent_false_interruption` the agent replied over an unfinished user.
            **This is the error the model exists to prevent**, counted by the
            framework instead of by listening to a recording afterwards.
        `metrics_collected`  `end_of_utterance_delay` + `ttft` + `ttfb` is what
            the caller actually waits for. Without the last two, an 80 ms win in
            the first gets reported as though the user felt it.
        """
        def sub(name: str, fn) -> None:
            def guarded(ev: Any) -> None:
                try:
                    fn(ev)
                except Exception:            # noqa: BLE001 — never break a call
                    logging.getLogger(__name__).debug("hook %s failed", name,
                                                      exc_info=True)
            try:
                session.on(name, guarded)
            except Exception:                # noqa: BLE001 — event renamed/gone
                logging.getLogger(__name__).debug("cannot subscribe %s", name,
                                                  exc_info=True)

        def stamp(ev: Any) -> dict:
            return {"t": time.time() - self._t0}

        def on_user_stt(ev: Any) -> None:
            if not getattr(ev, "is_final", True):
                return                        # interim hypotheses, many per turn
            self._write("transcript", {**stamp(ev), "role": "user",
                                       "text": getattr(ev, "transcript", ""),
                                       "language": getattr(ev, "language", None),
                                       "source": "stt"})

        def on_item(ev: Any) -> None:
            item = getattr(ev, "item", None)
            if item is None:
                return
            content = getattr(item, "content", None)
            text = " ".join(c for c in (content or []) if isinstance(c, str)) \
                if isinstance(content, list) else str(content or "")
            role = str(getattr(item, "role", "?"))
            self._write("transcript", {**stamp(ev), "role": role, "text": text,
                                       "interrupted": getattr(item, "interrupted", None),
                                       "source": "item"})
            # MetricsReport is a TypedDict -- a plain dict at runtime, and
            # `total=False`, so every key is optional and absence is normal
            # (a user message has no e2e_latency, an interrupted reply has no
            # ttfb). Take what is there.
            mr = getattr(item, "metrics", None) or {}
            row = {k: mr[k] for k in self._METRIC_FIELDS
                   if isinstance(mr.get(k), (int, float))}
            if row:
                self._write("metric", {**stamp(ev), "role": role, **row})

        def on_false_interruption(ev: Any) -> None:
            # `.message` is deprecated in 1.7.0 and reading it emits a warning,
            # so only `resumed` is taken -- whether the agent recovered on its
            # own or the turn was actually lost.
            self._write("interruption", {**stamp(ev),
                                         "resumed": getattr(ev, "resumed", None)})

        def on_overlap(ev: Any) -> None:
            self._write("overlap", {**stamp(ev),
                                    "is_interruption": getattr(ev, "is_interruption", None),
                                    "agent_ended": getattr(ev, "agent_ended", None),
                                    "probability": getattr(ev, "probability", None)})

        def on_error(ev: Any) -> None:
            err = getattr(ev, "error", None)
            self._write("error", {**stamp(ev),
                                  "source": str(getattr(ev, "source", "") or "")[:80],
                                  "recoverable": getattr(err, "recoverable", None),
                                  "error": str(err or "")[:300]})

        sub("user_input_transcribed", on_user_stt)
        sub("conversation_item_added", on_item)      # text *and* per-turn timings
        sub("agent_false_interruption", on_false_interruption)
        sub("overlapping_speech", on_overlap)
        # A turn that commits and never gets answered looks identical in every
        # other channel to a turn nobody spoke on. The first detector call left
        # 54 s of dead air with nothing in the log to explain it -- this is the
        # channel that would have.
        sub("error", on_error)

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        if self._handler is not None:
            logging.getLogger("livekit.agents").removeHandler(self._handler)
            self._handler = None
        self._write("end", {"t": time.time() - self._t0, "counts": dict(self._n)})
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "TurnRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"TurnRecorder({self.path.name}, {self._n['prediction']} predictions, "
                f"{self._n['session']} session events)")
