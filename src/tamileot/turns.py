"""Turn structure of a stereo call, derived from acoustics rather than from
transcript timestamps.

The transcript segments are padded -- both legs' segments summed cover 107% of
the call wall clock, which is only possible if they include lead-in and
trailing silence. Using their edges as turn boundaries therefore inflates
apparent overlap and misplaces every gap. So the boundaries come from the VAD
speech spans, and the transcript is used for two things only:

  1. confirming a span is real speech on this leg and not crosstalk bleed from
     the other leg -- the transcriber only wrote down what they heard on the
     leg they were transcribing, so a span with no transcript over it is echo;
  2. supplying the text at the boundary, for the backchannel and
     sentence-final filters.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from . import vad
from .corpus import LEFT, RIGHT, SIDES, StereoCall, leg_reco, other
from .kaldi import Segment

# A VAD span must share at least this much time with a transcript segment on
# the same leg to count as that leg's speaker rather than bleed-through.
MIN_TRANSCRIPT_OVERLAP = 0.10


@dataclass(slots=True)
class Span:
    """One confirmed run of speech on one leg."""

    start: float
    end: float
    side: str
    segs: tuple[str, ...] = ()          # transcript utt ids this span touches
    text: str = ""                      # words apportioned to this span
    seg_text: str = ""                  # full text of the segments it touches
    words: float = 0.0                  # apportioned word count (fractional)
    inside_one_seg: bool = False        # wholly inside a single transcript segment

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class CallTimeline:
    base: str
    spans: list[Span]                    # both legs, sorted by start
    speech: dict[str, list[tuple[float, float]]]   # raw VAD spans per leg
    segs: dict[str, list[Segment]]
    dropped_bleed: int = 0
    dropped_bleed_time: float = 0.0
    stats: dict = field(default_factory=dict)

    def side_spans(self, side: str) -> list[Span]:
        return [s for s in self.spans if s.side == side]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _apportion(a: float, b: float, seg: Segment,
               iv: list[tuple[float, float]]) -> tuple[str, float]:
    """Words of `seg` that fall in [a, b], spread across the segment's *speech*
    rather than across its elapsed time.

    Interpolating over elapsed time is the obvious thing and it is wrong here.
    The segments carry lead-in, trailing silence and internal pauses: measured
    over 30 calls and both legs, the median segment is **21% silence** and the
    10th percentile is 53%. Linear interpolation therefore hands about a fifth
    of the words to intervals where nobody was speaking -- and it does that
    worst exactly where `hold_intra` lives, because a mid-segment pause *is*
    that silence. Weighting by VAD speech seconds removes the bias, and changes
    the final three words on 32% of the QA clips.

    Still an approximation: it assumes a constant rate *while speaking*, which
    only real word times from forced alignment (step 1e) would fix. But it no
    longer assumes people talk during silence.

    This is also what lets the >=3-word backchannel test mean anything --
    without any apportionment a one-word "okay" sitting inside a long segment
    inherits that segment's whole word count and sails through the gate.
    """
    w = seg.text.split()
    if not w or seg.dur <= 0:
        return "", 0.0
    total = vad.speech_between(iv, seg.start, seg.end)
    if total <= 0:                       # segment with no VAD speech under it
        f0 = (max(a, seg.start) - seg.start) / seg.dur
        f1 = (min(b, seg.end) - seg.start) / seg.dur
    else:
        f0 = vad.speech_between(iv, seg.start, max(a, seg.start)) / total
        f1 = vad.speech_between(iv, seg.start, min(b, seg.end)) / total
    i0 = max(0, min(len(w), int(f0 * len(w))))
    i1 = max(i0, min(len(w), int(f1 * len(w) + 0.999)))
    return " ".join(w[i0:i1]), len(w) * max(0.0, f1 - f0)


def _confirm(raw: list[tuple[float, float]], segs: list[Segment], side: str) -> tuple[list[Span], int, float]:
    """Keep VAD spans backed by a transcript segment on this leg; attach text."""
    starts = [s.start for s in segs]
    kept: list[Span] = []
    dropped, dropped_t = 0, 0.0
    for a, b in raw:
        # step back a few, so a long segment that started well before `a` but
        # still covers it is not missed
        i = max(0, bisect.bisect_left(starts, a) - 4)
        hit, best = [], 0.0
        while i < len(segs) and segs[i].start < b:
            ov = _overlap(a, b, segs[i].start, segs[i].end)
            if ov > 0:
                hit.append((ov, segs[i]))
                best = max(best, ov)
            i += 1
        if best < MIN_TRANSCRIPT_OVERLAP:
            dropped += 1
            dropped_t += b - a
            continue
        hit.sort(key=lambda x: x[1].start)
        inside = len(hit) == 1 and hit[0][1].start <= a and hit[0][1].end >= b
        parts, nw = [], 0.0
        for _, s in hit:
            t, w = _apportion(a, b, s, raw)
            if t:
                parts.append(t)
            nw += w
        kept.append(
            Span(
                a, b, side,
                tuple(s.utt for _, s in hit),
                " ".join(parts).strip(),
                " ".join(s.text for _, s in hit).strip(),
                nw,
                inside,
            )
        )
    return kept, dropped, dropped_t


def build(
    call: StereoCall,
    thr_on: float = 0.50,
    thr_off: float = 0.35,
    min_speech_s: float = 0.10,
    min_silence_s: float = 0.10,
) -> CallTimeline:
    raw: dict[str, list[tuple[float, float]]] = {}
    spans: list[Span] = []
    dropped, dropped_t = 0, 0.0
    for side in SIDES:
        p = vad.load(leg_reco(call.base, side))
        iv = vad.intervals(p, thr_on, thr_off, min_speech_s, min_silence_s)
        raw[side] = iv
        keep, d, dt = _confirm(iv, call.legs[side], side)
        spans.extend(keep)
        dropped += d
        dropped_t += dt
    spans.sort(key=lambda s: (s.start, s.end))

    tl = CallTimeline(call.base, spans, raw, dict(call.legs), dropped, dropped_t)
    tl.stats = _stats(tl, call)
    return tl


def _stats(tl: CallTimeline, call: StereoCall) -> dict:
    span_end = max((s.end for s in tl.spans), default=0.0)
    seg_speech = {s: sum(x.dur for x in call.legs[s]) for s in SIDES}
    vad_speech = {s: sum(b - a for a, b in tl.speech[s]) for s in SIDES}
    kept = {s: sum(x.dur for x in tl.side_spans(s)) for s in SIDES}

    # simultaneous speech, measured on confirmed spans
    l, r = tl.side_spans(LEFT), tl.side_spans(RIGHT)
    both = 0.0
    j = 0
    for x in l:
        while j < len(r) and r[j].end <= x.start:
            j += 1
        k = j
        while k < len(r) and r[k].start < x.end:
            both += _overlap(x.start, x.end, r[k].start, r[k].end)
            k += 1
    tot = sum(kept.values())
    return {
        "span_end": span_end,
        "seg_speech": seg_speech,
        "vad_speech": vad_speech,
        "kept_speech": kept,
        "overlap_s": both,
        "overlap_frac": both / tot if tot else 0.0,
        "n_spans": len(tl.spans),
    }


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------

CHANGE, HOLD = "change", "hold"


@dataclass(slots=True)
class Boundary:
    """The moment one leg stops talking, and what happened next."""

    base: str
    side: str                 # who just stopped
    t: float                  # their VAD speech offset -- the cut point
    kind: str                 # CHANGE (other leg spoke next) | HOLD (same leg resumed)
    gap: float                # silence until whoever spoke next (negative = overlap)
    prev_span: Span
    next_span: Span
    prev_dur: float           # contiguous speech by `side` ending at t
    prev_text: str            # words apportioned to that spurt
    next_dur: float           # contiguous speech by the next speaker
    next_words: float
    next_text: str
    other_speech_before: float   # other leg's speech in the 1 s before t
    other_speech_gap: float      # other leg's speech inside the gap itself
    other_active_at_t: bool
    """The other leg is mid-span at the instant this speaker stops.

    This, not `other_speech_before`, is the barge-in signal. Measured over all
    50,532 boundaries the floor-change rate is flat at 36-42% no matter how
    much the other leg spoke in the preceding second, and `ends_segment` is
    *higher* where there was overlap -- so speech before the boundary is
    backchannel-rich conversation, not interruption, and gating on it discards
    a third of the corpus for nothing.

    A straddle is different on both counts: the speakers genuinely collide,
    and because the merge only looks forward from `t`, `gap` would be measured
    to some later span and be meaningless.
    """
    ends_segment: bool
    """Whether the human transcriber closed this speaker's segment here.

    True when the next VAD span on this leg belongs to a different transcript
    segment, i.e. `t` is the last speech in the segment. It is a second,
    independent opinion on the same boundary and it cuts both classes:

      CHANGE + True  the transcriber ended the utterance and the other side
                     took the floor -- two witnesses agreeing, the best
                     positives available.
      CHANGE + False the other side started talking inside an utterance the
                     transcriber treated as still running: an interruption
                     wearing a turn-change costume.
      HOLD   + False the pause sits inside one segment, so the transcriber
                     heard one continuous utterance -- the speaker had not
                     finished, which is exactly the negative class.
      HOLD   + True  the pause straddles a segment edge, so it may be a
                     finished sentence followed by another one.
    """


def _spurt(spans: list[Span], i: int, max_gap: float, back: bool) -> tuple[float, float, str]:
    """The contiguous talk-spurt containing spans[i] -- spans of one leg
    separated by less than `max_gap` count as one stretch of talking.

    Returns (duration, words, text). Working in spurts rather than single VAD
    spans is what makes "did the other speaker actually take the floor?"
    answerable: a reply is often several spans with breath pauses between them,
    and judging it on its first span alone would read as a backchannel.
    """
    lo = hi = i
    if back:
        while lo > 0 and spans[lo].start - spans[lo - 1].end < max_gap:
            lo -= 1
    else:
        while hi + 1 < len(spans) and spans[hi + 1].start - spans[hi].end < max_gap:
            hi += 1
    run = spans[lo : hi + 1]
    dur = run[-1].end - run[0].start
    words = sum(s.words for s in run)
    # Apportioned slices can share a boundary word, since a word straddling two
    # VAD spans is rounded into both. Drop the repeat when joining.
    out: list[str] = []
    for s in run:
        w = s.text.split()
        if w and out and out[-1] == w[0]:
            w = w[1:]
        out.extend(w)
    return dur, words, " ".join(out)


def boundaries(tl: CallTimeline, spurt_gap: float = 0.40) -> list[Boundary]:
    per: dict[str, list[Span]] = {s: tl.side_spans(s) for s in SIDES}
    idx: dict[str, list[float]] = {s: [x.start for x in per[s]] for s in SIDES}
    out: list[Boundary] = []

    for side in SIDES:
        oth = other(side)
        mine, theirs = per[side], per[oth]
        their_iv = [(x.start, x.end) for x in theirs]
        for i, sp in enumerate(mine[:-1] if mine else []):
            t = sp.end
            nxt_same = mine[i + 1]
            k = bisect.bisect_right(idx[oth], t)
            # first span on the other leg that starts at or after t
            nxt_oth = theirs[k] if k < len(theirs) else None
            # ... and whether the span before that one is still running at t
            straddle = k > 0 and theirs[k - 1].end > t

            if nxt_oth is not None and nxt_oth.start < nxt_same.start:
                kind, nxt, oi = CHANGE, nxt_oth, k
            else:
                kind, nxt, oi = HOLD, nxt_same, i + 1

            if kind == CHANGE:
                nd, nw, nt = _spurt(theirs, oi, spurt_gap, back=False)
            else:
                nd, nw, nt = _spurt(mine, oi, spurt_gap, back=False)
            pd, _, pt = _spurt(mine, i, spurt_gap, back=True)
            # always judged against this leg's own next span, so it means the
            # same thing for a floor change and for a hold
            ends_seg = not (set(sp.segs) & set(nxt_same.segs))

            out.append(
                Boundary(
                    base=tl.base,
                    side=side,
                    t=t,
                    kind=kind,
                    gap=nxt.start - t,
                    prev_span=sp,
                    next_span=nxt,
                    prev_dur=pd,
                    prev_text=pt,
                    next_dur=nd,
                    next_words=nw,
                    next_text=nt,
                    other_speech_before=vad.speech_between(their_iv, t - 1.0, t),
                    other_speech_gap=vad.speech_between(their_iv, t, nxt.start),
                    other_active_at_t=straddle,
                    ends_segment=ends_seg,
                )
            )
    out.sort(key=lambda b: b.t)
    return out
