"""Turns as floor occupancy: who holds the floor, and for how long.

`turns.py` answers a local question -- at this one offset, did the other leg
speak next? That is the right question for an 8 s clip and it is what the
`clips` config is built on. It is the wrong question for a turn, for two
reasons measured over the corpus:

  1. The gated `change` rows that survived `rules.classify` are 5,740 of the
     20,013 floor changes. Grouping those into turns swallows the other 14,273
     *inside* turns, so intervals where the other speaker held the floor get
     emitted as this speaker's mid-turn hesitations -- wrong on 66.1% of rows.

  2. `Boundary.kind` is decided by comparing which leg speaks next, so when
     speech overlaps a genuine floor change reads as a HOLD. The floor fails to
     alternate on 5,006 of 19,897 consecutive change pairs, and the speech in
     those gaps is not backchannels: median 1.47 s and 3.6 words, only 19.6%
     silent, against 0.00 s and 83.4% silent where it does alternate.

So turns are derived from the span sequence directly and never consult
`Boundary`. Every confirmed span lands in exactly one turn, which is a
partition `pipeline/24_verify_turns.py` can assert rather than spot-check.

See docs/turns_format.md for the full spec and the numbers above.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import vad
from .turns import CallTimeline, Span

# A run by one side, sandwiched between two runs by the other side, that is
# *both* this short and this word-poor did not take the floor.
#
# `rules.classify` rejects a positive when `next_words < 3` OR `next_dur < 1.0`.
# The `AND` here is deliberate. That gate decides whether there is enough
# evidence to call something a completed turn, so it errs toward rejecting.
# Segmentation errs the other way: over-absorbing merges two real turns and
# invents a mid-turn pause where a genuine end-of-turn was, manufacturing false
# negatives in the one field this format exists to carry. Under-absorbing only
# splits a turn, which is visible in `n_spans` and recoverable downstream.
BACKCHANNEL_WORDS = 3.0
BACKCHANNEL_DUR = 1.00

# Real audio kept after the turn ends, so a streaming detector has silence to
# decide against. Clamped per row to the owner's next speech -- see `Turn.trail`.
TRAIL_S = 0.50

# A pause this long with nobody else talking is a *lapse*: the speaker
# finished, the floor went empty, and they eventually resumed. Calling it a
# mid-turn hesitation asserts the opposite of what happened, and buries a real
# end-of-turn inside a row. `rules.NEG_MAX_PAUSE` is the same 2.0 s, chosen for
# the same reason -- beyond it the core pass declined to call a same-speaker
# pause "incomplete".
#
# `ta_IN_10102903_20230117_R_0021` is the case that forced this: a 42.8 s turn
# whose ten pauses are [8.45, 0.67, 0.64, 0.38, ...]. The transcriber agrees --
# segment 44721 ends at 310.95 and 44722 opens at 319.02.
#
# Splitting beats dropping: one bad row becomes two good ones plus an extra
# end-of-turn example, and the lapse itself becomes the first turn's `eot_gap`.
LAPSE_S = 2.00

# --- the `clean` gate ----------------------------------------------------
# Other-leg speech tolerated inside a mid-turn pause. `rules.MAX_CROSSTALK`,
# for the same reason: measured bleed between the legs is ~2% of speech, so
# anything above this is the other speaker actually talking.
MAX_PAUSE_CROSSTALK = 0.05
# Speech in the turn, so there is prosody to read. `rules.MIN_PREV_DUR`.
MIN_CLEAN_DUR = 1.00
# Apportioned words per second of actual speech. The corpus sits at a median of
# 2.66 with p5 at 1.48, so 1.0 is well outside the normal population and the
# rows it removes are a separate mode at ~0.08, not a tail. Two different
# defects land there and both disqualify a row:
#
#   - apportionment breaking down, where several spans each take a fractional
#     slice of one long segment and every slice rounds to the same single word.
#     `ta_IN_9829427_20230331_L_0183` is 13.18 s of speech rendered as "okay".
#   - a speaker who is not taking the floor at all, just acknowledging for
#     fifteen seconds ("ஹம் ஹம் ஹம் ஹம்"). Real audio, but not a turn.
MIN_WORD_RATE = 1.00


@dataclass(slots=True)
class Turn:
    """One speaker's continuous hold of the floor."""

    base: str
    side: str
    index: int                          # ordinal in the call, across both legs
    start: float                        # first span of the run
    end: float                          # last span of the run
    spans: list[Span]
    silence: list[tuple[float, float]]
    """This speaker's own mid-turn pauses, strictly inside [start, end].

    An absorbed backchannel needs no special handling: the other leg spoke, so
    on *this* leg the interval is silence between two of the owner's spans, and
    it is already a gap in this list.
    """
    eot_gap: float | None
    """Silence from `end` until the next speaker starts. `None` for the last
    turn of a call. **Negative where they overlapped** -- 40.6% of turns.

    Kept out of `silence` on purpose. The end-of-turn silence begins *at* `end`
    and runs past it, so it can never satisfy the containment check that every
    other span must satisfy, and a negative one cannot be written as an
    interval at all.
    """
    trail: float
    text: str
    words: float
    absorbed: int                       # backchannel runs merged into this turn

    starts_segment: bool = True
    """The turn opens a transcript segment rather than resuming one.

    False means the run was split mid-utterance: the speaker was still inside
    one segment when the other leg started talking, so start-sorted grouping
    cut them off and the remainder became its own "turn". Those rows begin
    mid-word. `ta_IN_10102736_20230329_L_0121` is the archetype -- 0.93 s, no
    pauses, opening in the middle of `SHA1P_utt00023228`.
    """
    ends_segment: bool = True
    """The transcriber closed the utterance here -- the same witness
    `turns.Boundary.ends_segment` carries, and the one that separates `change`
    (gold positive) from `change_midseg` (held back) in the clips pipeline.
    False means the floor changed inside an utterance the transcriber treated
    as still running: an interruption, not an end-of-turn.
    """
    speech_crosstalk: float = 0.0
    """Other-leg speech, backchannels excluded, across the whole turn.

    `pause_crosstalk` only inspects the pauses, so it is blind to a turn spoken
    entirely *over* the other party. `ta_IN_10102663_20230123_R_0057` is 1.63 s
    of speech cut off mid-sentence, sitting wholly inside the other leg's
    [613.60, 620.58]: the speaker tried to take the floor, failed, and gave up.
    It has no pauses, so it passed every pause-based check.

    This is `rules.drop_straddle` / `drop_bargein` at turn scale.
    """
    pause_crosstalk: float = 0.0
    """Most other-leg speech inside any one mid-turn pause.

    A pause is supposed to be this speaker hesitating. When the other leg is
    talking through it, the row claims as hesitation what was really the other
    speaker holding the floor -- audible immediately on listening, and the
    exact defect this format exists to avoid. It survives run-grouping because
    grouping splits on span *onsets*: an other-leg span that began before this
    turn can run through a pause inside it without ever starting there.
    """

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def speech(self) -> float:
        """Seconds the owner is actually speaking -- the turn minus its pauses."""
        return sum(s.dur for s in self.spans)

    @property
    def word_rate(self) -> float:
        """Apportioned words per second of speech. See `MIN_WORD_RATE`.

        Rounded to the precision that ships, so `clean` is decided by the same
        number a consumer reads. Unrounded, a rate of 0.9973 is excluded by the
        gate while the row shows `word_rate: 1.0` -- a row that contradicts
        itself, and one that step 24 flagged.
        """
        return round(self.words / self.speech, 2) if self.speech > 0 else 0.0

    @property
    def clean(self) -> bool:
        """Usable as an end-of-turn example.

        `eot_gap >= 0` and not `>= 0.2`: `rules.POS_MIN_GAP` is 0.0 on purpose,
        because fast handovers are "precisely the cases a slow endpointer gets
        wrong". A 200 ms floor would drop 339 of them. What is excluded is a
        *negative* gap -- the next speaker started first, so there is no
        end-of-turn silence to detect at all.
        """
        return (self.starts_segment and self.ends_segment
                and self.speech_crosstalk <= MAX_PAUSE_CROSSTALK
                and self.eot_gap is not None and self.eot_gap >= 0.0
                and self.dur >= MIN_CLEAN_DUR
                and self.word_rate >= MIN_WORD_RATE)


def runs(spans: list[Span]) -> tuple[list[tuple[str, list[Span]]], list[Span]]:
    """Maximal same-side runs over the merged span sequence, with backchannels
    absorbed to a fixed point.

    `spans` must be sorted by `(start, end)` across both legs. Sorting by start
    is what makes an overlapping barge-in split the run at the moment it begins,
    which is the behaviour we want: that is a floor change, an interrupting one.

    Returns the runs and the spans absorbed away, which the caller needs to tell
    a benign backchannel apart from the other speaker genuinely holding the floor.
    """
    out: list[tuple[str, list[Span]]] = []
    for s in spans:
        if out and out[-1][0] == s.side:
            out[-1][1].append(s)
        else:
            out.append((s.side, [s]))

    dropped: list[Span] = []
    changed = True
    while changed:
        changed = False
        merged: list[tuple[str, list[Span]]] = []
        i = 0
        while i < len(out):
            if (0 < i < len(out) - 1
                    and out[i - 1][0] == out[i + 1][0] != out[i][0]
                    and _is_backchannel(out[i][1])):
                # Extend what is already in `merged`, never rebuild from
                # `out[i - 1]`. Absorptions chain -- [A, b1, B, b2, C] absorbs
                # twice in one pass -- and rebuilding drops everything merged
                # by the previous one. That silently deleted real speech
                # (a 10.37 s span, caught by the totality check in step 24).
                merged[-1] = (merged[-1][0], merged[-1][1] + out[i + 1][1])
                dropped.extend(out[i][1])
                i += 2
                changed = True
                continue
            merged.append(out[i])
            i += 1
        out = merged
    return out, dropped


def _split_lapses(grouped: list[tuple[str, list[Span]]],
                  other_iv: dict[str, list[tuple[float, float]]]
                  ) -> list[tuple[str, list[Span]]]:
    """Cut a run wherever its internal pause is not a hesitation.

    Two ways that happens, and both mean the turn already ended there:

      - the pause runs past `LAPSE_S` with the floor empty -- the speaker
        finished and nobody answered;
      - the other leg was genuinely speaking across it. Grouping cannot catch
        this one on its own, because it splits on span *onsets*: an other-leg
        span that began before this run carries on through a pause inside it
        without ever starting there.

    `other_iv` must exclude absorbed backchannels. A backchannel is precisely
    the case where the other party does *not* take the floor, so splitting on
    one would end a turn that never ended.
    """
    out: list[tuple[str, list[Span]]] = []
    for side, run in grouped:
        cur = [run[0]]
        for prev, nxt in zip(run, run[1:]):
            gap = nxt.start - prev.end
            xtalk = vad.speech_between(other_iv[side], prev.end, nxt.start)
            if gap > LAPSE_S or xtalk > MAX_PAUSE_CROSSTALK:
                out.append((side, cur))
                cur = [nxt]
            else:
                cur.append(nxt)
        out.append((side, cur))
    return out


def _is_backchannel(run: list[Span]) -> bool:
    dur = run[-1].end - run[0].start
    return sum(s.words for s in run) < BACKCHANNEL_WORDS and dur < BACKCHANNEL_DUR


# A word can only be rounded into two spans if they are effectively touching.
# Beyond this the spans are separated by a real pause, and a repeat is the
# speaker actually saying it twice.
DEDUP_GAP = 0.05


def _text(run: list[Span]) -> str:
    """Join a run's apportioned text, dropping the word repeated across a span
    edge. `turns._spurt` does the same thing for the same reason: a word
    straddling two VAD spans is rounded into both.

    Only across a *touching* edge, though. `ta_IN_10102885_20230404_L_0078` is
    "Hello", a 2.75 s pause, then "Hello" again -- two different transcript
    segments -- and an unconditional dedup renders that turn as a single
    "Hello", deleting speech the listener can plainly hear.
    """
    words: list[str] = []
    prev_end: float | None = None
    for s in run:
        w = s.text.split()
        touching = prev_end is not None and s.start - prev_end < DEDUP_GAP
        if w and words and touching and words[-1] == w[0]:
            w = w[1:]
        words.extend(w)
        prev_end = s.end
    return " ".join(words)


def build(tl: CallTimeline, wav_dur: dict[str, float] | None = None) -> list[Turn]:
    """Every turn in one call, in time order."""
    ordered = sorted(tl.spans, key=lambda s: (s.start, s.end))
    grouped, absorbed = runs(ordered)

    sides = {s.side for s in ordered}
    # The other leg, minus the backchannels absorbed away. Everything that
    # decides "was the other speaker actually holding the floor?" reads this,
    # never the raw other-leg spans.
    gone = {(s.start, s.end, s.side) for s in absorbed}
    held_iv = {
        side: [(s.start, s.end) for s in ordered
               if s.side != side and (s.start, s.end, s.side) not in gone]
        for side in sides
    }
    grouped = _split_lapses(grouped, held_iv)

    # Every span the owner has anywhere in the call, for the trailing-window
    # clamp: the trail must not reach into this speaker's *own* next turn.
    per_side = {side: [s for s in ordered if s.side == side] for side in sides}
    own_starts = {side: [s.start for s in per_side[side]] for side in sides}
    at = {side: {id(s): k for k, s in enumerate(per_side[side])} for side in sides}
    other_iv = {
        side: [(s.start, s.end) for s in ordered if s.side != side]
        for side in sides
    }

    out: list[Turn] = []
    for i, (side, run) in enumerate(grouped):
        start, end = run[0].start, run[-1].end
        gaps = [(run[k].end, run[k + 1].start)
                for k in range(len(run) - 1)
                if run[k + 1].start > run[k].end]

        eot = grouped[i + 1][1][0].start - end if i + 1 < len(grouped) else None

        # The transcriber's opinion at both edges. A shared segment id with the
        # neighbouring same-side span means the human heard one continuous
        # utterance, so this edge is mid-sentence, not a turn boundary.
        mine = per_side[side]
        i0, i1 = at[side][id(run[0])], at[side][id(run[-1])]
        starts_seg = i0 == 0 or not (set(mine[i0 - 1].segs) & set(run[0].segs))
        ends_seg = i1 == len(mine) - 1 or not (set(mine[i1 + 1].segs) & set(run[-1].segs))

        xtalk = max((vad.speech_between(other_iv[side], a, b) for a, b in gaps),
                    default=0.0)
        # The other speaker talking across the turn *itself*, not just its
        # pauses. `ta_IN_10102663_20230123_R_0057` is the case: a 1.63 s
        # fragment cut mid-sentence, sitting entirely inside the other leg's
        # [613.60, 620.58]. It has no pauses at all, so nothing that only
        # inspects pauses can see it.
        speech_xtalk = vad.speech_between(held_iv[side], start, end)

        limit = TRAIL_S
        nxt = [a for a in own_starts[side] if a > end]
        if nxt:
            limit = min(limit, nxt[0] - end)
        if wav_dur is not None:
            limit = min(limit, max(0.0, wav_dur[side] - end))

        # A run that ends where the owner's next span begins can only happen
        # through a zero-length gap; clamp rather than emit a negative trail.
        trail = max(0.0, limit)

        out.append(
            Turn(
                base=tl.base,
                side=side,
                index=i,
                start=round(start, 3),
                end=round(end, 3),
                spans=run,
                silence=[(round(a, 3), round(b, 3)) for a, b in gaps],
                eot_gap=round(eot, 3) if eot is not None else None,
                trail=round(trail, 3),
                text=_text(run),
                words=round(sum(s.words for s in run), 2),
                absorbed=0,
                starts_segment=starts_seg,
                ends_segment=ends_seg,
                pause_crosstalk=round(xtalk, 3),
                speech_crosstalk=round(speech_xtalk, 3),
            )
        )

    _count_absorbed(out, ordered)
    return out


def _count_absorbed(built: list[Turn], ordered: list[Span]) -> None:
    """Fill in `Turn.absorbed`, after all turns exist.

    "Inside this turn's interval" is not the same as "absorbed". Two speakers
    can genuinely overlap, so a *whole other-side turn* can sit inside this
    one's time range without having been folded in. The absorbed spans are the
    ones no same-side turn of their own claims -- those are the backchannels
    `runs()` dropped, and they are the only other-side speech that has nowhere
    else to live.

    Membership is by *onset*, not containment: a backchannel can start inside
    the turn and run a fraction past its end, because the two speakers overlap
    there. Requiring containment orphans those -- 19 spans across the corpus,
    reported by step 24 as lost speech.

    The onset bound is inclusive at the start for the same reason. Both legs
    can open on the same VAD frame -- two people saying hello at once, which is
    how one call in R1 begins -- and a strict `<` orphans the absorbed leg.
    """
    claimed = {
        (s.start, s.end, s.side)
        for t in built
        for s in t.spans
    }
    for t in built:
        t.absorbed = sum(
            1 for s in ordered
            if s.side != t.side
            and t.start <= s.start < t.end
            and (s.start, s.end, s.side) not in claimed
        )
