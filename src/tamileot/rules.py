"""Turning boundaries into labelled EOT samples.

Two classes, matching Smart Turn's convention:

  complete  (1) -- the speaker finished; an agent should reply now.
  incomplete(0) -- the speaker paused mid-thought; an agent that replies here
                   cuts the user off.

Every sample is cut from one leg only. That is not a convenience: it is what
the deployed detector actually receives, because a voice agent's turn detector
sees the inbound user stream, not a mixdown of both sides.

The two classes are given *identical* trailing silence. If positives ended
with more silence than negatives the model would learn to read the silence
length instead of the speech, and would collapse the moment it met a
production endpointer with different timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .turns import CHANGE, HOLD, Boundary

# --- clip geometry -------------------------------------------------------
CLIP_S = 8.0        # Smart Turn v3 consumes at most the last 8 s
TRAIL_S = 0.20      # real post-offset audio kept, identical for both classes

# --- shared gates --------------------------------------------------------
MIN_PREV_DUR = 1.00     # speech ending at the cut, so there is prosody to read
MAX_CROSSTALK = 0.05    # other leg speech tolerated *inside a gap*, seconds
# Only used for the extreme case now: the other leg holding the floor for most
# of the second before the cut. See Boundary.other_active_at_t for why the
# tight version of this gate was wrong.
MAX_OTHER_BEFORE = 0.80

# --- positive gates ------------------------------------------------------
POS_MAX_GAP = 1.50      # beyond this the handover is too slow to be evidence
# No minimum gap. The clip is cut from this speaker's leg alone, so the audio
# after their offset is silence on that leg whatever the other speaker does --
# measured crosstalk bleed is ~2% of speech. Imposing a 200 ms floor would
# throw away fast handovers, which are precisely the cases a slow endpointer
# gets wrong and therefore the most valuable positives we have.
POS_MIN_GAP = 0.0
POS_MIN_NEXT_DUR = 1.00
POS_MIN_NEXT_WORDS = 3.0  # rejects backchannels: um, okay, aah

# --- negative gates ------------------------------------------------------
# Unlike the positive case there IS a hard floor here, and it is TRAIL_S: the
# speaker who paused is the one who resumes, on this same leg, so a pause
# shorter than the trailing window would put their resumed speech inside the
# clip and hand the model the answer. Both classes still end in 200 ms of real
# silence, so trailing-silence length carries no label information.
NEG_MIN_PAUSE = TRAIL_S
NEG_MAX_PAUSE = 2.00
NEG_MIN_NEXT_DUR = 0.30


@dataclass(frozen=True, slots=True)
class Gates:
    """The thresholds, in one object so a second pass can move them without
    duplicating `classify`'s branch structure -- two copies of these rules
    would drift, and the funnel would stop reconciling."""

    min_prev_dur: float = MIN_PREV_DUR
    max_other_before: float = MAX_OTHER_BEFORE
    max_crosstalk: float = MAX_CROSSTALK
    pos_min_gap: float = POS_MIN_GAP
    pos_max_gap: float = POS_MAX_GAP
    pos_min_next_dur: float = POS_MIN_NEXT_DUR
    pos_min_next_words: float = POS_MIN_NEXT_WORDS
    neg_min_pause: float = NEG_MIN_PAUSE
    neg_max_pause: float = NEG_MAX_PAUSE
    neg_min_next_dur: float = NEG_MIN_NEXT_DUR


DEFAULT = Gates()

# The second pass (`pipeline/07_relax_funnel.py`). Only two numbers move, and
# both moved for the same reason: they were set when the pipeline itself had to
# decide the label, and a labeller that agrees with a human 97.5% of the time
# now does that instead. See the script's docstring for why the other rejected
# pools stay rejected.
RELAXED = replace(DEFAULT, min_prev_dur=0.50, neg_max_pause=5.00)


# Sources where the acoustics and the transcriber agree. These are the gold
# set. The other two are kept, labelled and tagged, but held out of it: they
# are the cases where the two witnesses disagree, and promoting them needs
# evidence we do not have yet (a sentence-final text filter for `hold_inter`,
# and something better than a guess for `change_midseg`).
CORE = ("change", "hold_intra")
HELD_BACK = ("change_midseg", "hold_inter")


def is_core(row: dict) -> bool:
    return row["source"] in CORE


@dataclass(slots=True)
class Sample:
    sid: str
    base: str
    side: str
    label: int          # 1 complete, 0 incomplete
    t: float            # speech offset -- the decision point
    clip_start: float
    clip_end: float
    gap: float
    prev_dur: float
    text: str           # what the speaker had said in the span ending at t
    source: str         # change | hold_intra | hold_inter
    next_text: str = ""
    harvest: str = "core"
    """Which pass produced this row: `core` for the original gates, `relaxed`
    for the second pass. Provenance only -- it is *not* a feature, for the same
    reason `dispute` and `source` are not: it is unavailable at inference."""


def _mk(b: Boundary, label: int, source: str, harvest: str = "core") -> Sample:
    end = b.t + TRAIL_S
    return Sample(
        sid=f"{b.base}_{b.side[0]}_{int(round(b.t * 1000)):08d}",
        base=b.base,
        side=b.side,
        label=label,
        t=round(b.t, 3),
        clip_start=round(max(0.0, end - CLIP_S), 3),
        clip_end=round(end, 3),
        gap=round(b.gap, 3),
        prev_dur=round(b.prev_dur, 3),
        text=b.prev_text,
        source=source,
        next_text=b.next_text,
        harvest=harvest,
    )


def classify(bs: list[Boundary], g: Gates = DEFAULT,
             harvest: str = "core") -> tuple[list[Sample], dict[str, int]]:
    """Apply the gates. Returns samples plus a full rejection funnel, so every
    boundary is accounted for and no filter can quietly eat the data."""
    out: list[Sample] = []
    n: dict[str, int] = {}

    def bump(k: str) -> None:
        n[k] = n.get(k, 0) + 1

    for b in bs:
        bump("total")
        if b.prev_dur < g.min_prev_dur:
            bump("drop_short_prev")
            continue
        if not b.prev_text:
            bump("drop_no_text")
            continue
        if b.other_active_at_t:
            bump("drop_straddle")    # the two speakers collide across the cut
            continue
        if b.other_speech_before > g.max_other_before:
            bump("drop_bargein")     # the other side held the floor throughout
            continue

        if b.kind == CHANGE:
            bump("cand_change")
            if b.gap < g.pos_min_gap:
                bump("drop_pos_gap_short")
            elif b.gap > g.pos_max_gap:
                bump("drop_pos_gap_long")
            elif b.next_words < g.pos_min_next_words or b.next_dur < g.pos_min_next_dur:
                bump("drop_pos_backchannel")
            elif not b.ends_segment:
                # The other side started talking inside an utterance the
                # transcriber treated as still running. The floor changed, but
                # this speaker had not finished -- calling it "complete" would
                # teach the model to reply over people.
                bump("change_midseg")
                out.append(_mk(b, 1, "change_midseg", harvest))
            else:
                bump("keep_pos")
                out.append(_mk(b, 1, "change", harvest))
        else:
            bump("cand_hold")
            if b.gap < g.neg_min_pause:
                bump("drop_neg_pause_short")
            elif b.gap > g.neg_max_pause:
                bump("drop_neg_pause_long")
            elif b.other_speech_gap > g.max_crosstalk:
                bump("drop_neg_other_spoke")
            elif b.next_dur < g.neg_min_next_dur:
                bump("drop_neg_resume_tiny")
            elif b.ends_segment:
                # The pause straddles a transcript segment edge, so it may be a
                # finished sentence followed by another. Kept apart: labelling
                # those "incomplete" would teach the model the opposite of the
                # truth. Promoting them needs a sentence-final text filter.
                bump("hold_inter")
                out.append(_mk(b, 0, "hold_inter", harvest))
            else:
                # Pause wholly inside one transcript segment: the transcriber
                # judged this one continuous utterance, so the speaker had not
                # finished. That judgement is the label.
                bump("keep_neg")
                out.append(_mk(b, 0, "hold_intra", harvest))
    return out, n


def as_row(s: Sample) -> dict:
    return asdict(s)
