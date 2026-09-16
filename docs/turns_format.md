# `tamil-turns` — turn-level format

Spec for the turn-level corpus published as
[`santhosh-005/tamil-turns`](https://huggingface.co/datasets/santhosh-005/tamil-turns),
a **separate repo** from the 8 s `clips` dataset in `santhosh-005/tamil-eot`.
No new VAD pass and no new labels — it is a regrouping
of speech already on disk, built by `pipeline/23_build_turns.py` and gated by
`pipeline/24_verify_turns.py`.

> Revised 2026-09-13. The first draft reconstructed turns by grouping the
> *gated* `change` rows in `labels/samples_llm.jsonl.gz`. Measured against the
> corpus, that rule was wrong on **66.1%** of rows. The evidence and the
> replacement are in [Why not the boundary table](#why-not-the-boundary-table).

## What a row is

**One speaker's continuous hold of the floor**, from the moment they start
speaking to the moment the other speaker takes over. Audio is **that speaker's
leg only, never a mixdown** — same rule as `clips`, and the same thing a
deployed detector receives.

Inside a turn, the speaker's own pauses are mid-turn hesitations. The silence
after the turn is the real end-of-turn. That distinction is the whole point of
the format: the 8 s clips destroy it by construction.

## The definition

Turns come from **floor occupancy over the confirmed span sequence**, not from
the `CHANGE`/`HOLD` boundary classifier:

1. take every confirmed span from `turns.build()` — both legs, sorted by
   `(start, end)`;
2. group maximal runs of consecutive spans belonging to the same side;
3. **absorb backchannels**: a run sandwiched between two runs by the *other*
   side that is both `< 3` apportioned words and `< 1.0 s` long did not take the
   floor. Merge the two runs around it and repeat to a fixed point;
4. **split at any pause that is not a hesitation** — longer than `LAPSE_S`
   (2.0 s) with the floor empty, or one the other leg is genuinely speaking
   across. Both mean the turn already ended there;
5. every surviving run is one turn.

The property that makes this checkable: it is **total**. Every confirmed span
lands in exactly one turn — the only exceptions are the absorbed backchannels,
which must pass the backchannel gate — so `24_verify_turns.py` asserts the
partition rather than spot-checking it. Neither boundary-table rule has that
property.

### Why a lapse ends the turn

A pause past 2.0 s with nobody else talking is not a hesitation. The speaker
finished, the floor went empty, and they eventually resumed — so burying it
inside a row asserts the opposite of what happened and hides a real
end-of-turn. `rules.NEG_MAX_PAUSE` is the same 2.0 s, for the same reason:
beyond it the core pass declined to call a same-speaker pause "incomplete".

`ta_IN_10102903_20230117_R_0021` is the case that forced it — a 42.8 s turn
whose ten pauses were `[8.45, 0.67, 0.64, 0.38, …]`. The transcriber agrees:
segment 44721 closes at 310.95 and 44722 opens at 319.02. It is now two clean
turns, and the 8.45 s lapse is the first one's `eot_gap`.

**Splitting beats dropping.** One bad row becomes two good ones plus an extra
end-of-turn example, which is why the clean count *rose* from 4,277 to 4,774
when this landed even though the gates got stricter.

### Why the backchannel gate is `AND`, not `OR`

`rules.CLASSIFY` rejects a positive when `next_words < 3` **or**
`next_dur < 1.0` — deliberately conservative, because it is deciding whether
there is strong enough evidence to *call something a completed turn*.

Segmentation asks a different question: did the floor actually change? Here the
errors are not symmetric. Over-absorbing **merges two real turns into one and
invents a mid-turn pause where a genuine end-of-turn was** — it manufactures
false negatives in the one field the format exists to carry. Under-absorbing
merely splits a turn, which is visible in `n_spans` and recoverable downstream.
So absorption requires *both* conditions, and the divergence from `rules.py` is
intentional.

## Schema

| field | source |
|---|---|
| `id` | `{conversation_id}_{speaker_id}_{turn_index}` |
| `audio` | slice of the leg WAV, 16 kHz mono, `[start_time, end_time + trail_s]` |
| `language` | `"ta"` |
| `duration` | `end_time - start_time` |
| `transcript` | apportioned text of the spans in the run |
| `n_words` | apportioned word count — fractional, **approximate**, see below |
| `silence_spans` | `[{start, end}]` — the speaker's own mid-turn pauses, **all strictly inside `[start_time, end_time]`** |
| `eot_gap` | silence from `end_time` until the next speaker starts. **Negative means they overlapped** |
| `trail_s` | real audio kept after `end_time`, `min(0.5, own next speech)` |
| `conversation_id` | `base` |
| `speaker_id` | `{base}_L` / `{base}_R` from `side` |
| `turn_index` | ordinal within the conversation, across both speakers |
| `start_time` | first span of the run |
| `end_time` | last span of the run |
| `n_spans` | spans in the run |
| `n_absorbed` | backchannels by the other side folded into this turn |
| `speech_s` | `duration` minus the pauses — seconds actually spoken |
| `word_rate` | `n_words / speech_s` |
| `starts_segment` | the turn opens an utterance rather than resuming one |
| `ends_segment` | the transcriber closed the utterance here |
| `pause_crosstalk` | most other-leg speech inside any one mid-turn pause |
| `speech_crosstalk` | other-leg speech across the whole turn, backchannels excluded |
| `clean` | **passes all six quality gates — see below** |
| `split` | **carried through unchanged from the call-level split** |
| `label`, `llm_verdict`, `source`, `sid` | joined from `samples_llm_all.jsonl` where a gold boundary lands on `end_time`; `null` otherwise |

### `silence_spans` and `eot_gap` are separate fields

The first draft put the end-of-turn in `silence_spans` as "the last one", and
also required every span to lie inside `[start_time, end_time]`. Those two rules
contradict each other — the end-of-turn silence starts *at* `end_time` and runs
past it, so it can never satisfy the containment check.

Splitting them makes both true, and it lets `eot_gap` be **negative**, which it
is for **44.2%** of turns: the next speaker began before this one stopped.
Forcing that into a span list would have required either dropping those rows or
emitting inverted intervals. It ships as a signed number, and consumers filter.

### `messages` — exactly one, added at pack time

Not on the local table: it is fully reconstructible by grouping on
`conversation_id` and sorting by `turn_index`, and storing the history per row
would duplicate each call ~143 times over.

`25_pack_turns_hf.py` adds it for the published dataset as **one**
`role: assistant` message holding the other speaker's preceding turn. That is
eot-bench's convention — all 400 of their Hindi rows carry exactly one
assistant message and never a user one, because the harness appends the
current speaker's progressively-revealed words itself. Shipping the full
history instead (56 turns on the median row) would hand Tamil far more context
than any other language in the benchmark and make the numbers incomparable.

## Measured yield

Whole corpus, 116 calls, both legs. Regenerate with `reports/turns_build.txt`.

| | |
|---|---|
| turns | **21,596** |
| duration | median 2.69 s · mean 5.63 s · p90 12.48 s · max 410.34 s |
| turn audio (+ trail) | 33.76 h (36.61 h) |
| mid-turn pauses per turn | median 0 · mean 1.09 · 38.6% have at least one |
| mid-turn pause length | median 0.32 s · p99 1.82 s · **max 1.98 s** (`LAPSE_S` caps it) |
| words per turn | median 6.3 |
| backchannels absorbed | 4,930, into 3,099 turns |
| `eot_gap` | median 0.45 s · p10 −1.60 s · p90 2.72 s |
| `eot_gap` ≥ 0.2 s | 58.9% (12,644 turns) |
| `eot_gap` < 0 (overlap) | 34.0% (7,309 turns) |
| turn ends carrying a gold label | 5,619 (26.0%) — 5,020 `complete`, 599 `incomplete` |

| split | turns | calls | labelled | clean |
|---|---|---|---|---|
| train | 13,866 | 71 | 3,716 | 3,024 |
| dev | 2,349 | 15 | 608 | 482 |
| test | 5,381 | 30 | 1,295 | 1,268 |

That is the whole table. **What publishes is the `clean` subset — 4,774 rows,
0.68 GB of FLAC, as `santhosh-005/tamil-turns`**; see the next section for why. A label is not
required for a row to be useful, since `silence_spans`, `eot_gap` and the floor
structure are *derived* rather than labelled, so `label` is `null` where absent
rather than guessed.

## `clean` — what actually ships

A spot-listen of the raw table found two defects immediately, and both are
structural rather than rare:

- **fragments.** 31.6% of turns open mid-word. The speaker was still inside one
  transcript segment when the other leg started talking, so start-sorted
  grouping cut them and the remainder became its own row.
  `ta_IN_10102736_20230329_L_0121` is the archetype: 0.93 s, no pauses, opening
  in the middle of `SHA1P_utt00023228`. This is `change_midseg` in a different
  costume, and the clips pipeline already holds those back.
- **the other speaker inside a "mid-turn pause."** It survives run-grouping
  because grouping splits on span *onsets*: a long other-leg span that began
  before this turn runs through a pause inside it without ever starting there.
  The row then claims as hesitation what was really the other speaker holding
  the floor — the exact confusion this format exists to prevent. Now **split**
  rather than filtered, alongside the lapse.
- **the turn spoken entirely over the other party.**
  `ta_IN_10102663_20230123_R_0057` is 1.63 s cut off mid-sentence, sitting
  wholly inside the other leg's `[613.60, 620.58]`: the speaker tried to take
  the floor, failed, and gave up. It has **no pauses at all**, so every
  pause-based check passed it. That is what `speech_crosstalk` is for, and it
  is `rules.drop_straddle` / `drop_bargein` at turn scale.

So every row carries the witnesses, and `clean` is the conjunction:

```
all turns                              21,596 (100.0%)
opens an utterance, not mid-word       13,562 ( 62.8%)
+ transcriber closed the utterance      9,346 ( 43.3%)
+ other speaker not talking across it   5,502 ( 25.5%)
+ handover not overlapped (eot >= 0)    5,400 ( 25.0%)
+ at least 1.0 s of speech              4,891 ( 22.6%)
+ transcript rate >= 1.0 word/s         4,774 ( 22.1%)   <= clean
```

**4,774 turns, 14.00 h**, median 5.02 s and 13.0 words, 2,113 carrying a gold
label. Split 3,024 / 482 / 1,268.

Every threshold is one this project already validated. `speech_crosstalk <= 0.05`
is `rules.MAX_CROSSTALK`; `duration >= 1.0` is `rules.MIN_PREV_DUR`;
`starts_segment` / `ends_segment` are the transcriber witness that separates
`change` from `change_midseg`. The handover gate is `eot_gap >= 0`, **not**
`>= 0.2`: `rules.POS_MIN_GAP` is 0.0 deliberately, because fast handovers are
"precisely the cases a slow endpointer gets wrong", and a 200 ms floor drops
339 of them. What is excluded is a *negative* gap, where the next speaker
started first and there is no end-of-turn silence to detect at all.

`word_rate` is the one new gate. The corpus sits at a median of 2.66 words per
second of speech with p5 at 1.48, and the rows below 1.0 are a separate mode at
~0.08 — two different defects, both disqualifying:

- apportionment breaking down, where several spans each take a fractional slice
  of one long segment and every slice rounds to the same single word.
  `ta_IN_9829427_20230331_L_0183` is 13.18 s of speech rendered as `"okay"`.
- a speaker who never takes the floor, just acknowledging for fifteen seconds
  (`"ஹம் ஹம் ஹம் ஹம்"`). Real audio, not a turn.

All 21,596 rows stay in `data/gold/turns.jsonl` with the flags on them, so the
full floor structure is still there for duplex work and nothing has to be
rebuilt. **What publishes is the `clean` subset**, all 4,774 rows — 12× the
~400 any benchmark needs.

### The long tail is real

2.13% of turns run past 30 s and hold 7.36 h — a fifth of the audio in 2% of
the rows. It is not a segmentation artefact. `ta_IN_10255852_20230117` is a
20-minute call that is two sequential monologues (Left 6.75–607.01 s, Right
607–1204.83 s) at **0.5%** overlap, and its two 600 s turns are a correct
reading of it.

The lapse split does not touch it either — those long turns have no pause over
2 s, they are continuous speech. So no *length* filter goes into the corpus.
Length is not a defect; `clean` gates on structure, and a 410 s monologue that
opens and closes an utterance cleanly is a correct row.

### The trailing window is not silent, and should not be

`trail_s` is the owner's own leg, after they stopped, with their next speech
excluded by assertion — so it should be near-silent, and mostly is: over 800
sampled turns the trail peaks at **1.2%** of the turn's own peak (median), 9.1%
at p90. But 4.4% exceed 20%, and that is the decay of the speaker's last
syllable, which Silero cuts at `thr_off` while it is still audible.

That is the right thing to ship — it is what a deployed detector hears — but it
means a nonzero trail is normal and **one loud sample is not evidence of a
bug**. `24_verify_turns.py --acoustic N` reports the distribution rather than
asserting a threshold, because there is no correct one.

## Why not the boundary table

Three separate failures, each measured against the corpus. Kept because each one
produces **plausible, confident, wrong** rows rather than an error — the
`pitfalls.md` pattern.

**1. The gated `change` rows are 29% of the floor changes.** The funnel that
built `clips` dropped 14,273 of 20,013 floor changes (`drop_short_prev`,
`drop_straddle`, `drop_bargein`, `drop_pos_backchannel`…). Grouping only what
survived means every dropped change is swallowed *inside* a turn, so intervals
where **the other speaker held the floor** are emitted as this speaker's
mid-turn hesitations:

```
gated-row turns            : 5,740
correct (0 swallowed)      : 1,943  (33.9%)
WRONG (>=1 swallowed)      : 3,797  (66.1%)
worst case                 : 60 real floor changes inside one "turn"
```

**2. `start_time` = "previous speaker-change on this leg" starts the turn before
the owner has spoken.** It attributes the previous speaker's entire hold to this
turn as leading dead air:

```
lead-in dead air   median 0.99s   mean 1.99s   p90 4.45s
leads > 1 s : 9,898 (49.5%)      leads > 5 s : 1,622 (8.1%)
total audio  36.61 h  ->  25.57 h using the owner's first span  (30.2% dead air)
```

**3. Even with the full inventory, `CHANGE` boundaries miss real turn ends.**
`turns.boundaries()` classifies a boundary by comparing which leg speaks *next*,
so when speech overlaps it reads a genuine floor change as a `HOLD`. The floor
therefore fails to alternate on **5,006** of 19,897 consecutive change pairs
(25.2%), and those gaps are not backchannels:

| | n | other-leg speech between the two changes |
|---|---|---|
| same leg (L→L) | 5,006 | median 1.47 s / 3.6 words · p90 6.94 s / 18.1 words · only 19.6% silent |
| alternating (L→R) | 14,891 | median 0.00 s / 0.0 words · 83.4% silent |

Sampled text from the same-leg interstitials is full sentences —
*"இல்ல, உங்களுக்கு ஏதாவது idea இருக்கான்னு, கேட்டேன் sir நமக்க"* (9 words, 3.2 s).
Those are turns. A `CHANGE`-keyed definition drops them.

Floor runs have none of these failures because they never consult the
classifier. The classifier stays exactly where it is validated — deciding
`clips` labels.

## Two hard rules

**Carry `split` through unchanged.** The test split has been frozen since before
training and the paper depends on it. If turns reshuffle calls, anyone comparing
against published numbers gets contaminated results, and two datasets built from
the same 116 calls disagree about what "test" means. `split` is a property of
the **call**, so it transfers without ambiguity — assert that no call appears in
two splits.

**Validate before publishing.** The replay-pump bug cost 26 accuracy points and
produced entirely plausible numbers. `24_verify_turns.py` runs 26 checks over
every row and re-derives its facts from the corpus, so a bug in the builder
cannot hide itself. It earned its keep on the first run: **the totality check
caught a data-loss bug that had silently deleted 4,572 spans — 169 minutes of
real speech**, including one continuous 10.37 s utterance. See `pitfalls.md`.

- **no speech lost / no speech double-counted** — every confirmed span is
  covered by exactly one same-side turn, or is a backchannel that began inside
  an other-side turn
- absorbed speech really passes the backchannel gate
- every `silence_span` lies inside `[start_time, end_time]`, ordered, disjoint, non-empty
- `duration == end_time - start_time`; `eot_gap` agrees with the next turn's start
- the audio window lies inside the leg WAV — no silent zero-padding
- no owner-leg speech inside the trailing window
- `turn_index` contiguous and time-ordered; `id` unique; joined `sid` decodes
  to this exact turn end
- every call's rows carry one `split`, and the 71/15/30 call counts are unchanged
- **no pause hides a floor change** and **no pause is a lapse** — the two
  guarantees `_split_lapses` makes, checked on all 23,477 spans
- `starts_segment`, `ends_segment`, `pause_crosstalk`, `speech_crosstalk`,
  `speech_s` and `word_rate` all re-derived from the corpus; `clean` agrees
  with its six gates
- `--listen N --clean` cuts wavs to spot-listen; `--acoustic N` reports trail
  loudness. **Listen to the `--clean` pool.** A spot-listen of the full table
  keeps turning up fragments and pauses with the other speaker in them, because
  77.9% of rows are not clean — that is what the flag is for, not a defect.

## Resolved: `words`

**Omit word alignments.** SPRING-INX ships segment-level transcripts, and
`04_refresh_text.py` *apportions* words across VAD speech seconds — an estimate,
not forced alignment. `n_words` ships as a fractional count because the
backchannel gate already depends on it and it is honest about being approximate,
but no per-word timings ship. Audio-only consumers, including our own models and
the eot-bench SmartTurn adapter, never read them.

Revisit only if a consumer asks; force-alignment (MMS / WhisperX on Tamil) is
1–3 days and unverified on narrowband Tamil.

## Format compatibility

The span layout follows the convention `livekit/eot-bench` uses, so the dataset
is consumable by that kind of harness without a conversion step. Two details of
that convention shape the published schema:

**The last span is the end-of-turn.** Consumers of this layout read
`silence_spans[-1]` as the turn ending and every earlier entry as a mid-turn
hold. The published rows therefore fold the end-of-turn back into
`silence_spans` — the one place the published schema departs from the local
table, where the two are kept separate so `24_verify_turns.py` can assert that
every span lies inside its turn.

**Spans are clip-relative.** Time 0 is the first audio sample of the row, not
call time.

`eot_bench_eligible` marks the 4,322 rows that satisfy the usual release rules
for this format — final silence ≥ 0.2 s so every turn is scorable at a 0.2 s
score point, and no span longer than 5.0 s. The other 452 rows are fast
handovers and long lapses, kept in the dataset because interruption and timing
work wants exactly those.

**No benchmark results are claimed here.** This section describes the shipped
format only.

Malayalam R1 (22 calls) would clear the same bar and would ship as its own repo.

## Running it

`paths.py` defaults `SPRING_INX_R1` to `PROJECT.parent/SPRING_INX_Tamil_R1`,
which no longer resolves — the corpus sits beside `Project_Tamil_EOT`, not
inside it. It fails as **"0 calls"**, not as an error:

```bash
export SPRING_INX_R1=/home/santhosh/Desktop/SPRING_INX_Tamil_R1
python pipeline/23_build_turns.py      # ~1 min, VAD is cached
python pipeline/24_verify_turns.py     # must pass before packing
```
