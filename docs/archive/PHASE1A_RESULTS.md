# Step 1a — the Tamil EOT gold set

Built from the 116 `ta_IN_` stereo calls in `SPRING_INX_Tamil_R1`. Every
number here is produced by `pipeline/0*.py` and reproducible from the corpus.

> **Superseded in part — read §9 first.** The negative class described below
> was measured at 43.4% by ear, withdrawn, and has since been rebuilt by an
> audio labeller measured at 95.9%. The 2,909 held-back samples were cut and
> labelled in the same pass. Current set: **16,213 samples, 10,493 complete /
> 5,720 incomplete**, in `data/gold/samples_llm.jsonl`. §1–§8 record how the
> set was built and what broke, which is still the reason it is trustworthy;
> §9 records the fix.

```
 3,901 complete          verified 95.9% against a human listener
 9,406 incomplete        WITHDRAWN at 43.4% -- rebuilt, see §9
 2,909 held back         labelled, tagged, out of the core set -- now promoted
   116 calls             all of them contributed
    30 calls             held out as test, never trained on
```

For scale: Smart Turn ships roughly 12k samples per language. This is one
corpus, one licence (CC BY 4.0), and no diarization anywhere in the chain.

---

## 1. Where the labels come from

The `ta_IN_` family is 116 real Tamil call-centre calls shipped **one audio
file per speaker**, each leg separately transcribed. Which file the audio came
from *is* the speaker label. There is no diarizer in this pipeline and
therefore no diarization error to propagate.

On top of that structural fact, every boundary is judged by **two independent
witnesses**:

| | acoustics (VAD) | transcriber | verdict |
|---|---|---|---|
| `change` | other leg takes the floor | closed the segment here | **complete** — 3,901 |
| `hold_intra` | same leg resumes | kept the segment open | **incomplete** — 9,406 |
| `change_midseg` | other leg takes the floor | segment still running | held back — 850 |
| `hold_inter` | same leg resumes | closed the segment | held back — 2,059 |

The core set is where the two agree. The other 2,909 are where they disagree:
they are kept, labelled and tagged, but excluded, because that is precisely
where a confident label would be a guess. `change_midseg` is somebody being
interrupted; `hold_inter` may be a finished sentence followed by another one,
and promoting it needs a Tamil sentence-final filter that does not exist yet.

That the two witnesses correlate at all is itself evidence. Sorting floor
changes by how long the reply was, the transcriber's agreement rises
monotonically — 33.8% for replies under 0.5 s, 39.0%, 59.8%, and 85.0% for
replies over 2 s. A backchannel and a real turn are different things, and two
unrelated signals say so together.

---

## 2. Three measurements that changed the design

**The transcript timestamps are padded, and using them would have been wrong.**
Summed across both legs the segments cover 107% of the call wall clock, which
is only possible if they carry lead-in and trailing silence. Measured at the
segment level, simultaneous speech looks like 28.1% of speech. Measured on the
VAD, it is **5.7%**. So turn boundaries here come from the VAD, and the
transcript is used only to confirm speech and supply text.

**The barge-in gate I started with was throwing away a third of the corpus.**
Rejecting any boundary with other-leg speech in the preceding second dropped
8,600 of 50,532 boundaries. Checking it: the floor-change rate is flat at
36–42% across every level of prior other-leg speech, and transcriber agreement
is *higher* where there was overlap (44–47% vs 38%). Speech before a boundary
is backchannel-rich conversation, not interruption. Replaced with the narrow
version — the other leg mid-span at the instant of the cut, which is both a
genuine collision and the case where the measured gap would be meaningless.
Recovered ~2,100 boundaries and 1,123 samples.

**Crosstalk bleed is real but small.** A VAD span on a leg with no transcript
over it is echo from the other side; 4,912 such spans, 0.69 h, **2.1% of VAD
speech**, dropped. This is what makes it safe to cut clips from a single leg —
which is also what a deployed detector receives, since a voice agent's turn
detector sees the inbound user stream and not a mixdown.

---

## 3. Trying to break it

`pipeline/09_verify.py` does not report accuracy. A turn-detection set is easy
to build wrong: if the classes differ in loudness, clip length or trailing
silence, a model learns that instead of prosody, scores well offline and
collapses in a live call. So the script attacks the set.

- **Leakage** — 0 calls in more than one split, 0 duplicate ids.
- **Geometry** — every crude audio statistic separates the classes by less
  than 0.5 standard deviations. Trailing-200 ms energy: d = −0.14, so
  trailing silence carries no label information. Both classes are cut with
  identical 200 ms of real trailing audio, deliberately.
- **Shortcut probe** — a logistic regression on 10 cheap global features
  (energy, duration, spectral tilt, voiced fraction), trained on train calls
  and scored on held-out test calls: **AUC 0.601, accuracy 0.701 against a
  majority-class baseline of 0.702.** A model that cannot hear prosody cannot
  do better than guessing the common class. The label is in the speech.

### The one confound worth knowing about

`prev_dur` — how long the speaker had been talking into the cut — is 4.57 s
for complete and 6.49 s for incomplete, d = −0.33, and it leaks into the clip
as voiced fraction (d = −0.46, the largest audio effect measured). This is
**induced by the gate, not by the phenomenon**: `hold_intra` requires the pause
to fall inside one transcript segment, so it is drawn from longer utterances by
construction. The shortcut probe bounds the damage at AUC 0.601. Phase 2 should
sample matched on `prev_dur` rather than ignore it.

`gap` differs more (d = +0.57) but is metadata only — it is never inside the
clip, so no model can read it.

---

## 4. The held-out test set

**30 calls, 3,439 samples (1,024 complete / 2,415 incomplete), never trained on.**

Exact channel-derived labels, real Tamil BPO calls, native narrowband
telephony. Split by call throughout; both SPRING_INX releases ship
utterance-level splits that put 511 of 545 recordings on both sides, and those
are ignored entirely.

Two corrections to the R1 audit, from re-measuring here. The energy above
4 kHz is **0.14% on average and up to 0.96%**, not the 0.00% that audit
reported — still overwhelmingly G.711-band, so the conclusion is unchanged
(the 8 kHz request in the unanswered email is moot, and Smart Turn wants
16 kHz anyway), but the number itself was wrong. Separately, **3.7% of clips
contain digital clipping** (peak ≥ 0.999); that is ordinary for telephony
capture and affects no label, but it should be in the model card rather than
discovered later.

As far as I can tell nothing comparable exists publicly for Tamil end-of-turn
detection. It is worth more than the model that gets trained on the rest.

---

## 5. Verify the labels before trusting any of this

Everything above rests on one claim: that a channel-derived label matches what
a Tamil speaker hears. Nobody should take that on trust, and you are the one
person on this project who can check it.

```bash
.venv/bin/python pipeline/10_qa_sample.py -n 200
```

198 clips are already drawn — balanced 99/99, stratified across gap length,
from 66 calls, and taken only from `train` so the test set stays sealed.

```
reports/qa/blind/001.wav ...   listen
reports/qa/sheet.tsv           write complete / incomplete / unsure
reports/qa/answers.tsv         open only afterwards
```

An hour of listening turns "the labels are exact by construction" into a
measured number, and that number is the thing to put in front of Sarvam.

**Expected failure mode to watch for:** an `hold_intra` where the speaker had
in fact finished a sentence and then started another. The gate uses the
transcriber's segment as the arbiter, and a segment can hold more than one
sentence. If those show up often in the blind sample, the fix is a Tamil
sentence-final filter — which would also unlock the 2,059 held-back
`hold_inter` samples.

---

## 6. State of the plan

| | |
|---|---|
| 1a build the gold set | **done** — 13,307 samples |
| 1b reserve ~30 test calls | **done** — 30 calls, sealed |
| 1f measure label accuracy by ear | ready, 198 clips drawn, needs you |
| 1c calibrate diarization DER | not started |
| 1d scale to the 722 label-less recordings | not started |
| 1e negatives via forced alignment | not started |

1c–1e are what take this from 13k samples to the ~208 h of conversational
audio in R1+R2 that has no speaker labels. None of them are blocked. But the
honest sequencing is 1f first: if the labels are wrong, scaling multiplies the
error, and everything downstream inherits it.

---

## 7. The blind listening pass — result

198 clips, labelled by ear with the answers sealed. This is the section that
matters, because it is the only part of the project that was tested rather
than argued.

| source | n | agreement |
|---|---|---|
| `change` (complete) | 98 | **95.9%** |
| `hold_intra` (incomplete) | 99 | **43.4%** — below chance |

56 of 99 negatives were heard as complete, and agreement *falls* as the pause
lengthens: 48.5% under 0.35 s, 42.4%, 39.4% over 0.8 s.

**Why the two classes came apart.** The positive class is grounded in
behaviour recorded in the call — the other person heard the speaker finish and
took the floor with a substantive turn. That is a real human judgement, and it
survives contact with another human's ears. The negative class was grounded in
"a pause occurred inside a transcript segment", which is not a judgement about
completeness at all. It scores at chance because it *is* chance: pauses follow
finished sentences roughly as often as they land mid-clause.

The §1 claim of "two independent witnesses" was therefore half right. For
positives the second witness is real. For negatives there was only ever one
witness, and it was not answering the question.

**Three fixes tried, all failed.**

1. *Structural* — no re-derivation from segment boundaries helps, because the
   boundary encodes no completeness judgement to begin with.
2. *Textual* — Tamil finite-verb morphology at the cut point appears in 28.6%
   of the clips heard as complete and 30.2% of those heard as incomplete. Zero
   signal. **The reason given here was wrong, and the bug it was hiding is now
   fixed.** The shipped transcripts are accurate at the word level — the
   "substantial word errors" claim does not survive a closer read. The defect
   was ours: `_apportion()` spread a segment's words evenly over *elapsed*
   time, but measured across 30 calls the median transcript segment is **21%
   silence** (p10: 53%). Linear interpolation therefore handed about a fifth of
   the words to intervals where nobody was speaking — worst exactly where
   `hold_intra` lives, because a mid-segment pause *is* that silence.

   `_apportion()` now weights by VAD speech seconds instead.
   `pipeline/04_refresh_text.py` applied it to the existing samples: **61.7% of
   texts changed and 38.1% had different final three words.** The old text was
   wrong that often.

   Re-running the test on the corrected cut, honestly: **it did not help.** The
   final word moved on 32% of the QA clips, and a leave-one-out vote on word
   -final suffixes went 72.1% → **72.6%**, both *below* the 76.1% majority
   baseline. Fixing the cut was necessary and it was real; it was not
   sufficient. A 2-character suffix vote is a weak probe, so "can a Tamil
   -capable LLM judge completeness from the corrected text" is still open — but
   the cheap version of the textual route is now properly dead rather than
   merely suspected.
3. *Metadata* — every feature the pipeline computes, tested against the human
   verdicts. Best singles are `span_words` (d = 0.61) and `span_dur` (d = 0.58);
   all of them combined reach **AUC 0.637**, far short of a usable filter.

**Note this also explains the §3 shortcut probe.** AUC 0.601 was read as "the
classes are not trivially separable". Part of that was really "the negative
class is half mislabelled", which caps any classifier including a crude one. A
clean set should be re-probed once the negatives are rebuilt.

**The fix: label by listening, and prove the labeller first.**
`pipeline/11_label.py --validate` scores an audio-capable LLM against these
same 198 human labels, on both classes, before it is allowed to touch the
9,406-clip pool. The bar is 90%. The 198 clips are worth more as this
validation set than they were as a spot check.

One design decision worth stating: the labelling question is *"was the
utterance finished?"*, not *"would replying here be wrong?"*. A speaker who
completes a sentence and then continues produces audio acoustically identical
to one who completes a sentence and stops — no model can separate them, so
training on that distinction only adds noise. The listening convention used
for the 198 is both the right one and the only learnable one.

---

## 8. Honest state, as of the blind listening pass

- **Positives: 3,901, verified 95.9%.** Real, and the harder half to build.
- **Negatives: none usable.** 9,406 clips exist and are cut; they need
  relabelling, not rebuilding.
- **Test set: 30 sealed calls** — still valid, but its negatives inherit the
  same problem and must be relabelled too before it can score anything.
- 1c–1e (diarization calibration, scaling to the 722 label-less recordings,
  forced alignment) are unchanged and still unblocked — but scaling a broken
  negative class would only multiply the error, so 1g comes first.

---

## 9. 1g — the negative class, rebuilt

§7 ended with a plan: prove an audio labeller against the same 197 human
labels, then let it relabel. Both halves are now done.

**Choosing the labeller.** Seven Gemini models were run over the identical 197
clips with the identical prompt on Vertex AI batch, for $1.20 total.
`gemini-3.7-flash` won at **95.9% agreement, 93.9% on `hold_intra`, 89.8%
precision on `incomplete`** — against 43.4% for the pipeline it replaces. It
ties `gemini-3.1-pro-preview` exactly (paired McNemar, 5 discordant clips each
way, p = 1.000) at one seventh the price. Full table in
`reports/model_bakeoff.md`.

**Relabelling.** 16,019 clips in three batch jobs, all succeeded at 100%, 3
unparsed responses, **$5.69**. `pipeline/06_cut_heldback.py` cut the 2,909
held-back clips first so they could be judged too.

| source | n | flipped | new complete | new incomplete |
|---|---|---|---|---|
| `change` | 3,901 | 289 (7%) | 3,612 | 289 |
| `hold_intra` | 9,404 | 4,988 (53%) | 4,988 | 4,416 |
| `change_midseg` | 850 | 106 (12%) | 744 | 106 |
| `hold_inter` | 2,058 | 1,149 (56%) | 1,149 | 909 |

**53% of `hold_intra` flipped**, against 57% (56 of 99) in the human listening
pass. Two independent measurements of how wrong that class was, agreeing to
four points. §7's diagnosis was right.

**The `prev_dur` confound is gone.** §3 flagged it as the one confound worth
knowing about: `hold_intra` required a pause inside a transcript segment, so
the negative class was drawn from longer utterances *by construction*
(d = −0.33), and it leaked into the clip as voiced fraction (d = −0.46, the
largest audio effect measured). Re-measured on the relabelled set: **d = −0.09
and −0.17.** The gate no longer decides the label, so the gate's bias no longer
rides along with it. That was not the goal of the relabelling; it is a
consequence of grounding the label in something other than the gate.

**The shortcut probe went up, not down: AUC 0.601 → 0.622.** Worth stating
plainly. Both are inside the near-chance band, and §3's reading of 0.601 was
anyway partly "the negative class is half mislabelled", which caps any
classifier. The class balance moved at the same time (majority 0.702 → 0.631),
so AUC is the comparable number here and accuracy is not — the probe now beats
the majority class by 1.1 points where before it lost to it by 0.1.

**The held-back 2,909 are promoted.** §1 said `hold_inter` needed "a Tamil
sentence-final filter that does not exist yet". That is exactly what a
95.9%-agreeing audio labeller is. Including them leaves AUC unchanged at 0.622,
so they carry no shortcut of their own.

One thing §7 got wrong and this corrects: it asserted the labelling question
should be *"was the utterance finished?"* rather than *"would replying here be
wrong?"*, and called the second "not learnable". That reasoning stands, but the
stated justification — that the two produce acoustically identical audio — is
not what settled it. What settled it is that Smart Turn is pretrained on the
first question, and this data is going to fine-tune Smart Turn.

Detail, including the label-policy choice on the `change` class, is in
`reports/relabelling.md`.
