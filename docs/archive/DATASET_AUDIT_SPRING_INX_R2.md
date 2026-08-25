# SPRING_INX Tamil R2 — structural audit and the revised phase-1 pipeline

> **Partly superseded — see `DATASET_AUDIT_SPRING_INX_R1.md`.**
> R1 contains 116 real call-centre calls split one-speaker-per-file
> (`ta_IN_*_Left/_Right`), which provide exact speaker labels for free. The
> "no speaker labels" blocker in §2 and the diarization pipeline in §4 below
> still apply to R2's 403 conversational recordings, but they are no longer
> the starting point, and pyannote's error rate is now measurable rather than
> assumed. The R2 measurements themselves stand unchanged.

Audited the actual download, not the paper. 545 wav files, 17 GB on disk,
16 kHz / mono / 16-bit, ~156 hours. Everything below is measured.

---

## 1. What the release actually contains

```
SPRING_INX_Tamil_R2/
├── Audio/            545 × wav, 16 kHz mono, 11–30 min each, 156.2 h
├── train/  dev/  eval/
     segments  text  utt2dur  utt2spk  spk2utt  wav.scp
```

**The filename encodes the mode** — this is the number the paper never published:

| Prefix | Meaning | Recordings | Audio hours | Segments |
|---|---|---|---|---|
| `ta_SHA1P_C2_*` | conversation, 2 speakers | 293 | 84.1 | 45,483 |
| `ta_SHA1P_C3_*` | conversation, 3 speakers | 77 | 31.4 | 10,995 |
| `ta_SHA1P_C_*` | conversation (unnumbered) | 33 | 11.2 | 6,625 |
| `ta_SHA1P_M_*` | **monologue** (read/extempore) | 142 | 29.5 | 8,449 |
| | **conversational total** | **403** | **126.7** | **63,103** |

**81% of the corpus is conversational.** My earlier estimate assumed 30%. The
data budget is roughly 2.7× better than planned.

Total segmented speech 148.3 h; median per-recording coverage 95.7%; no
recording is unsegmented.

---

## 2. The blocker: there are no speaker labels

```
$ head -3 train/utt2spk
utt00000000 utt00000000
utt00000001 utt00000001
utt00000002 utt00000002
```

100% of rows have `speaker_id == utt_id`. That is the Kaldi convention for
*speaker information unavailable*. `spk2utt` is the same file transposed and
equally empty. 702 speakers exist in the corpus; **not one of them is
identified in this release.**

Second correction to my earlier read: the `segments` second field is the
**recording id**, not the speaker id — standard Kaldi
`<utt_id> <reco_id> <start> <end>`. I misread the paper's description.

**Consequence: the diarization oracle from §6 of the plan does not exist.**
We have turn *boundaries* but not turn *ownership*. Positives — "A finished,
B took the floor" — are undecidable from the shipped files alone.

### Also: the shipped splits are unusable

511 of 545 recordings appear in **both** `train` and `eval`. The splits are
utterance-level random draws from the same recordings, so the same voices,
same rooms and same conversations sit on both sides. Fine for the ASR
benchmark it was built for, fatal for us — a turn detector would memorise
voices. **Ignore the shipped splits; re-split by recording.**

---

## 3. What survived, and it is the important part

### The acoustic pause is real and recoverable

The segment timestamps are butt-joined — 90.6% of consecutive conversational
segments have a gap under 10 ms, so the file looks like a contiguous partition
with the pauses written out. **They are written out of the timestamps, not out
of the audio.** Energy probe over 15 random conversational recordings, 1,944
butt-joined boundaries:

| silence found within ±1.2 s of the boundary | share |
|---|---|
| < 100 ms | 4.7% |
| 100–200 ms | 8.7% |
| 200–350 ms | 13.7% |
| 350–600 ms | 25.0% |
| 600–1000 ms | 23.0% |
| 1000–1500 ms | 22.8% |
| > 1500 ms | 2.0% |

Median 560 ms. **95% of boundaries have a locatable pause, and 93% fall inside
the 0–1500 ms window the positive class requires.** A VAD pass recovers the
exact offset. The rounding is a bookkeeping artefact, not data loss.

### Negative-class supply is not a constraint

Pauses ≥250 ms *inside* a single segment — same speaker by construction, no
diarization needed:

| pause length | share |
|---|---|
| 250–400 ms | 30.8% |
| 400–600 ms | 30.5% |
| 600–1000 ms | 20.2% |
| 1–2 s | 14.9% |
| > 2 s | 3.6% |

217 per recording → **~88,000 across the 403 conversational recordings**, plus
the 142 monologues (guaranteed single-speaker, 29.5 h). Even at 10% survival
after word-boundary and sentence-final filtering that is ~8,800 negatives.

### The far end of the call *is* transcribed

I checked whether these are one-sided phone recordings with an inaudible
counterpart. They are not. Segments ≥3 s: median voiced ratio 0.74, median
1.93 words/sec — normal conversational density. Only 1.6% fall below 0.3
words/sec, and those have a *low* voiced ratio (median 0.27), i.e. they are
genuinely mostly silence, not untranscribed speech. Both sides are present and
both sides are transcribed.

Median voiced ratio of 0.74 also means **26% of segment time is non-speech** —
these mock conversations kept their hesitations. The concern that prepared
speech would flatten the negative class does not show up in the measurements.

### Backchannels are abundant and must be filtered

4,954 conversational segments under 1.2 s: `உம்`, `okay`, `ஆஹ்`, `தெரிலடா`,
`அடுத்தது`. These are exactly the floor-not-taken cases that turn a positive
into a false positive. The `>1 s, >3 words` rule in §6 stands, and this corpus
is where it earns its keep.

---

## 4. Revised pipeline

The plan changes in one place: **we build the speaker labels ourselves.** The
filename hands us the hardest hyperparameter for free — the speaker count.

**Step 0 — get R1 as well** (the 18 GB archive). This is R2 only; the paper's
226 h Tamil figure means ~70 h more sits in R1. Same audit script applies.

**Step 1 — re-split by recording.** Discard `train/dev/eval`. Hold out whole
recordings, never utterances.

**Step 2 — VAD pass** (Silero VAD, CPU, ~real-time × 100). All 403
conversational recordings → precise speech/silence timeline. Gives the true
pause offsets the butt-joined timestamps hide, and the measured gap length
each label rule needs.

**Step 3 — diarization pass** (pyannote 3.1) with `num_speakers` pinned from
the filename: `C2`→2, `C3`→3, `M`→1. Pinned speaker count is the single
biggest accuracy lever in diarization, and we get it from a string split.
Restrict output to the segment boundaries already known — this is speaker
*assignment*, not blind diarization, and far easier.

**Step 4 — LLM cross-check on the transcripts.** For each boundary, does the
text imply a speaker change (question→answer, register shift, address terms)?
Agreement with pyannote → keep. Disagreement → drop. Two independent signals,
and the disagreement rate is a free quality metric. **This is the LLM's correct
role — it never sees a prosody question.**

**Step 5 — positives.** Speaker change, VAD gap 0–1500 ms, no overlap, next
speaker's turn ≥1 s and ≥3 words, previous speaker does not resume within ~3 s.
Take the last ≤8 s of A, trim to ~200 ms trailing silence.

**Step 6 — negatives.** VAD pause ≥250 ms inside one diarized speaker region
where that speaker resumes. Force-align (WhisperX or MMS Tamil) and cut at a
word boundary — mandatory. Reject where the final word is a sentence-final
form. Cap the monologue share so read-speech prosody does not dominate.

**Step 7 — gold set.** Hand-label ~200 boundaries by ear and score the
pipeline against them. Without this the label accuracy is unknown and every
downstream number is unfalsifiable. This is listening, not recording.

**In parallel — email SPRING Lab.** Two asks: the original 8 kHz narrowband
audio they offer on request (better than upsampled for a telephony detector),
and the diarization metadata the paper says is planned. **If they release
speaker labels, steps 3 and 4 disappear.** One email is worth a week of work.

---

## 5. Supply estimate

| | count |
|---|---|
| conversational boundaries in R2 | 63,103 |
| × speaker-change rate (assume 50–60%) | ~32–38k |
| × gap in 0–1500 ms (measured 93%) | ~30–35k |
| × survives overlap / backchannel / resume filters (assume 35%) | **~10–12k positives** |
| internal pauses ≥250 ms | ~88,000 |
| × word-boundary + sentence-final filters (assume 10%) | **~8,800 negatives** |

Roughly balanced, and comparable to Smart Turn's ~12k per language — from one
CC BY 4.0 corpus, before R1 is even opened.

---

## 6. What is now uncertain

1. **pyannote's DER on this audio.** Indian-accented, upsampled-from-narrowband,
   mock-conversation speech is not its training distribution. Measure it on the
   gold set before trusting step 3. This is the project's main technical risk.
2. **Do segments ever span two speakers?** If yes, some step-6 negatives are
   actually speaker changes. The diarization pass answers this; quantify it.
3. **Gender skew** (160 M / 542 F for Tamil, per the paper) — report per-gender
   metrics rather than averaging over it.
4. **Mono channel** — overlapping speech is mixed and cannot be separated.
   Overlap-flagged boundaries get dropped, not repaired.

---

## 7. Verdict

`segments` shipped, transcripts are clean, the pauses are in the audio, 81% of
the corpus is conversational, and the licence is CC BY 4.0. The one missing
piece — speaker identity — is recoverable with a pinned-speaker-count
diarization pass plus a text cross-check, and might arrive for free by email.

**Data is no longer the bottleneck. Label accuracy is.** Steps 3, 4 and 7 are
the whole of phase 1 now.
