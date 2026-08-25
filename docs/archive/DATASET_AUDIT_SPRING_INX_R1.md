# SPRING_INX Tamil R1 — audit, and the revised phase-1 plan for R1+R2

Supersedes the pipeline section of `DATASET_AUDIT_SPRING_INX_R2.md`.
R1 contains something R2 does not, and it removes the main technical risk of
the project.

985 recordings, 29 GB, 16 kHz mono. All measurements below are from the files.

---

## 1. R1 has four families, not one

| Family | Recordings | Segmented h | Speaker labels | What it is |
|---|---|---|---|---|
| **`ta_IN_*_Left/_Right`** | 232 files = **116 calls** | 39.7 | **YES — exact, free** | real call-centre calls, one file per speaker |
| `ta_SHA1P_C_*` | 319 | 88.3 | no | mock conversations |
| `ta_SHA1P_M_*` | 414 | 93.1 | n/a (single speaker) | monologue / readings |
| `ta_CRESC_C_*` | 20 | 4.8 | no | conversation |

`utt2spk` is degenerate in R1 exactly as in R2 — 94,281 of 94,281 rows have
`speaker_id == utt_id`. **But the `ta_IN_` family does not need it.**

---

## 2. The `ta_IN_` family is a speaker oracle

Same call, split into two audio files, one speaker per channel. Verified, not
assumed:

- **116 / 116** base names have both a `_Left` and a `_Right` file
- **0 ms** duration mismatch between the legs — sample-aligned, no drift
- mean simultaneous-speech fraction **0.094** — the channels are not the same mix
- frame-energy correlation between legs is **negative on every pair**
  (−0.08 to −0.37) — the anti-correlation signature of turn-taking
- **each leg is separately transcribed with its own timestamps**

```
Left  [   4.23-  12.68] நாகர்கோவில் tourist service க்கு call பண்ணதுக்கு மிக்க நன்றி
Right [   9.33-  10.10] hello
Right [  12.78-  15.00] ஆ mam, நான் சென்னையில இருந்து call பண்றேன்
Left  [  15.41-  17.23] ஆ, okay sir சொல்லுங்க
Right [  17.03-  19.92] mam எனக்கு வந்து, நாகர்கோயிலை full ஆ, சுத்தி பாக்கணும்.
```

Merging the two legs by timestamp yields a **human-transcribed, speaker-attributed
turn sequence with exact ground truth.** No pyannote, no DER, no LLM
cross-check. The label is a property of which file the audio came from.

These are also genuine call-centre calls — CRM-style ids and dates
(`ta_IN_10102450_20230112`), tourist-service bookings, agent/customer register.
That is the exact deployment domain for an Indic voice agent.

### They are real telephony

Spectral check on the `ta_IN_` legs: **0.00% of energy above 4 kHz**, hard cut
at 3.4 kHz. Pure narrowband G.711-band telephony, upsampled to 16 kHz.

**This retires the 8 kHz request in your email.** The band-limiting is already
baked into the samples, upsampling added no information and destroyed none, and
Smart Turn's Whisper encoder wants 16 kHz input anyway. The release format is
already the right one. Only the speaker metadata for the `SHA1P` families is
still worth chasing — and for `ta_IN_` you no longer need it at all.

---

## 3. Yield from the 116 stereo calls

16,067 speaker-change transitions. Gap distribution at those transitions:

| | share |
|---|---|
| overlap > 1 s | 41.2% |
| overlap 0.5–1 s | 5.4% |
| overlap 250–500 ms | 3.5% |
| overlap < 250 ms (normal transition) | 4.8% |
| gap 0–250 ms | 6.5% |
| gap 250–750 ms | 11.1% |
| gap 0.75–1.5 s | 11.5% |
| gap > 1.5 s | 15.8% |

Applying the §6 rules — speaker change, gap ≤ 1.5 s, next turn ≥ 1 s and
≥ 3 words, previous turn ≥ 1 s — with a 250 ms tolerance for the small overlap
that is normal at a healthy transition:

| overlap tolerance | positives |
|---|---|
| 0 ms | 3,866 |
| 150 ms | 4,245 |
| **250 ms** | **4,474** |
| 500 ms | 4,894 |

**4,474 exact-ground-truth positives**, median gap 500 ms. Negatives from
between-segment holds where the other channel stays silent: **1,041**
(median 810 ms); pauses *inside* a segment need the VAD pass and add more.

### That 4,474 is a floor, and here is why

Cross-channel overlap measured at the **segment** level is 28.1% of speech, but
at the **VAD** level only 9.4%. The transcript segments are padded — they
include lead-in and trailing silence. So a large share of the 6,617 transitions
rejected as ">1 s overlap" are not interruptions at all, just two padded
segments touching. **Tightening segment edges with VAD before applying the gap
rule will recover a meaningful fraction of them.** Expect the real number to
land above 4,474, not below.

---

## 4. What this changes: the main risk becomes measurable

The R2 audit named the project's biggest risk as *"pyannote's DER on this
audio is unknown, and it is outside its training distribution."* That risk is
now retired, because the stereo calls give a way to measure it on this exact
corpus:

1. Sum `_Left + _Right` to a single mono mix — a synthetic single-channel
   recording whose true diarization you already know exactly.
2. Run pyannote 3.1 on the mix with `num_speakers=2`.
3. Score against the channel-derived truth. **You now have a measured DER on
   Tamil call-centre telephony**, not a guess.
4. Apply pyannote to the 722 label-less conversational recordings
   (319 R1 `SHA1P_C` + 403 R2 `C/C2/C3`, ~208 h) with that error rate known,
   and set the confidence threshold from the calibration.

Nothing else in the corpus offers this. It converts the diarization step from
an act of faith into a measured component with an error bar — which is also the
single most defensible thing you can put in the writeup.

---

## 5. Combined inventory, R1 + R2

| | recordings | segmented hours |
|---|---|---|
| **stereo call legs (exact labels)** | 116 calls | 39.7 |
| conversational, labels to be inferred | 722 | ~208 |
| monologue (single speaker by definition) | 556 | ~122 |
| **total** | **1,530** | **~254 h** |

46 GB, one licence, CC BY 4.0.

---

## 6. Revised plan

**1a — Build the gold set. No ML required.** Merge the 116 stereo calls,
Silero VAD to tighten segment edges, apply the positive/negative rules. Output
~4.5 k+ positives and ~1–3 k negatives with exact labels. This is a scripting
job, and it is the part of the project that cannot fail.

**1b — Reserve ~30 of the 116 calls as the held-out test set.** Exact labels,
real BPO calls, correct telephony band. This is the most defensible evaluation
set available anywhere for Tamil EOT, and it should never enter training.

**1c — Calibrate diarization** on the mixed-down remainder (§4). Report DER.

**1d — Scale out** to the 722 conversational recordings with the calibrated
pipeline plus the LLM transcript cross-check, discarding low-confidence
boundaries.

**1e — Negatives at scale** via VAD pauses inside single-speaker regions;
force-align (WhisperX / MMS Tamil) and cut at word boundaries; cap the
monologue share so read-speech prosody does not dominate.

**1f — Hand-check 200 boundaries by ear** against the pipeline's labels.

**Split by call/recording throughout.** Both releases ship `train`/`dev`/`eval`
splits that share recordings across sides — 511 of 545 recordings in R2 appear
in both train and eval. Ignore them entirely.

---

## 7. Status

- Data: sufficient, legally clean, and better matched to the deployment domain
  than anything that could have been scraped.
- Labels: exact for 116 real call-centre calls; inferable with a *measured*
  error rate for another ~208 h.
- Email to SPRING Lab: no longer blocking. The 8 kHz ask is moot; the speaker
  metadata would only accelerate step 1d.

**Step 1a is unblocked and depends on nothing. Start there.**
