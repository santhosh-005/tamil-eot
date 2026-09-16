# SPRING_INX Malayalam R1 — audit

Companion to `DATASET_AUDIT_SPRING_INX_R1.md` (Tamil). Same question: does this
release contain the dual-channel family that makes the speaker label free?

**It does. 22 calls.** Every structural property the Tamil pipeline depends on
holds here. The corpus is ~5× smaller, which is what decides how it can be used.

29 GB, 21,966 audio files, 16 kHz. All measurements below are from the files.

---

## 1. Why it looks like it isn't there

The dual-channel calls are **44 files out of 21,966**, and 20,899 of the rest
are single-speaker monologue. A directory listing buries them completely.

| Family | Files | What it is | Speaker labels |
|---|---|---|---|
| `NNN_District_S_AA_monologue_*` | 20,899 | read / monologue, pre-segmented | n/a — single speaker |
| `ml_SHA1P_M_*` | 452 | monologue | n/a |
| other (`JV*`, `KSKL*`, …) | 438 | mixed | no |
| `ml_SHA1P_C_*` | 133 (9 recordings) | conversation | no |
| **`ml_IN_*_{Left,Right}`** | **44 = 22 calls** | **real call-centre calls** | **YES — exact, free** |

Unlike Tamil R1, most of this release ships **pre-segmented utterance files**
rather than whole recordings. The `ml_IN_` family is the exception: full-length
call legs, which is what the boundary extraction needs.

`utt2spk` is degenerate on the `ml_IN_` family — **4,824 / 4,824** rows have
`speaker_id == utt_id`, exactly as in Tamil. The channel is the only speaker
label, and it is sufficient.

---

## 2. The `ml_IN_` family is a speaker oracle — verified, not assumed

Same naming convention as Tamil (`ta_IN_10102450_20230112`): CRM-style id plus
date, `ml_IN_9828342_20230112`. Dates span 2023-01-12 to 2023-04-04.

| check | Tamil R1 | Malayalam R1 |
|---|---|---|
| complete `_Left`/`_Right` pairs | 116 / 116 | **22 / 22** |
| duration mismatch between legs | 0 ms | **0.0 ms on all 22** |
| sample rate / channels | 16 kHz mono | 16 kHz mono |
| energy above 4 kHz | 0.00% | **0.51% mean, 2.86% max** |
| leg frame-energy correlation | −0.08 to −0.37 | −0.555 to +0.120 |
| each leg separately transcribed | yes | **yes** |

**Narrowband telephony.** 0.51% of energy above 4 kHz. Higher than Tamil's hard
3.4 kHz cut but unambiguously band-limited — same acoustic condition, slightly
less aggressively filtered.

**Channel separation.** Frame-energy correlation between legs is negative on
19 of 22 calls, the anti-correlation signature of turn-taking. The three
exceptions are `9828342` (+0.120), `9828406` (+0.083) and `9828477` (+0.010).
`9828342` has heavy simultaneous speech in its transcript, so the positive
correlation is a conversational property rather than channel bleed. Measured on
the first 60 s of each call; worth re-checking full-length before use.

**Total audio.** 7.58 h per leg, 15.2 h across both.

---

## 3. Transcripts

4,824 segments on the `ml_IN_` family, **4,824 of which have transcript text**.
Both legs carry their own timestamps, and merging by time yields the turn
sequence directly:

```
Right [   0.77-   2.29] SBI customer care
Left  [   0.79-   3.81] ഫെഡറൽ ബാങ്ക് customer service ലേക്ക് സ്വാഗതം. പറഞ്ഞോളൂ
Right [   7.32-   8.89] റാം പറയൂ.
Right [  20.64-  24.77] okay ഉം ഉം റാം എന്താ ചെയ്യുന്നെ ഇപ്പൊ?
Left  [  22.30-  33.17] അപ്പം ആഹ് okay അപ്പം ഇത് ശരിക്കും PayPal ല്‍ …
```

Banking / customer-service call centre, code-switched Malayalam–English. The
same deployment domain as the Tamil tourist-service calls.

---

## 4. Yield

8.86 h of segmented speech (Left 4.37 h, Right 4.49 h), **3,454 speaker-change
transitions**.

| gap at speaker change | Malayalam | Tamil |
|---|---|---|
| overlap > 1 s | 53.4% | 41.2% |
| overlap 0.5–1 s | 4.8% | 5.4% |
| overlap 250–500 ms | 2.8% | 3.5% |
| overlap < 250 ms | 3.0% | 4.8% |
| gap 0–250 ms | 4.7% | 6.5% |
| gap 250–750 ms | 8.9% | 11.1% |
| gap 0.75–1.5 s | 7.0% | 11.5% |
| gap > 1.5 s | 15.5% | 15.8% |

Applying the Tamil ruleset — speaker change, gap ≤ 1.5 s, previous and next turn
≥ 1 s, next turn ≥ 3 words — at segment level, before VAD edge-tightening:

| overlap tolerance | Malayalam positives | Tamil positives |
|---|---|---|
| 0 ms | 527 | 3,866 |
| 150 ms | 582 | 4,245 |
| **250 ms** | **606** | **4,474** |
| 500 ms | 684 | 4,894 |

Between-segment holds with the other leg silent: **84** (Tamil: 1,041). As in
Tamil, this undercounts negatives badly — the dominant negative class is
`hold_intra`, pauses *inside* a segment, which only the VAD pass surfaces.

**Malayalam is more overlappy than Tamil** — 53.4% of transitions have >1 s
overlap against 41.2%. That is why the positive ratio (606/4,474 = 13.5%) is
lower than the call ratio (22/116 = 19%). VAD tightening recovered a large share
of these in Tamil and should here too.

### Projected final size

Scaling by the ratios the Tamil pipeline actually produced: **roughly
2,500–4,000 clips**, against Tamil's 18,485. Call it 15–20%.

---

## 5. What this does and does not support

**Does not support a shippable Malayalam model.** `learning_curve.md` measured
Tamil as data-bound at base capacity, with ~40k training rows needed to reach
90%. Malayalam yields ~1,600–2,600 training rows after a split. For reference,
the Tamil 25% run trained on 2,502 rows and reached dev `hold_intra` 81.51%,
AUC 0.886 — so a Malayalam model would land near there and stay: usable, not
shippable, permanently capped.

A test split is the harder constraint. Reserving 5–6 of 22 calls gives ~600–800
test clips against Tamil's 4,168. Set against the measured 0.87-point
reproducibility floor, that is too thin for fine-grained claims.

**Does support the portability claim.** The paper's conclusion asserts the
pipeline ports to any language with a two-channel conversational corpus. Every
structural precondition holds here — free speaker labels, aligned legs,
per-leg transcripts, narrowband telephony, same domain — on a second South-Indian
language, without changing anything. That is the claim this release can settle.

---

## 6. Gotchas

- **`text` is TAB-separated**, while `segments`, `utt2spk` and `wav.scp` are
  space-separated. Splitting `text` on whitespace silently yields zero matches
  and a plausible-looking empty result.
- **The shipped splits leak.** All 22 `ml_IN_` recordings appear in more than
  one of train/dev/eval. Same defect the Tamil R2 audit found (511 of 545).
  Ignore the shipped splits; assign by call.
- **`ml_SHA1P_C` is not a second source.** 133 files from 9 recordings, no
  speaker labels, and already pre-segmented. Same status as Tamil's `SHA1P_C`
  family: unusable without diarization.
- **`ls` hides the family.** 44 files among 21,966. Grep for
  `_Left\.wav$`, not the directory listing.

---

## 7. Status

The oracle exists and is clean. The corpus is a fifth the size of Tamil's,
which rules out a headline model and rules in a portability result.

**Resolved 2026-09-11** — see `dataset_kannada_r1_malayalam_r2.md`. Kannada R1 has
**no** `*_IN_*_{Left,Right}` family (0 of 1,351 files), and SPRING-INX contains no
Telugu at all; its ten languages are Assamese, Bengali, Gujarati, Hindi, Kannada,
Malayalam, Marathi, Odia, Punjabi, Tamil.

So all three South-Indian languages in the corpus are now checked, and **138 calls
(Tamil 116 + Malayalam 22) is the ceiling.** Remaining upside is Indo-Aryan:
Assamese, Gujarati, Odia and Punjabi are unchecked and uncovered by Smart Turn v3.
