# SPRING_INX Kannada R1 and Malayalam R2 — audit

Companion to `docs/dataset_malayalam_r1.md` and
`docs/archive/DATASET_AUDIT_SPRING_INX_R1.md`. Same question: does either release
contain the dual-channel family that makes the speaker label free?

**Neither does. Both are unusable for this pipeline.** Not for size reasons — the
audio is plentiful and the content is right — but because the speaker oracle is
absent, and it is the one thing the method cannot substitute for.

All measurements below are from the files.

---

## 1. The oracle is not there

| corpus | files | `_IN_*_{Left,Right}` | channels | speaker labels in metadata |
|---|---|---|---|---|
| Tamil R1 | 21,966 | **232 = 116 calls** | mono, one file per speaker | free, exact |
| Malayalam R1 | 21,966 | **44 = 22 calls** | mono, one file per speaker | free, exact |
| **Kannada R1** | 1,351 | **0** | all mono, no pairing | **none** |
| **Malayalam R2** | 355 | **0** | all mono, no pairing | **none** |

Three independent routes to a speaker label were checked, and all three fail:

1. **No per-speaker files.** Zero filenames match `_Left` / `_Right` in either
   release. The `*_IN_*` family does not exist in either.
2. **No stereo split.** Every file in every family is 1 channel, 16 kHz.
   A stereo file with one speaker per channel would have worked equally well;
   there are none.
3. **No speaker labels in the Kaldi metadata.** `utt2spk` is degenerate — each
   utterance is its own speaker:

   | corpus | segments | unique speakers in `utt2spk` |
   |---|---|---|
   | Kannada R1 | 76,356 (train) | 76,356 |
   | Malayalam R2 | 29,919 (train) | 29,919 |

   That is the Kaldi convention for *no speaker information available*; it makes
   per-speaker normalisation a no-op. It is not a speaker annotation.

### `_C2` is not a second channel

Worth ruling out explicitly, because the name invites the assumption.
Malayalam R2's `ml_SHA1P_C` covers recordings r16–r19 and `ml_SHA1P_C2` covers
r20–r22 — **disjoint**. `C2` is a content type, not channel 2. No pairing exists
between any two families in either release.

---

## 2. What is actually in them

**Kannada R1 — 103.07 h, 1,351 files**

| family | files | hours | share | what |
|---|---|---|---|---|
| `kn_DC023_C` | 963 | 67.63 | 65.6% | conversation, ~213 s/file |
| `kn_DC023_M` | 352 | 25.70 | 24.9% | monologue |
| `ka_CRESC_C` | 25 | 6.71 | 6.5% | conversation, long-form |
| `kn_CRESC_C` | 11 | 3.02 | 2.9% | conversation, long-form |

15 conversation recordings in `kn_DC023_C`, plus 2 in the CRESC families.
Note the inconsistent language prefix: `ka_CRESC_C` against `kn_CRESC_C`.

**Malayalam R2 — 88.49 h, 355 files**

| family | files | hours | share | what |
|---|---|---|---|---|
| `ml_SHA1P_M` | 216 | 43.62 | 49.3% | monologue |
| `ml_SHA1P_C` | 85 | 27.94 | 31.6% | conversation, r16–r19 |
| `ml_SHA1P_C2` | 54 | 16.93 | 19.1% | conversation, r20–r22 |

Transcripts exist for both, with timestamped segments. Malayalam R2's are
noticeably code-switched (`hello`, `school` inline in Malayalam script), which is
its own point of interest but not one this pipeline uses.

Segment coverage of wall clock is sane in both — median 96.2% (Kannada) and
93.6% (Malayalam R2). Unlike Tamil R1's dual-channel family there is no 107%
over-coverage, because there is only one leg to sum.

---

## 3. Malayalam R2 splits are fully leaked

| corpus | recordings referenced | in more than one split |
|---|---|---|
| Kannada R1 | 1,351 | **2 (0.1%)** — effectively clean |
| Malayalam R2 | 355 | **355 (100%)** |

Every Malayalam R2 recording appears in train, dev *and* eval. `ml_SHA1P_M_r022_s098`
is in all three. This is the same defect the paper notes for Tamil R2 (511 of 545
recordings on both sides), taken to its limit.

It is not a blocker on its own — this pipeline re-splits by call anyway — but it
confirms that shipped SPRING-INX splits cannot be trusted, and that R2 releases
are the wrong place to look.

---

## 4. Why no oracle means no dataset

The Tamil positive class is grounded in behaviour recorded in the call: the other
speaker heard the turn end and took the floor. That is what made it 95.9% right
against a blind human pass, with no diarisation and therefore no speaker-error
rate propagating into labels.

Without speaker attribution, the `change` class cannot be constructed at all.
What remains derivable is `hold_intra` — pauses inside a transcript segment —
and that is precisely the class that scored **44.4%, below chance**, and forced
the entire audio-LLM relabelling effort. These corpora would hand us only the
broken half.

Recovering the positive class would require running diarisation over the
conversation families and accepting its error rate inside the labels. That
contradicts the central methodological claim of the paper, and it would have to
be validated by a fresh human listening pass per language before any number from
it could be published.

**Verdict: not usable for EOT at any effort level short of diarisation.** The
audio is fine and the domain is right; the structure is wrong.

---

## 5. What this resolves

`docs/dataset_malayalam_r1.md` §7 left open whether Telugu and Kannada R1 carry
the `*_IN_*_{Left,Right}` family.

- **Kannada R1: answered — no.** 0 matching files out of 1,351.
- **Malayalam R2: answered — no**, and R2 releases are the wrong branch regardless.
- **Telugu: the question was malformed.** SPRING-INX does not contain Telugu.
  The corpus covers ten languages — Assamese, Bengali, Gujarati, Hindi, Kannada,
  Malayalam, Marathi, Odia, Punjabi, Tamil (arXiv:2310.14654).

So the family is not a SPRING-INX-wide convention. It is present in Tamil R1 and
Malayalam R1 and absent from both releases audited here, which means each
language has to be checked individually and most will fail.

### South-Indian is exhausted

SPRING-INX has exactly three South-Indian languages, and all three are now checked:

| language | `_IN_` family | calls |
|---|---|---|
| Tamil R1 | yes | 116 |
| Malayalam R1 | yes | 22 |
| Kannada R1 | **no** | — |

**138 calls is the ceiling for a South-Indian set from this corpus.** There is no
fourth South-Indian language to find.

### The remaining upside is Indo-Aryan, not South-Indian

Smart Turn v3 covers three of SPRING-INX's ten: Bengali, Hindi, Marathi. The
seven it does not cover are Assamese, Gujarati, Kannada, Malayalam, Odia, Punjabi
and Tamil. Of those, Tamil is done, Malayalam is available at 22 calls, and
Kannada has no oracle — leaving **Assamese, Gujarati, Odia and Punjabi
unchecked.** Four releases, one grep each (§6).

That reframes any follow-up from "South-Indian" to "the Indic languages no open
detector covers", which is both a larger claim and one the corpus can actually
support if even two of the four carry the family.

## 6. How to check a new release in one command

```bash
ls <CORPUS>/Audio | grep -cE '_(Left|Right)\.wav$'
```

Non-zero means the oracle is present and the rest of the audit is worth running.
Zero means stop. Both corpora here return zero.
