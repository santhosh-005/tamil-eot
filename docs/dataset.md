# The dataset

**18,485 labelled turn boundaries from 116 real Tamil telephone conversations.**

Published: [`santhosh-005/tamil-eot`](https://huggingface.co/datasets/santhosh-005/tamil-eot)

---

## The label oracle

Everything rests on one property of `SPRING_INX_Tamil_R1`: the
`ta_IN_*_{Left,Right}` family is **116 real Tamil telephone conversations
shipped as one audio file per speaker**, each leg separately transcribed by a
human.

**Which file the audio came from *is* the speaker label.** No diarization, no
speaker-error rate, no LLM guessing at who spoke — the ground truth is
structural.

Verified in `pipeline/01_manifest.py`: 116/116 pairs complete, **0 ms** duration
mismatch between legs.

Licence **CC BY 4.0**, which is what makes the derived set redistributable.

## Two decisions worth knowing about

### Boundaries come from the VAD, not the transcript timestamps

The shipped transcript segments are **padded**: summed across both legs they
cover ~107% of the call wall clock, which is only possible if they carry lead-in
and trailing silence. Using their edges as turn boundaries inflates apparent
overlap and misplaces every gap.

The transcript is used for exactly two things:

1. confirming a VAD span is real speech on *that* leg — a span with no
   transcript over it is crosstalk bleed from the other leg;
2. supplying the text for the backchannel and sentence-final filters.

**Words are apportioned across VAD speech seconds, not elapsed time.** The
median segment is 21% silence and the 10th percentile is 53%, so linear
interpolation hands about a fifth of the words to intervals where nobody spoke —
worst exactly where `hold_intra` lives, because a mid-segment pause *is* that
silence.

### Both classes get identical trailing silence — 200 ms of real audio

If positives ended with more silence than negatives, a model would learn to read
**silence length** rather than speech, score well offline, and collapse against
a production endpointer with different timing.

`pipeline/09_verify.py` exists to catch exactly that. It does **not** report
accuracy — it reports whether the set can be cracked by a model that cannot hear
prosody. Near-chance is the pass condition.

## Classes

| | label | meaning |
|---|---|---|
| `complete` | 1 | speaker finished; an agent should reply now |
| `incomplete` | 0 | speaker paused mid-thought; replying cuts them off |

Every clip is cut from **one leg only** — that is what a deployed detector
receives, because a voice agent sees the inbound user stream, not a mixdown.

## Sources

How each boundary was found, and what the two witnesses said about it:

| source | n | found as | the two witnesses |
|---|---|---|---|
| `hold_intra` | 10,402 | negative | pause wholly inside one transcript segment |
| `change` | 4,660 | positive | other leg took the floor **and** the transcriber closed the segment — agreeing |
| `hold_inter` | 2,343 | negative | pause straddles a segment edge — witnesses disagree |
| `change_midseg` | 1,080 | positive | floor changed mid-segment — an interruption in a turn-change costume |

"Found as" is how the boundary was *detected*. The shipped `label` is the
labeller's verdict, which overrules it on 53% of `hold_intra` —
[experiments/03](../experiments/03-relabelling.md).

`source` is provenance, **not a feature** — it is unavailable at inference. Same
for `dispute` and `harvest`.

## Splits — by call, never by clip

| split | clips | complete | incomplete | calls |
|---|---|---|---|---|
| train | 11,992 | 7,908 | 4,084 | 71 |
| dev | 2,325 | 1,532 | 793 | 15 |
| **test** | **4,168** | 2,629 | 1,539 | **30** |

Both SPRING_INX releases ship utterance-level splits that put the same recording
on both sides — 511 of 545 R2 recordings appear in train and eval alike. A turn
detector trained across such a split **memorises voices instead of learning
prosody**.

**The test split has never moved.** Byte-identical across every repack, which is
why every number in `experiments/` is on one benchmark.

## Schema

`data/gold/samples_llm_all.jsonl` — **the training file**.

| field | |
|---|---|
| `sid` | `{call}_{L/R}_{offset_ms:08d}` — the join key for every cache |
| `label` | what to train on, per the `llm` policy |
| `label_pipeline` | what the channel-derived pipeline said |
| `llm_verdict` | what the labeller said |
| `dispute` | the two disagree |
| `llm_conf`, `llm_reason` | the labeller's confidence and stated reason |
| `llm_ok` | false where no clip exists or the response did not parse — **filter on this** |
| `t`, `clip_start`, `clip_end` | the decision point and the 8 s window |
| `gap`, `prev_dur` | silence after, contiguous speech before |
| `text`, `next_text` | words apportioned to the spurt |
| `source`, `harvest`, `split` | provenance |

> ⚠️ **`samples.jsonl` and `samples_llm_all.jsonl` both have a `label` field and
> they agree on 62.3% of test.** Score against the wrong one and `tiny` reads
> **58% instead of 84%** — plausible enough to publish. Every scorer loads
> `samples_llm_all.jsonl` filtered to `llm_ok`.

## Labelling provenance

State it accurately, because it is what the dataset is: **human-validated,
LLM-assisted labelling.**

- The audio is from a CC BY 4.0 corpus.
- Labels are a binary verdict produced by `gemini-3.7-flash` on that audio —
  not generated content, and not a transcript.
- Agreement with a human listener is **measured and published**: 97.5% on 197
  blind-listened clips, with the per-class breakdown and the noise ceiling.
- Labelling ran on the **paid** tier, where prompts and responses are not used
  for training.

Both the pipeline's own label and the LLM verdict ship on every row, so anyone
can re-derive the set under a different policy.

## Audit trail

| | |
|---|---|
| `docs/archive/DATASET_AUDIT_SPRING_INX_R1.md` | why R1, and the 116-pair verification |
| `docs/archive/DATASET_AUDIT_SPRING_INX_R2.md` | why not R2 — no one-speaker-per-channel split |
| `reports/gold_build.txt` | the full rejection funnel, every boundary accounted for |
| `reports/verify_*.txt` | the shortcut probe, run against each label revision |

## Attribution

Derived from **SPRING_INX Tamil R1**, SPRING Lab, IIT Madras — CC BY 4.0.
Paper: [arXiv:2310.14654](https://arxiv.org/abs/2310.14654).
The derived dataset carries the same licence.
