# Relabelling the pool with `gemini-3.7-flash`

Prompt `ca216b25`, audio only, Vertex AI batch. **2,272 clips labelled for $0.78** (0 returned no usable verdict). Label policy: `llm`.

The pipeline's `hold_intra` class agreed with a human listener 44.4% of
the time — below chance — because "a pause fell inside a transcript
segment" is not a judgement about completeness. This is the pass that
replaces it with one that is.

## What moved

| source | n | stayed | flipped | new `complete` | new `incomplete` |
|---|---|---|---|---|---|
| `change` | 759 | 628 (83%) | 131 | 628 | 131 |
| `hold_intra` | 998 | 436 (44%) | 562 | 562 | 436 |
| `change_midseg` | 230 | 185 (80%) | 45 | 185 | 45 |
| `hold_inter` | 285 | 84 (29%) | 201 | 201 | 84 |

**Net class balance: 1,576 complete / 696 incomplete** (was 989 / 1,283).

## Split coverage

| split | labelled | complete | incomplete |
|---|---|---|---|
| train | 1,987 | 1,373 | 614 |
| dev | 285 | 203 | 82 |

## Does it still survive being attacked?

`pipeline/09_verify.py` fits a deliberately dumb classifier on ten crude
global features — energy, duration, spectral tilt, voiced fraction — that
carry no prosody. Near-chance is the pass condition.

| | samples | AUC | acc vs majority | `prev_dur` d | `voiced_frac` d |
|---|---|---|---|---|---|
| pipeline labels | 13,307 | 0.601 | 0.701 / 0.702 | -0.33 | -0.46 |
| relabelled, core | 13,305 | 0.622 | 0.642 / 0.631 | -0.09 | -0.17 |
| relabelled, all | 16,213 | 0.622 | 0.645 / 0.631 | -0.11 | -0.20 |

**The `prev_dur` confound is gone.** It was the one structural defect the
build knew about: `hold_intra` required a pause *inside* a transcript
segment, so the negative class was drawn from longer utterances by
construction (d = -0.33), and it leaked into the clip as voiced
fraction (d = -0.46, the largest audio effect measured). Labelling by
listening severs that link: d = -0.09 and -0.17. The gate no longer
decides the label, so the gate's bias no longer rides along with it.

AUC moved 0.601 → 0.622, which is up, not down — worth saying plainly.
Both are well inside the near-chance band, and the class balance moved at
the same time (0.702 → 0.631 majority), so AUC is the comparable
number and accuracy is not. The probe beats the majority class by 1.1
points where before it lost to it by 0.1.

## The one judgement call in this file

The model contradicts the pipeline on **131 of 759 `change`**
samples (17.3%). Those are the positives verified at 95.9%
by ear, and grounded in behaviour recorded in the call — the other person
took the floor with a substantive turn. The model is the better labeller
against the human on this class too (98.0% vs 95.9%), so the default policy
`llm` takes it — settled, see `reports/label_policy.md`.
Every row carries both labels plus `dispute`.

Disputed rows overall: **939** of 2,272. Training on the undisputed subset
is the high-precision option if neither witness settles it.

## Files

`data/gold/samples_relaxed_llm.jsonl` — every row from `samples.jsonl`, plus:

| field | meaning |
|---|---|
| `label` | the label to train on, per the policy above |
| `label_pipeline` | what the channel-derived pipeline said |
| `llm_verdict` | what the model said |
| `dispute` | the two disagree |
| `llm_conf`, `llm_reason` | the model's confidence and its stated reason |
| `llm_ok` | false where no clip exists, or the response did not parse |

Re-check it at any time with:

```bash
python pipeline/09_verify.py --samples samples_llm.jsonl --sources all --save
```
