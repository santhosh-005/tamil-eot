# Relabelling the pool with `gemini-3.7-flash`

Prompt `ca216b25`, audio only, Vertex AI batch. **16,213 clips labelled for $5.69** (3 returned no usable verdict). Label policy: `llm`.

The pipeline's `hold_intra` class agreed with a human listener 44.4% of
the time — below chance — because "a pause fell inside a transcript
segment" is not a judgement about completeness. This is the pass that
replaces it with one that is.

Scored free against the 197 human-labelled clips that ride along in
the pool: **97.5% agreement**, 97.0% on `hold_intra`, 93.9% precision on `incomplete`.

## What moved

| source | n | stayed | flipped | new `complete` | new `incomplete` |
|---|---|---|---|---|---|
| `change` | 3,901 | 3,612 (93%) | 289 | 3,612 | 289 |
| `hold_intra` | 9,404 | 4,416 (47%) | 4,988 | 4,988 | 4,416 |
| `change_midseg` | 850 | 744 (88%) | 106 | 744 | 106 |
| `hold_inter` | 2,058 | 909 (44%) | 1,149 | 1,149 | 909 |

**Net class balance: 10,493 complete / 5,720 incomplete** (was 4,751 / 11,462).

## Split coverage

| split | labelled | complete | incomplete |
|---|---|---|---|
| train | 10,005 | 6,535 | 3,470 |
| dev | 2,040 | 1,329 | 711 |
| test | 4,168 | 2,629 | 1,539 |

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

The model contradicts the pipeline on **289 of 3,901 `change`**
samples (7.4%). Those are the positives verified at 95.9%
by ear, and grounded in behaviour recorded in the call — the other person
took the floor with a substantive turn. The model is the better labeller
against the human on this class too (98.0% vs 95.9%), so the default policy
`llm` takes it — settled, see `reports/label_policy.md`.
Every row carries both labels plus `dispute`.

Disputed rows overall: **6,532** of 16,213. Training on the undisputed subset
is the high-precision option if neither witness settles it.

## Files

`data/gold/samples_llm.jsonl` — every row from `samples.jsonl`, plus:

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
