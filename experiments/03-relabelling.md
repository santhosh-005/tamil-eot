# 03 — Relabelling with an audio LLM ✅

**Question.** Can a model listening to the clip beat a pipeline reasoning about
transcript structure?

**Setup.** Seven Gemini models, same 197 human-labelled clips, same prompt
(`ca216b25`), audio only — no transcript, no pipeline label. Paired, so the
comparison is like-for-like. Vertex AI Batch.

---

## Bake-off

| model | agreement | on `hold_intra` | $ full pool |
|---|---|---|---|
| **`gemini-3.7-flash`** | **97.5%** | **97.0%** | **$5.77** |
| `gemini-3.1-pro-preview` | 96.4% | 93.9% | $41.17 |
| `gemini-3.6-flash` | 96.4% | 97.0% | $9.10 |
| `gemini-3.5-flash` | 93.9% | 92.9% | $11.82 |
| `gemini-3-flash-preview` | 93.4% | 90.9% | $17.09 |
| `gemini-2.5-flash` | 91.8% | 90.8% | $13.50 |
| `gemini-3.5-flash-lite` | 66.5% | 68.7% | $1.18 |
| *the pipeline* | 70.1% | 44.4% | – |

Bar was 90%; 6 of 7 clear it. **Ties broken on price, not accuracy** — McNemar
says the top three are statistically indistinguishable (p = 0.75, 0.73). Only
`gemini-2.5-flash` (p = 0.013) and `-lite` (p < 0.001) are genuinely behind.

**97.0% on the exact class the pipeline got wrong.**

## What relabelling moved

16,213 clips, **$5.69**.

| source | n | flipped |
|---|---|---|
| `change` | 3,901 | 289 (7%) |
| `hold_intra` | 9,404 | **4,988 (53%)** |
| `change_midseg` | 850 | 106 (12%) |
| `hold_inter` | 2,058 | 1,149 (56%) |

Class balance **4,751/11,462 → 10,493/5,720**.

**53% of the negative class flipped, against 56% in the independent human
pass** — two measurements of the same error, three points apart.

## ✅ The confound is gone

`09_verify.py` fits a deliberately dumb classifier on ten crude global features
that carry no prosody. Near-chance is the pass condition.

| | AUC | `prev_dur` d | `voiced_frac` d |
|---|---|---|---|
| pipeline labels | 0.601 | **−0.33** | **−0.46** |
| relabelled | 0.622 | **−0.09** | **−0.17** |

The gate no longer decides the label, so its bias no longer rides along.

**AUC went up, 0.601 → 0.622 — worth saying plainly.** Both sit inside the
near-chance band, and the class balance moved at the same time (majority
0.702 → 0.631), so AUC is the comparable number and accuracy is not.

## The noise ceiling

| subset | share of test | label accuracy |
|---|---|---|
| both witnesses agree | 62.3% | 99.3% |
| disputed | 37.7% | 93.5% |
| **weighted** | | **97.1%** |

**Above 97.1% you are fitting labeller error.** 11 points of headroom remain, so
label quality is not what limits this model.

## ✅ Label policy — `llm`, settled

`llm` vs `core-safe` (keep the pipeline's label on `change`) differ on **420 of
18,485 rows** — 2.27%, all one direction.

| | |
|---|---|
| human verdict on the 6 QA clips it decides | **4–2 for the LLM** |
| rows in the sealed test split | **70** |
| Δ test accuracy if flipped | tiny −0.10, base +0.29 |

Opposite signs, both far inside the 0.87-point seed spread. Nothing to win —
and `core-safe` would not reweight training, it would **re-label the
benchmark**. The tie breaks on benchmark integrity.

Reversible: every row keeps `label_pipeline`, `llm_verdict` and `dispute`.

---

`reports/model_bakeoff.md`, `reports/relabelling.md`, `reports/dispute_analysis.md`,
`reports/label_policy.md` ·
`python pipeline/11_label.py` → `12_bakeoff.py` → `17_dispute_analysis.py` → `20_label_policy.py`
