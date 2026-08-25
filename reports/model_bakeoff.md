# Labeller bake-off — 197 human-labelled clips, Vertex AI Batch

Prompt `ca216b25`. Audio only, no transcript. Same clips, same human
labels, every model — so the comparison is paired.

**Winner: `gemini-3.7-flash` — 97.5% agreement, 93.9% precision on `incomplete`, $5.77 for the whole pool.**

Bar was 90%. 6 of 7 clear it. Ties at the top broken on price.

| model | n | agreement | 95% CI | `change` | `hold_intra` | prec `incomplete` | $ run | in/out tok | think |
|---|---|---|---|---|---|---|---|---|---|
| `gemini-3.7-flash` | 197 | **97.5%** | 94–99% | 98.0% | 97.0% | 93.9% (49) | $0.07 | 944/147 | 120 |
| `gemini-3.1-pro-preview` | 197 | **96.4%** | 93–98% | 99.0% | 93.9% | 91.8% (49) | $0.50 | 944/375 | 338 |
| `gemini-3.6-flash` | 197 | **96.4%** | 93–98% | 95.9% | 97.0% | 88.7% (53) | $0.11 | 944/312 | 282 |
| `gemini-3.5-flash` | 197 | **93.9%** | 90–96% | 94.9% | 92.9% | 86.0% (50) | $0.14 | 944/414 | 377 |
| `gemini-3-flash-preview` | 197 | **93.4%** | 89–96% | 95.9% | 90.9% | 85.7% (49) | $0.21 | 944/512 | 451 |
| `gemini-2.5-flash` | 196 | **91.8%** | 87–95% | 92.9% | 90.8% | 83.3% (48) | $0.16 | 944/497 | 464 |
| `gemini-3.5-flash-lite` | 197 | **66.5%** | 60–73% | 64.3% | 68.7% | 41.2% (102) | $0.01 | 944/29 | 0 |
| *our pipeline* | 197 | 70.1% | – | 95.9% | 44.4% | 44.4% | – | – | – |

Cost columns are carried from the 2026-08-20 run — the GCP project holding
the job output is gone, and a label correction does not change what was billed.
Agreement columns are recomputed from the local verdict caches.

## Is the gap real? Paired McNemar against `gemini-3.7-flash`

Discordant pairs only — clips where exactly one of the two was right.

| vs | winner-only right | other-only right | p |
|---|---|---|---|
| `gemini-3.1-pro-preview` | 6 | 4 | 0.754 — not significant |
| `gemini-3.6-flash` | 5 | 3 | 0.727 — not significant |
| `gemini-3.5-flash` | 11 | 4 | 0.118 — not significant |
| `gemini-3-flash-preview` | 11 | 3 | 0.057 — not significant |
| `gemini-2.5-flash` | 14 | 3 | 0.013 — **significant** |
| `gemini-3.5-flash-lite` | 62 | 1 | 0.000 — **significant** |

## Clips where the models agree against the human label

None. The reference is one human's single listen to an isolated 8 s
clip with no future audio, so a clip every model calls the other way is
evidence against the label. Three were, and `reports/qa/sheet.tsv` now
holds the corrected verdicts.

## Cost to label the full pool at batch rates

| model | `hold_intra` 9,406 | everything 16,216 |
|---|---|---|
| `gemini-3.7-flash` | $3.35 | $5.77 |
| `gemini-3.1-pro-preview` | $23.88 | $41.17 |
| `gemini-3.6-flash` | $5.28 | $9.10 |
| `gemini-3.5-flash` | $6.86 | $11.82 |
| `gemini-3-flash-preview` | $9.91 | $17.09 |
| `gemini-2.5-flash` | $7.83 | $13.50 |
| `gemini-3.5-flash-lite` | $0.69 | $1.18 |

Errors / unparsed per model: `gemini-3.7-flash` 0, `gemini-3.1-pro-preview` 0, `gemini-3.6-flash` 0, `gemini-3.5-flash` 0, `gemini-3-flash-preview` 0, `gemini-2.5-flash` 1, `gemini-3.5-flash-lite` 0

Regenerate with `python pipeline/12_bakeoff.py`.
