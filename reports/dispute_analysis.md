# What `dispute` is doing to the numbers

`dispute` marks the 6,532 clips where the gap-based
pipeline and the LLM disagreed. It is not an incidental flag — it splits the
corpus into two populations with different label quality *and* different class
balance, and every aggregate number hides that.

## 1. Label accuracy, from the 197 human labels

| subset | n | % of QA | LLM vs human | pipeline vs human |
|---|---|---|---|---|
| `agree` | 135 | 68.5% | **99.3%** | 99.3% |
| `disputed` | 62 | 31.5% | **93.5%** | 6.5% |
| **all** | 197 | 100.0% | **97.5%** | 70.1% |

The test split is **37.7% disputed**, so the labels it is scored against
are roughly `0.62×99.3 + 0.38×93.5` = **97.1%** accurate.

That is the ceiling on any measured accuracy. **90% sits 7.1 points below it** —
label quality is not what is holding the model back.

## 2. Every subset accuracy needs its own base rate

| subset | n | % complete | majority baseline |
|---|---|---|---|
| `agree` | 2,595 | 44.4% | **55.6%** |
| `disputed` | 1,573 | 93.9% | **93.9%** |
| **all** | 4,168 | 63.1% | 63.1% |

The disputed bucket is near-single-class. Both shipped models score above
their overall accuracy here and still **below** the majority baseline — see
the `vs baseline` column in `reports/finetune_results_*.md`. A subset
accuracy without its own base rate is unreadable, and this one covers 38%
of the test set.

Those clips are acoustically adversarial positives: the pipeline called them
`incomplete` on gap and prosody, and both the human and the LLM overrule
it ~94% of the time. The per-batch `pos_weight` (≈0.59 at 63% positives)
pushes the model toward `incomplete` — wrong for all of them.
