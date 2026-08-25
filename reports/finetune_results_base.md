# Fine-tuned Smart Turn — Tamil

`smart-turn-tamil-base/smart-turn-tamil.onnx` on the sealed test split: **4,168 clips from 30 calls**, never trained on.

## Two FPR conventions — check which one a number is in

Smart Turn's published tables report **FP/N and FN/N**, which sum to the
error rate. The training notebook reports **FP/(FP+TN)**, the standard rate.
They are not the same number: this model is **24.2% standard** and
**8.9% in Smart Turn's units.** Compared like for like:

| | accuracy | FPR (FP/N) | FNR (FN/N) |
|---|---|---|---|
| **Tamil (this model)** | **86.23%** | **8.95%** | **4.82%** |
| English | 94.26% | 3.75% | 1.99% |
| overall (all languages) | 92.63% | 4.73% | 2.64% |
| Hindi | 90.11% | 8.57% | 1.32% |
| Bengali | 83.80% | 10.90% | 5.30% |
| Marathi | 82.43% | 15.12% | 2.45% |
| Vietnamese | 79.38% | 8.86% | 11.75% |

Tamil is above **Bengali, Marathi, Vietnamese** on accuracy and **Bengali, Marathi** on false-positive rate. The published
rows come from a different benchmark on TTS-generated audio and this one is
real narrowband telephone conversation, so read them for scale, not as a
like-for-like comparison.

## Against the zero-shot baseline

| | accuracy | FPR (std) | ROC-AUC |
|---|---|---|---|
| `smart-turn-v3.2` zero-shot | 70.30% | 36.39% | 0.751 |
| **fine-tuned** | **86.23%** | 24.24% | **0.922** |
| delta | **+15.93** | -12.15 | +0.171 |

| | predicted complete | predicted incomplete |
|---|---|---|
| **actually complete** | 2,428 | 201 |
| **actually incomplete** | 373 | 1,166 |

## Choosing the threshold

ROC-AUC is **0.922** — the ranking is good. 0.5 is inherited from the
training loop, not chosen. In a voice agent a false `complete` interrupts the
user; a false `incomplete` costs a little latency. Those are not equal, so
the cut belongs wherever the product wants it.

| threshold | accuracy | FPR (std) | FNR (std) | FPR (FP/N) |
|---|---|---|---|---|
| 0.30 | 85.53% | 30.08% | 5.33% | 11.11% |
| 0.35 | 85.70% | 28.33% | 6.09% | 10.46% |
| 0.40 | 86.06% | 26.58% | 6.54% | 9.81% |
| 0.45 | 86.25% | 25.28% | 7.00% | 9.33% |
| 0.50 | 86.23% | 24.24% | 7.65% | 8.95% |
| 0.55 | 86.35% | 22.87% | 8.25% | 8.45% |
| 0.60 | 86.25% | 21.64% | 9.13% | 7.99% |
| 0.65 | 86.37% | 20.47% | 9.62% | 7.56% |
| 0.70 | 86.13% | 19.36% | 10.65% | 7.15% |
| 0.75 | 85.65% | 18.45% | 11.94% | 6.81% |
| 0.80 | 85.27% | 16.50% | 13.69% | 6.09% |
| 0.85 | 84.72% | 15.14% | 15.37% | 5.59% |
| 0.90 | 83.25% | 13.19% | 18.83% | 4.87% |
| 0.95 | 79.25% | 8.19% | 28.11% | 3.02% |

Best accuracy is **86.37% at threshold 0.65** (FPR 20.5% std).
Pick it on `dev`, never on this split — quoted here only as a ceiling.

To hold interruptions at 10% of pauses, threshold **0.94**: accuracy 80.97%, FPR 9.7% std (3.6% FP/N), FNR 24.5%.
That is the trade a live agent would actually take.

## By source

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `change` | 1,024 | 91.11% | 93.16% | **-2.05** | 55.71% |
| `hold_intra` | 2,415 | 83.73% | 50.31% | **+33.42** | 23.00% |
| `change_midseg` | 224 | 88.84% | 88.39% | **+0.45** | 57.69% |
| `hold_inter` | 505 | 87.13% | 51.88% | **+35.25** | 17.70% |

## By label agreement

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `agree` | 2,595 | 85.32% | 55.61% | **+29.71** | 22.11% |
| `disputed` | 1,573 | 87.73% | 93.90% | **-6.17** | 56.25% |
