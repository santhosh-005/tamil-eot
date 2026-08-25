# Fine-tuned Smart Turn — Tamil

`base-auc.onnx` on the sealed test split: **4,168 clips from 30 calls**, never trained on.

## Two FPR conventions — check which one a number is in

Smart Turn's published tables report **FP/N and FN/N**, which sum to the
error rate. The training notebook reports **FP/(FP+TN)**, the standard rate.
They are not the same number: this model is **27.6% standard** and
**10.2% in Smart Turn's units.** Compared like for like:

| | accuracy | FPR (FP/N) | FNR (FN/N) |
|---|---|---|---|
| **Tamil (this model)** | **85.63%** | **10.20%** | **4.17%** |
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
| **fine-tuned** | **85.63%** | 27.62% | **0.917** |
| delta | **+15.33** | -8.77 | +0.166 |

| | predicted complete | predicted incomplete |
|---|---|---|
| **actually complete** | 2,455 | 174 |
| **actually incomplete** | 425 | 1,114 |

## Choosing the threshold

ROC-AUC is **0.917** — the ranking is good. 0.5 is inherited from the
training loop, not chosen. In a voice agent a false `complete` interrupts the
user; a false `incomplete` costs a little latency. Those are not equal, so
the cut belongs wherever the product wants it.

| threshold | accuracy | FPR (std) | FNR (std) | FPR (FP/N) |
|---|---|---|---|---|
| 0.30 | 85.24% | 31.64% | 4.87% | 11.68% |
| 0.35 | 85.34% | 30.80% | 5.21% | 11.37% |
| 0.40 | 85.58% | 29.50% | 5.59% | 10.89% |
| 0.45 | 85.72% | 28.33% | 6.05% | 10.46% |
| 0.50 | 85.63% | 27.62% | 6.62% | 10.20% |
| 0.55 | 85.82% | 26.38% | 7.04% | 9.74% |
| 0.60 | 85.72% | 25.73% | 7.57% | 9.50% |
| 0.65 | 85.84% | 24.43% | 8.14% | 9.02% |
| 0.70 | 85.41% | 23.91% | 9.13% | 8.83% |
| 0.75 | 85.56% | 22.35% | 9.81% | 8.25% |
| 0.80 | 85.10% | 20.92% | 11.37% | 7.73% |
| 0.85 | 84.84% | 18.97% | 12.93% | 7.01% |
| 0.90 | 83.59% | 16.50% | 16.36% | 6.09% |
| 0.95 | 79.29% | 10.40% | 26.74% | 3.84% |

Best accuracy is **85.84% at threshold 0.65** (FPR 24.4% std).
Pick it on `dev`, never on this split — quoted here only as a ceiling.

To hold interruptions at 10% of pauses, threshold **0.96**: accuracy 75.19%, FPR 7.3% std (2.7% FP/N), FNR 35.0%.
That is the trade a live agent would actually take.

## By source

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `change` | 1,024 | 91.70% | 93.16% | **-1.46** | 68.57% |
| `hold_intra` | 2,415 | 82.53% | 50.31% | **+32.22** | 26.33% |
| `change_midseg` | 224 | 88.84% | 88.39% | **+0.45** | 61.54% |
| `hold_inter` | 505 | 86.73% | 51.88% | **+34.85** | 18.52% |

## By label agreement

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `agree` | 2,595 | 84.32% | 55.61% | **+28.71** | 25.02% |
| `disputed` | 1,573 | 87.79% | 93.90% | **-6.10** | 66.67% |
