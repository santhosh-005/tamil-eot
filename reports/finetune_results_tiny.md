# Fine-tuned Smart Turn — Tamil

`smart-turn-tamil-tiny/smart-turn-tamil.onnx` on the sealed test split: **4,168 clips from 30 calls**, never trained on.

## Two FPR conventions — check which one a number is in

Smart Turn's published tables report **FP/N and FN/N**, which sum to the
error rate. The training notebook reports **FP/(FP+TN)**, the standard rate.
They are not the same number: this model is **20.9% standard** and
**7.7% in Smart Turn's units.** Compared like for like:

| | accuracy | FPR (FP/N) | FNR (FN/N) |
|---|---|---|---|
| **Tamil (this model)** | **83.35%** | **7.73%** | **8.93%** |
| English | 94.26% | 3.75% | 1.99% |
| overall (all languages) | 92.63% | 4.73% | 2.64% |
| Hindi | 90.11% | 8.57% | 1.32% |
| Bengali | 83.80% | 10.90% | 5.30% |
| Marathi | 82.43% | 15.12% | 2.45% |
| Vietnamese | 79.38% | 8.86% | 11.75% |

Tamil is above **Marathi, Vietnamese** on accuracy and **Hindi, Bengali, Marathi, Vietnamese** on false-positive rate. The published
rows come from a different benchmark on TTS-generated audio and this one is
real narrowband telephone conversation, so read them for scale, not as a
like-for-like comparison.

## Against the zero-shot baseline

| | accuracy | FPR (std) | ROC-AUC |
|---|---|---|---|
| `smart-turn-v3.2` zero-shot | 70.30% | 36.39% | 0.751 |
| **fine-tuned** | **83.35%** | 20.92% | **0.904** |
| delta | **+13.05** | -15.47 | +0.153 |

| | predicted complete | predicted incomplete |
|---|---|---|
| **actually complete** | 2,257 | 372 |
| **actually incomplete** | 322 | 1,217 |

## Choosing the threshold

ROC-AUC is **0.904** — the ranking is good. 0.5 is inherited from the
training loop, not chosen. In a voice agent a false `complete` interrupts the
user; a false `incomplete` costs a little latency. Those are not equal, so
the cut belongs wherever the product wants it.

| threshold | accuracy | FPR (std) | FNR (std) | FPR (FP/N) |
|---|---|---|---|---|
| 0.30 | 83.93% | 27.03% | 9.66% | 9.98% |
| 0.35 | 83.83% | 25.60% | 10.65% | 9.45% |
| 0.40 | 83.59% | 24.24% | 11.83% | 8.95% |
| 0.45 | 83.69% | 22.16% | 12.89% | 8.18% |
| 0.50 | 83.35% | 20.92% | 14.15% | 7.73% |
| 0.55 | 83.21% | 19.43% | 15.25% | 7.17% |
| 0.60 | 82.58% | 18.39% | 16.85% | 6.79% |
| 0.65 | 82.32% | 16.63% | 18.30% | 6.14% |
| 0.70 | 81.74% | 14.81% | 20.27% | 5.47% |
| 0.75 | 80.64% | 13.58% | 22.75% | 5.01% |
| 0.80 | 79.51% | 12.41% | 25.22% | 4.58% |
| 0.85 | 77.81% | 10.66% | 28.95% | 3.93% |
| 0.90 | 73.99% | 8.12% | 36.48% | 3.00% |
| 0.95 | 52.90% | 1.69% | 73.68% | 0.62% |

Best accuracy is **83.93% at threshold 0.30** (FPR 27.0% std).
Pick it on `dev`, never on this split — quoted here only as a ceiling.

To hold interruptions at 10% of pauses, threshold **0.88**: accuracy 75.65%, FPR 9.7% std (3.6% FP/N), FNR 32.9%.
That is the trade a live agent would actually take.

## By source

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `change` | 1,024 | 85.74% | 93.16% | **-7.42** | 47.14% |
| `hold_intra` | 2,415 | 81.90% | 50.31% | **+31.59** | 19.67% |
| `change_midseg` | 224 | 87.05% | 88.39% | **-1.34** | 34.62% |
| `hold_inter` | 505 | 83.76% | 51.88% | **+31.88** | 18.11% |

## By label agreement

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `agree` | 2,595 | 84.08% | 55.61% | **+28.48** | 19.40% |
| `disputed` | 1,573 | 82.14% | 93.90% | **-11.76** | 43.75% |
