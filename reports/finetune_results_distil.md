# Fine-tuned Smart Turn — Tamil

`../data/colab/smart-turn-tamil-distil.onnx` on the sealed test split: **4,168 clips from 30 calls**, never trained on.

## Two FPR conventions — check which one a number is in

Smart Turn's published tables report **FP/N and FN/N**, which sum to the
error rate. The training notebook reports **FP/(FP+TN)**, the standard rate.
They are not the same number: this model is **24.8% standard** and
**9.1% in Smart Turn's units.** Compared like for like:

| | accuracy | FPR (FP/N) | FNR (FN/N) |
|---|---|---|---|
| **Tamil (this model)** | **83.76%** | **9.14%** | **7.10%** |
| English | 94.26% | 3.75% | 1.99% |
| overall (all languages) | 92.63% | 4.73% | 2.64% |
| Hindi | 90.11% | 8.57% | 1.32% |
| Bengali | 83.80% | 10.90% | 5.30% |
| Marathi | 82.43% | 15.12% | 2.45% |
| Vietnamese | 79.38% | 8.86% | 11.75% |

Tamil is above **Marathi, Vietnamese** on accuracy and **Bengali, Marathi** on false-positive rate. The published
rows come from a different benchmark on TTS-generated audio and this one is
real narrowband telephone conversation, so read them for scale, not as a
like-for-like comparison.

## Against the zero-shot baseline

| | accuracy | FPR (std) | ROC-AUC |
|---|---|---|---|
| `smart-turn-v3.2` zero-shot | 70.30% | 36.39% | 0.751 |
| **fine-tuned** | **83.76%** | 24.76% | **0.906** |
| delta | **+13.46** | -11.63 | +0.155 |

| | predicted complete | predicted incomplete |
|---|---|---|
| **actually complete** | 2,333 | 296 |
| **actually incomplete** | 381 | 1,158 |

## Choosing the threshold

ROC-AUC is **0.906** — the ranking is good. 0.5 is inherited from the
training loop, not chosen. In a voice agent a false `complete` interrupts the
user; a false `incomplete` costs a little latency. Those are not equal, so
the cut belongs wherever the product wants it.

| threshold | accuracy | FPR (std) | FNR (std) | FPR (FP/N) |
|---|---|---|---|---|
| 0.30 | 84.12% | 28.20% | 8.67% | 10.41% |
| 0.35 | 84.14% | 27.36% | 9.13% | 10.10% |
| 0.40 | 84.05% | 26.25% | 9.93% | 9.69% |
| 0.45 | 83.95% | 25.28% | 10.65% | 9.33% |
| 0.50 | 83.76% | 24.76% | 11.26% | 9.14% |
| 0.55 | 83.64% | 23.52% | 12.17% | 8.69% |
| 0.60 | 83.35% | 22.87% | 13.01% | 8.45% |
| 0.65 | 83.18% | 21.90% | 13.85% | 8.09% |
| 0.70 | 82.87% | 20.99% | 14.87% | 7.75% |
| 0.75 | 82.51% | 19.62% | 16.24% | 7.25% |
| 0.80 | 81.91% | 18.26% | 17.99% | 6.74% |
| 0.85 | 80.88% | 15.92% | 21.00% | 5.88% |
| 0.90 | 79.39% | 13.32% | 24.88% | 4.92% |
| 0.95 | 72.24% | 5.13% | 41.00% | 1.90% |

Best accuracy is **84.14% at threshold 0.35** (FPR 27.4% std).
Pick it on `dev`, never on this split — quoted here only as a ceiling.

To hold interruptions at 10% of pauses, threshold **0.93**: accuracy 77.52%, FPR 9.9% std (3.6% FP/N), FNR 29.9%.
That is the trade a live agent would actually take.

## By source

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `change` | 1,024 | 88.96% | 93.16% | **-4.20** | 52.86% |
| `hold_intra` | 2,415 | 81.45% | 50.31% | **+31.14** | 23.33% |
| `change_midseg` | 224 | 85.71% | 88.39% | **-2.68** | 57.69% |
| `hold_inter` | 505 | 83.37% | 51.88% | **+31.49** | 20.16% |

## By label agreement

| | n | accuracy | majority baseline | vs baseline | FPR (std) |
|---|---|---|---|---|---|
| `agree` | 2,595 | 83.74% | 55.61% | **+28.13** | 22.80% |
| `disputed` | 1,573 | 83.79% | 93.90% | **-10.11** | 54.17% |
