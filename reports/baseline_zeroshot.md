# Zero-shot baseline — released Smart Turn on Tamil

**70.3% on real Tamil call audio, against a 63.1% always-say-complete baseline.**

It clears the majority-class baseline by 7.2 points, so it is hearing
*something* — the ROC-AUC below says the ranking carries real signal — but at
the shipped threshold it is not a usable turn detector for Tamil.

> **Runtime tolerance.** This figure is reproducible to about ±0.3% (~12 clips
> of 4,168). ONNX Runtime selects different GEMM kernels by batch size, so float
> summation order changes and clips sitting on the 0.5 threshold flip. Compare a
> trained model against a baseline computed the same way in the same session,
> not against this number to two decimals.

---

`test` split of `samples_llm.jsonl`: **4,168 clips, 2,629 complete / 1,539 incomplete**, from 30 calls.

The released Smart Turn model does not cover Tamil. This is what it does
anyway, before any fine-tuning — the number every later result is measured
against.

Positive class is `complete`. **FP = the agent talks over the user**; FN = dead
air. `FPR/FNR (of all)` is Smart Turn's own benchmark convention (FP/N, FN/N,
summing to the error rate); `FPR (std)` is FP/(FP+TN).

| model | n | accuracy | 95% CI | precision | recall | F1 | FPR (of all) | FNR (of all) | FPR (std) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| `smart-turn-v3.0.onnx` | 4,168 | **65.64%** | 64.2–67.1% | 0.835 | 0.568 | 0.676 | 7.08% | 27.28% | 19.17% | 0.779 |
| `smart-turn-v3.2-cpu.onnx` | 4,168 | **70.30%** | 68.9–71.7% | 0.777 | 0.742 | 0.759 | 13.44% | 16.27% | 36.39% | 0.751 |
| *always say `complete`* | 4,168 | 63.08% | – | – | – | – | – | – | – | 0.500 |

Published v3.2 figures for context: **Marathi 82.43%, Bengali 83.80%**,
against 92.63% overall and 94.26% on English. Those come from a different
benchmark on TTS-generated audio; this one is real narrowband telephone
conversation, so the two are not directly comparable.

## Where `smart-turn-v3.2-cpu.onnx` fails

| | predicted complete | predicted incomplete |
|---|---|---|
| **actually complete** | 1,951 | 678 |
| **actually incomplete** | 560 | 979 |

Of 1,539 clips where the speaker had *not* finished, it says they had on **560** (36.4%).

### By source

| source | n | accuracy | FPR (std) |
|---|---|---|---|
| `change` | 1,024 | 73.54% | 50.00% |
| `hold_intra` | 2,415 | 69.11% | 34.83% |
| `change_midseg` | 224 | 69.20% | 50.00% |
| `hold_inter` | 505 | 69.90% | 38.68% |

### By label agreement

Rows where the pipeline and the labeller agree are the cleanest labels
available. A model that is genuinely reading the audio should do better
there; if it does not, the gap is the model, not the labels.

| | n | accuracy |
|---|---|---|
| both witnesses agree | 2,595 | 69.02% |
| disputed | 1,573 | 72.41% |

### Threshold sweep

The 0.5 default is not sacred. In a voice agent a false 'complete' is much
more expensive than a little added latency, so the operating point should be
chosen, not inherited.

| threshold | accuracy | FPR (std) | FNR (std) |
|---|---|---|---|
| 0.3 | 72.17% | 41.33% | 19.93% |
| 0.4 | 71.47% | 38.53% | 22.67% |
| 0.5 | 70.30% | 36.39% | 25.79% |
| 0.6 | 69.22% | 34.24% | 28.76% |
| 0.7 | 67.49% | 30.93% | 33.43% |
| 0.8 | 65.12% | 27.03% | 39.48% |
| 0.9 | 60.41% | 19.88% | 51.12% |

Best accuracy over the sweep: **74.23%** — an upper bound for what threshold tuning alone can buy, and it needs a
held-out split to choose on, so it is not a free win.
