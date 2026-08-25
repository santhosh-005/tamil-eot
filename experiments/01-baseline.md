# 01 — Zero-shot baseline

**Question.** The released Smart Turn model does not cover Tamil. What does it
do on Tamil anyway?

**Why first.** The "before" number has to exist before a training step runs, and
it decides whether the project is worth doing. It costs two minutes of CPU.

**Setup.** Test split, 4,168 clips / 30 calls. Threshold 0.5. Positive class is
`complete`, so **FP = the agent talks over the user**.

---

## Result

| | accuracy | ROC-AUC | FPR (std) |
|---|---|---|---|
| always say `complete` | 63.08% | 0.500 | – |
| `smart-turn-v3.0` | 65.64% | **0.779** | 19.17% |
| **`smart-turn-v3.2-cpu`** | **70.30%** | 0.751 | **36.39%** |

For scale, against its published per-language figures — **a different benchmark
on TTS-generated audio, so not directly comparable:**

| | accuracy |
|---|---|
| English | 94.26% |
| Marathi | 82.43% |
| **Tamil, measured here** | **70.30%** |

## Where it fails

| | predicted complete | predicted incomplete |
|---|---|---|
| actually complete | 1,951 | 678 |
| actually incomplete | **560** | 979 |

**36.4% of unfinished turns are called finished.** An agent using it talks over
the user on a third of their mid-sentence pauses.

## The part that made this worth doing

**ROC-AUC 0.751, not 0.5.** Whisper's encoder already hears Tamil prosody — the
ranking carries real signal, only the decision boundary is wrong. That is the
gap fine-tuning closes, and it is why 70 → 86 was plausible before it was
measured.

Threshold tuning alone tops out at **74.23%** over a test-set sweep — a ceiling,
not a setting, and it still needs a held-out split to pick on.

## ⚠️ Reproducibility tolerance

**±0.3%** (~12 clips). ONNX Runtime picks different GEMM kernels by batch size,
float summation order changes, clips sitting on 0.5 flip. Compare a trained
model against a baseline computed *in the same session*, never against this
number to two decimals.

---

`reports/baseline_zeroshot.md` · `python pipeline/13_baseline.py`
