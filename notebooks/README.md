# Training

Both shipped models come from `train_smart_turn_tamil.ipynb`, on a Colab T4.
One line switches variant:

```python
ENCODER = 'openai/whisper-base'      # 'openai/whisper-tiny' for the small one
EPOCHS, LR = 6, 5e-5
SEED = 0
```

Everything else — data loading, the attention-pooling head, the training loop,
the test pass, the ONNX export — is shared.

| | |
|---|---|
| `train_smart_turn_tamil.ipynb` | the notebook to run. Outputs cleared |
| `runs/base-86.23-2026-08-22.ipynb` | the **executed** run that produced the shipped `base` model, outputs intact |

## The run record

`runs/base-86.23-2026-08-22.ipynb` is kept because a cleared notebook proves
nothing about which code produced the weights. Its outputs carry:

```
20.3M params on cuda                          <- whisper-base
best dev accuracy 86.80%
TEST  accuracy 86.23%   FPR 24.24%   AUC 0.922
onnx vs torch max abs diff: 1.43e-06
-rw-r--r-- 1 root root 81320895 /content/smart-turn-tamil.onnx
```

86.23% / 0.922 is the `base` fp32 row in the results tables, and 81,320,895
bytes is the size of the shipped `smart-turn-tamil.onnx`.

**It is an earlier revision of the same notebook, not a re-run of the current
one.** It hardcodes `'openai/whisper-base'` in two places rather than reading
`ENCODER`, and it predates the learning-rate sweep cell. The training loop,
optimiser, schedule, seeding and export are identical. Read it as the record of
what ran, and `train_smart_turn_tamil.ipynb` as the code to run.

**No equivalent record survives for `tiny`.** That session hit its Colab usage
limit shortly after training; `/content` was lost and only what had already been
copied to Drive survived — the weights, not the notebook. The `tiny` run used
the same notebook with `ENCODER` set to `openai/whisper-tiny`.

## Reproducing is not the same as reproducing exactly

Three `base` runs at identical config and `SEED = 0` scored **86.23 / 85.63 /
85.36** on test. SDPA's backward pass uses atomics, so the seed does not pin the
result. Re-running this notebook will land somewhere in that band, not on
86.23 — treat any difference under ~1 point as noise.
