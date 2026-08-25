# 08 — int8 quantisation ✅ dynamic, ❌ static

**Question.** Upstream ships `smart-turn-v3.2-cpu.onnx` at 8.6 MB. Ours was
32 MB fp32, same architecture. Why?

**Answer.** 78 `QuantizeLinear` nodes and uint8/int8 initialisers. **Upstream
ships int8 and we were shipping fp32** — "drop-in replacement" was true of the
graph signature and false of the artefact.

**Setup.** Frozen test set. Activations calibrated on **256 `dev` clips**, never
test. p50 single-thread, batch 1, idle i5-12450H.

---

## Result

| | build | size | p50 | accuracy | AUC | FP/N | agrees w/ fp32 |
|---|---|---|---|---|---|---|---|
| tiny | fp32 | 32 MB | 133 ms | 83.35% | 0.904 | 7.73% | — |
| tiny | **int8 dynamic** | **8.7 MB** | **83 ms** | **83.71%** | **0.905** | 7.94% | 97.6% |
| tiny | int8 static | 8.4 MB | 73 ms | 79.32% | 0.899 | 4.94% | 90.0% |
| base | fp32 | 81 MB | 232 ms | 86.23% | 0.922 | 8.95% | — |
| base | **int8 dynamic** | **21 MB** | **143 ms** | **86.13%** | **0.921** | 9.17% | 98.6% |
| base | int8 static | 21 MB | 125 ms | **73.46%** | **0.792** | 13.70% | 76.2% |

## ✅ Dynamic is free at both sizes

~3.8× smaller, ~38% faster, −0.10 to +0.36 accuracy — **both inside the noise
band, so: lossless.**

Our tiny build is **8.65 MB against upstream's 8.68 MB**. The same choice they
made, now measured rather than copied.

## ❌ Static is a trap, and the damage scales with capacity

| | tiny | base |
|---|---|---|
| accuracy cost | −4.03 | **−12.76** |
| AUC | 0.904 → 0.899 | 0.922 → **0.792** |
| agreement with fp32 | 90.0% | 76.2% |

At tiny, static only shifts the **decision boundary** — AUC barely moves, the
ranking survives, the threshold is wrong. At base it destroys the **ranking**:
AUC 0.922 → 0.792 is most of the way back to the 0.751 zero-shot model this
entire project exists to beat.

20.3M parameters have a wider activation range than 256 dev clips can calibrate.

## ⚠️ The reason to never read one metric

**tiny int8 static has the best FP/N in the whole table — 4.94% — and is the
second-worst model.** Static was also the one *expected* to win: weights and
activations both quantised, fastest of the three.

Reading one metric would have shipped it.

## Latency, with its conditions attached

Idle i5-12450H, batch 1, inference only — add ~12 ms for the numpy mel.

| | 1 thread | all 12 cores |
|---|---|---|
| tiny fp32 | 133 ms | 31 ms |
| **tiny int8 dynamic** | **83 ms** | 27 ms |
| base fp32 | 232 ms | 72 ms |
| **base int8 dynamic** | **143 ms** | 52 ms |

**1 thread is the number that matters** — Pipecat defaults to `cpu_count=1`, and
a server carrying concurrent calls cannot give each one 12 cores.

> ⚠️ **Three wrong latency figures were published before this table**, each
> failing differently. (1) "~25 MB / ~25 ms" for base — a guess, never measured.
> (2) "tiny ~95–150 ms, base ~240–370 ms" — measured while another job saturated
> all 12 cores; 3–5× inflated, and the conclusion drawn from it (*base is too
> slow for realtime*) was **wrong**. (3) "31 ms, batch 1, single thread" — the
> figure was right, the label was not.
>
> **A latency number without thread count and machine load attached is not a
> measurement.** `bench.py` now warns above 30% load.

So tiny vs base is **accuracy vs size, not latency** — +2.42 points for 2.4× the
file and 1.7× the latency. Both realtime.

## What ships

**int8 dynamic, both variants.** fp32 stays as the reference graph. Static
builds are deleted, not kept — `pipeline/19_quantize.py` regenerates them
deterministically if anyone wants to re-check.

---

`reports/quantisation.md` · `python pipeline/19_quantize.py`
