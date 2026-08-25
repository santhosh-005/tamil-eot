# int8 quantisation

Upstream's `smart-turn-v3.2-cpu.onnx` is int8 (78 `QuantizeLinear` nodes,
uint8/int8 initialisers) and ours was fp32 — 8.6 MB against 32 MB for the
same architecture. This closes that gap and measures what it costs.

Activations calibrated on **256 `dev` clips**, never test. Latency is
single-threaded, batch 1, on an i5-12450H — the shape a live agent runs.

## tiny

| build | size | p50 | p95 | accuracy | AUC | FP/N | vs fp32 |
|---|---|---|---|---|---|---|---|
| fp32 | 32 MB | 133 ms | 149 ms | 83.35% | 0.904 | 7.73% | — |
| int8 dynamic | 8.7 MB | 83 ms | 94 ms | 83.71% | 0.905 | 7.94% | +0.36 acc, 97.6% same |
| int8 static | 8.4 MB | 73 ms | 81 ms | 79.32% | 0.899 | 4.94% | -4.03 acc, 90.0% same |


## base

| build | size | p50 | p95 | accuracy | AUC | FP/N | vs fp32 |
|---|---|---|---|---|---|---|---|
| fp32 | 81 MB | 232 ms | 238 ms | 86.23% | 0.922 | 8.95% | — |
| int8 dynamic | 21 MB | 143 ms | 148 ms | 86.13% | 0.921 | 9.17% | -0.10 acc, 98.6% same |
| int8 static | 21 MB | 125 ms | 128 ms | 73.46% | 0.792 | 13.70% | -12.76 acc, 76.2% same |

