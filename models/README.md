# models/

Nothing here is committed. Two kinds of file land in this directory.

## Third-party, fetched and hash-checked

```bash
python pipeline/00_fetch_models.py
```

| file | what | licence |
|---|---|---|
| `silero_vad.onnx` | Silero VAD v5, pinned to one commit | MIT |
| `smart-turn-v3.2-cpu.onnx` | upstream Smart Turn v3.2 — the zero-shot baseline | BSD-2-Clause |
| `smart-turn-v3.0.onnx` | upstream Smart Turn v3.0 | BSD-2-Clause |

Pinned by SHA256. A mismatch means the number this project is measured against
has moved — treat it as a finding, not as a stale constant.

`mel_filters.npz` is the 80-mel filterbank and *is* committed: 4 KB, and the
numpy feature path is meaningless without it.

## Ours, published on HuggingFace

[`santhosh-005/smart-turn-tamil`](https://huggingface.co/santhosh-005/smart-turn-tamil)

```
smart-turn-tamil-tiny/   best.pt, smart-turn-tamil.onnx, smart-turn-tamil-int8-dynamic.onnx
smart-turn-tamil-base/   same three
```

**int8 dynamic is what ships**; fp32 is the reference graph. The models are on
the Hub rather than in git because they are 264 MB and the Hub is where the
plugin downloads them from anyway.

| | accuracy | ROC-AUC | FP/N | size |
|---|---|---|---|---|
| tiny int8 | 83.71% | 0.905 | 7.94% | 8.7 MB |
| base int8 | 86.13% | 0.921 | 9.17% | 21 MB |

Rebuild them from a Colab checkpoint with `pipeline/15_export_ckpt.py`
(fp32 ONNX, CPU only) then `pipeline/19_quantize.py` (int8).
