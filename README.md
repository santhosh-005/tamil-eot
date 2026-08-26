# TamilEOT

**The first open-source semantic end-of-turn detector for Tamil.** Given the
last 8 seconds of a speaker's audio, it predicts whether they have finished
their turn — from prosody, without waiting for a transcript.

[![PyPI](https://img.shields.io/pypi/v/smart-turn-livekit)](https://pypi.org/project/smart-turn-livekit/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/santhosh-005/smart-turn-tamil)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/santhosh-005/tamil-eot)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](LICENSE)

| | |
|---|---|
| **paper** | [`paper/tamileot.pdf`](paper/tamileot.pdf) — 12 pages: the method, every measurement, and the negative results |
| **model** | [`santhosh-005/smart-turn-tamil`](https://huggingface.co/santhosh-005/smart-turn-tamil) — int8 ONNX, 8.7 MB / 21 MB, CPU |
| **dataset** | [`santhosh-005/tamil-eot`](https://huggingface.co/datasets/santhosh-005/tamil-eot) — 18,485 labelled boundaries, CC BY 4.0 |
| **plugin** | [`smart-turn-livekit`](https://pypi.org/project/smart-turn-livekit/) — `pip install`, runs on LiveKit Agents |
| **this repo** | the data pipeline, the training method, and every experiment behind those |

This is a self-hostable model that runs on a CPU, with the dataset
and the method published alongside it.

---

## What it does

A voice agent has to decide, at every pause, whether the user has finished
speaking. Without a model of the language, that decision falls back to a fixed
silence timeout — set it short and the agent interrupts, set it long and every
turn pays the full wait.

A semantic detector reads the prosody instead. This one is a fine-tune of
[Smart Turn v3](https://github.com/pipecat-ai/smart-turn), whose released model
covers a set of languages that does not include Tamil.

## Results

Held-out test set: **4,168 clips from 30 calls**. Both models scored locally from their shipped ONNX at threshold 0.5.

| | accuracy | ROC-AUC | FP/N | size | p50, 1 thread |
|---|---|---|---|---|---|
| majority class (`complete`) | 63.08% | 0.500 | – | – | – |
| `smart-turn-v3.2`, zero-shot | 70.30% | 0.751 | – | 8.7 MB | – |
| **tiny int8** — drop-in replacement | 83.71% | 0.905 | **7.94%** | **8.7 MB** | **83 ms** |
| **base int8** — best accuracy | **86.13%** | **0.921** | 9.17% | 21 MB | 143 ms |

**+15.83 points over zero-shot**, and ROC-AUC 0.751 → 0.921.

Both shipped ONNX files were **re-scored locally** and reproduce their training
numbers, and re-exporting the shipped checkpoint gives a graph **bit-identical**
to the published one (`max|delta| 0.00e+00`) — so these figures come from the
artefact you can download, not from a training log.

For scale, Smart Turn's published per-language figures. **Those come from a
different benchmark on TTS-generated audio; this is real narrowband telephone
conversation, so the two are not directly comparable:**

| | accuracy | FP/N |
|---|---|---|
| Hindi | 90.11% | 8.57% |
| **Tamil (base int8)** | **86.13%** | 9.17% |
| Bengali | 83.80% | 10.90% |
| **Tamil (tiny int8)** | 83.71% | **7.94%** |
| Marathi | 82.43% | 15.12% |

`FP/N` is Smart Turn's own convention — FP/N and FN/N sum to the error rate —
not the standard FP/(FP+TN). The two differ by ~3×, so every table here states
which it uses.

> **Treat anything under ~1 point as noise.** Three base runs at identical
> config and `SEED=0` scored 86.23 / 85.63 / 85.36.
> → [experiments/04](experiments/04-training.md)

### Running end-to-end

Run as a complete Tamil voice agent on **LiveKit Agents 1.7**, with an
all-Sarvam stack around it:

| | |
|---|---|
| turn detection | **`smart-turn-tamil`** tiny int8 |
| VAD | Silero, `min_silence_duration=0.25` |
| STT | Sarvam `saarika:v2.5` |
| LLM | Sarvam `sarvam-105b` |
| TTS | Sarvam `bulbul:v3` |

Also verified on **Pipecat 1.7** via `LocalSmartTurnAnalyzerV3`. Adapter latency
end-to-end is **~120–155 ms** on live audio — mel plus ONNX plus the thread
handoff — against the 83 ms inference-only figure above.

On recorded calls against fixed-timeout endpointing, **+120 ms of endpointing
bought −35% fragmentation**. Directional only — one speaker, a few minutes per
arm. Arms, caveats and instrumentation:
→ [experiments/09](experiments/09-live-path.md)

### What serving costs

The accuracy figures above score **pre-cut clips**, ending 0.20 s past the
speech offset. A production endpointer cuts a different window, so the same
labelled boundaries were replayed through the **real Silero VAD** and the **real
streaming adapter** and scored both ways in one run:

| | pre-cut clip | live window |
|---|---|---|
| accuracy | 86.12% | **83.51%** |
| FP/N | 5.21% | 6.51% |

**−2.60 points**, identical verdict on **90.9%** of boundaries (n=461, McNemar
p=0.09 — consistent in direction, not formally significant). The cause is a
90 ms window shift: the VAD closes at +0.29 s where the clips stop at +0.20 s.

**Coverage 92.2%** — LiveKit will not request a prediction below
`min_silence_duration + 50 ms`, so the model is never consulted on the shortest
pauses. → [experiments/09](experiments/09-live-path.md)

---

## How it was built

### Data — 116 real Tamil telephone calls

Source is `SPRING_INX_Tamil_R1` (CC BY 4.0, SPRING Lab, IIT Madras). Its
`ta_IN_*_{Left,Right}` family is **116 two-party calls shipped as one audio file
per speaker**, each leg separately transcribed by a human.

**Which file the audio came from *is* the speaker label** — so speaker
attribution needs no diarization and carries no speaker-error rate. Verified:
116/116 pairs complete, 0 ms duration mismatch between legs.

Turn boundaries come from Silero VAD, not the transcript timestamps: the shipped
segments are padded and summed cover ~107% of the call wall clock. The
transcript is used only to confirm a span is speech on that leg and to supply
text for the backchannel filter. → [docs/dataset.md](docs/dataset.md)

### Labels — measured, then replaced

A blind listening pass over 197 clips scored the rule-derived labels at **70.1%
overall**: 95.9% on the positive class, **44.4% on the negative**.

The positive class is grounded in behaviour recorded in the call — the other
person heard the speaker finish and took the floor. The negative class was
grounded in *"a pause fell inside a transcript segment"*, which answers a
different question than the one the model is trained on.

The replacement was chosen by measurement. Seven Gemini models, same 197 clips,
same prompt: `gemini-3.7-flash` agrees with the human **97.5%** — 97.0% on the
class the rules got wrong. It relabelled all 16,213 clips for **$5.69**, and
**53% of the negative class flipped, against 56% in the independent human pass.**

Relabelling also removed the one structural confound in the set: `prev_dur`
separated the classes at d = −0.33 because the *gate* drew negatives from longer
utterances. Relabelled: **−0.09**.
→ [experiments/02](experiments/02-labelling.md) ·
[experiments/03](experiments/03-relabelling.md)

### Training — Google Colab, one T4

Everything trained in `notebooks/train_smart_turn_tamil.ipynb` on a single
**Colab T4**. No multi-GPU step anywhere in this project.

| | |
|---|---|
| encoder | `openai/whisper-tiny` (8.0M) or `openai/whisper-base` (20.3M) |
| head | attention pooling → binary classifier |
| loss | `BCEWithLogitsLoss`, per-batch `pos_weight` |
| schedule | 6 epochs, lr 5e-5, batch 32, `SEED=0` |
| export | `.pt` → fp32 ONNX → **int8 dynamic**, on CPU |

Both shipped models come from that one notebook — the encoder string is the only
difference between them.

Of every lever tried, **capacity was the only one that moved the number**:

| lever | ΔAUC |
|---|---|
| +20% training rows | +0.003 |
| learning-rate retune | +0.003 |
| 50% → 100% of the data | +0.001 |
| **whisper-tiny → whisper-base** | **+0.018** |

Raising capacity then moved the constraint back to data — at base the learning
curve is still rising and the train−dev gap opens to +11.81 points. *tiny was
capacity-bound; base is data-bound.* → [experiments/04](experiments/04-training.md)

### Dataset

| split | clips | complete | incomplete | calls |
|---|---|---|---|---|
| train | 11,992 | 7,908 | 4,084 | 71 |
| dev | 2,325 | 1,532 | 793 | 15 |
| **test** | **4,168** | 2,629 | 1,539 | **30** |

Split by **call** (not by clip) to prevent voice leakage between splits and ensure true prosodic generalization.

---

## Quickstart: Using the Model

### livekit

```bash
pip install 'smart-turn-livekit[livekit]'
```

```python
from smart_turn_livekit import SmartTurnDetector

SmartTurnDetector()                                  # smart-turn-tamil-tiny
SmartTurnDetector(model="smart-turn-tamil-base")     # +2.4 points, 2.4x size
SmartTurnDetector(model="smart-turn-v3")             # upstream, other languages
```

Weights download from HuggingFace on first use. Runs on CPU (numpy + onnxruntime only — no PyTorch or transformers required). Defaults to **tiny int8 at threshold 0.5**.

### Pipecat

Pipecat's built-in `LocalSmartTurnAnalyzerV3` loads any Smart Turn ONNX model directly:

```python
from smart_turn_livekit import resolve_model
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

LocalSmartTurnAnalyzerV3(smart_turn_model_path=str(resolve_model("smart-turn-tamil-tiny")))
```

Plugin source: [`santhosh-005/smart-turn-livekit`](https://github.com/santhosh-005/smart-turn-livekit).

### Which threshold

**Pick it on `dev`, not test.** 0.5 is inherited from the training loop and
neither model peaks there — but tuning it for accuracy on dev *lost* base 1.13
points on test, so the shipped default stays 0.5.

Targeting a rate transfers better than targeting an argmax: holding false
positives to 10% of pauses roughly halves FP/N (tiny 7.73% → 5.01%) for ~3
accuracy points. → [experiments/07](experiments/07-thresholds.md)

---

## Reproduce

```bash
export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1    # CC BY 4.0, from IIT Madras
pip install -r requirements.txt
python pipeline/00_fetch_models.py
```

Then walk `pipeline/` in order — **00 → 22**, each step writing a file the next
one reads. Training is step 14½, in `notebooks/`, on a Colab T4.

**Two steps reproduce from a bare clone** — no corpus, no GPU, no cloud account:

```bash
python pipeline/12_bakeoff.py            # the labeller bake-off, 97.5% vs human
python pipeline/17_dispute_analysis.py   # label quality per bucket, noise ceiling
```

Both read the text-stripped label tables and labeller verdict caches committed
under `labels/` (1.1 MB gzipped), and both regenerate their reports in `reports/`
byte-identically. Everything else needs the corpus audio or the mel caches that
step 14 builds — 1.5 GB, too large to ship.

→ [pipeline/README.md](pipeline/README.md) for the step table, the credentials
each one needs, and the two steps that must not be re-run casually.

## Layout

```
paper/               the write-up, PDF and LaTeX source
pipeline/            00-22, corpus -> dataset -> model -> release
src/tamileot/        the library: corpus, vad, turns, rules, labelling
notebooks/           the training notebook, plus the executed run behind base
experiments/         every experiment, including the ones that failed
labels/              the label tables, text-stripped and gzipped -- 1.1 MB
reports/             machine-written outputs the experiments cite
docs/                dataset spec, pitfalls, release runbook, archive
integrations-test/   live Tamil voice agent, LiveKit + Sarvam, instrumented
models/              fetched or downloaded, never committed
```

`data/`, `models/*` and `dist/` are build artifacts generated by running the pipeline scripts and are not tracked in git.

## Read next

| | |
|---|---|
| [experiments/](experiments/README.md) | every experiment, with verdicts and the levers ranked |
| [docs/dataset.md](docs/dataset.md) | the label oracle, schema, splits, clip geometry |
| [docs/pitfalls.md](docs/pitfalls.md) | what bit us — each once, each expensive |
| [pipeline/README.md](pipeline/README.md) | the reproducible flow, step by step |

---

## Author

Built and maintained by **Santhosh** ([@santhosh-005](https://github.com/santhosh-005)).

## Licence and citation

Code **BSD-2-Clause**. Data derives from **SPRING_INX Tamil R1** (CC BY 4.0),
SPRING Lab, IIT Madras — the derived dataset carries the same licence. Models
are fine-tunes of `pipecat-ai/smart-turn` (BSD-2-Clause). Full scope in
[LICENSE](LICENSE).

```bibtex
@software{tamileot,
  author = {santhosh-005},
  title  = {TamilEOT: semantic end-of-turn detection for Tamil},
  year   = {2026},
  url    = {https://github.com/santhosh-005/tamil-eot}
}
```
