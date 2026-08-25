#!/usr/bin/env python3
"""Step 3a -- stage the two HuggingFace repos under `dist/hf/`.

Nothing is uploaded here. This builds two directories whose contents are
*exactly* what goes to the Hub, so the upload is one command per repo and the
review happens before anything is public. Then, once you have checked what is
in `dist/hf/`:

    hf auth login
    hf upload <owner>/tamil-eot        dist/hf/dataset . --repo-type=dataset
    hf upload <owner>/smart-turn-tamil dist/hf/model   . --repo-type=model

Publish the research repo *first*: both cards link to it, so uploading before
it is public ships two dead links.

  dist/hf/model/     ONNX + torch checkpoints + model card
  dist/hf/dataset/   parquet shards + dataset card

The model directory mirrors `models/` on purpose: the shipped package resolves
weights by that layout, so

    hf download <repo> --local-dir ./models
    # weights download from HF on first use

works with no further wiring. A flat directory would break it -- tiny and base
share a filename.

Audio goes out as FLAC-in-parquet with the `datasets` Audio struct
(`{bytes, path}`), which is what makes the Hub's dataset viewer play the clips.
Numbers in both cards are read from `reports/`, never typed in.

    python pipeline/21_pack_for_hf.py            # ~2 min, writes ~2.0 GB
    python pipeline/21_pack_for_hf.py --cards    # rewrite the cards only
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from tamileot.paths import CLIPS, GOLD, MODELS, PROJECT  # noqa: E402

DIST = PROJECT / "dist" / "hf"
MODEL_DIR, DATA_DIR = DIST / "model", DIST / "dataset"

INDEX = GOLD.parent / "colab" / "tamileot_index.jsonl"
SPLITS = ("train", "dev", "test")
SHARD_BYTES = 400 * 1024 * 1024      # Hub guidance: keep shards under ~500 MB

# Where the released artefacts live. Override to stage a fork:
#   export HF_OWNER=your-username
OWNER = os.environ.get("HF_OWNER", "santhosh-005")
MODEL_REPO = f"{OWNER}/smart-turn-tamil"
DATA_REPO = f"{OWNER}/tamil-eot"

# What ships, and under which name on the Hub.
WEIGHTS = [
    ("smart-turn-tamil-tiny/smart-turn-tamil-int8-dynamic.onnx", "tiny, int8 dynamic — the default"),
    ("smart-turn-tamil-base/smart-turn-tamil-int8-dynamic.onnx", "base, int8 dynamic"),
    ("smart-turn-tamil-tiny/smart-turn-tamil.onnx", "tiny, fp32 reference graph"),
    ("smart-turn-tamil-base/smart-turn-tamil.onnx", "base, fp32 reference graph"),
    ("smart-turn-tamil-tiny/best.pt", "tiny torch checkpoint — for further fine-tuning"),
    ("smart-turn-tamil-base/best.pt", "base torch checkpoint — for further fine-tuning"),
]


def rows() -> list[dict]:
    return [json.loads(l) for l in INDEX.read_text(encoding="utf-8").splitlines()]


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


# --------------------------------------------------------------------- audio

SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("endpoint_bool", pa.bool_()),
    ("language", pa.string()),
    ("dataset", pa.string()),
    ("sid", pa.string()),
    ("call", pa.string()),
    ("split", pa.string()),
    ("source", pa.string()),
    ("harvest", pa.string()),
    ("gap", pa.float32()),
    ("prev_dur", pa.float32()),
    ("n_samples", pa.int32()),
    ("label_pipeline", pa.bool_()),
    ("llm_verdict", pa.bool_()),
    ("dispute", pa.bool_()),
    ("llm_conf", pa.float32()),
])


def flac_bytes(path: Path) -> bytes:
    """Re-encode the stored PCM16 wav to FLAC. Lossless, ~3x smaller."""
    data, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def pack_split(split: str, rs: list[dict], out: Path) -> list[Path]:
    """One or more parquet shards, split at SHARD_BYTES of encoded audio."""
    planned = []
    size, cur = 0, []
    for r in rs:
        b = flac_bytes(CLIPS / split / f"{r['sid']}.wav")
        cur.append((r, b))
        size += len(b)
        if size >= SHARD_BYTES:
            planned.append(cur)
            size, cur = 0, []
    if cur:
        planned.append(cur)

    written = []
    for i, chunk in enumerate(planned):
        path = out / f"{split}-{i:05d}-of-{len(planned):05d}.parquet"
        tbl = pa.Table.from_pydict({
            "audio": [{"bytes": b, "path": f"{r['sid']}.flac"} for r, b in chunk],
            "endpoint_bool": [bool(r["endpoint_bool"]) for r, _ in chunk],
            "language": [r["language"] for r, _ in chunk],
            "dataset": [r["dataset"] for r, _ in chunk],
            "sid": [r["sid"] for r, _ in chunk],
            "call": [r["call"] for r, _ in chunk],
            "split": [r["split"] for r, _ in chunk],
            "source": [r["source"] for r, _ in chunk],
            "harvest": [r["harvest"] for r, _ in chunk],
            "gap": [r["gap"] for r, _ in chunk],
            "prev_dur": [r["prev_dur"] for r, _ in chunk],
            "n_samples": [r["n_samples"] for r, _ in chunk],
            "label_pipeline": [bool(r["label_pipeline"]) for r, _ in chunk],
            "llm_verdict": [bool(r["llm_verdict"]) for r, _ in chunk],
            "dispute": [bool(r["dispute"]) for r, _ in chunk],
            "llm_conf": [r["llm_conf"] for r, _ in chunk],
        }, schema=SCHEMA)
        pq.write_table(tbl, path, compression="zstd")
        written.append(path)
        print(f"    {path.name}  {len(chunk):,} clips  {mb(path):.0f} MB")
    return written


# ---------------------------------------------------------------------- cards

def model_card(rs: list[dict]) -> str:
    return f"""---
license: bsd-2-clause
language: [ta]
library_name: onnx
pipeline_tag: audio-classification
tags: [end-of-turn, turn-detection, endpointing, voice-agent, pipecat, livekit, smart-turn, tamil]
datasets: [{DATA_REPO}]
---

# smart-turn-tamil

Semantic **end-of-turn detection for Tamil**. Fine-tune of
[Smart Turn v3](https://github.com/pipecat-ai/smart-turn) on real Tamil
telephone conversations. Drop-in for Pipecat and LiveKit.

Indic voice agents commonly run with semantic turn detection **off**, because
the available turn models do not cover these languages and a wrong turn call is
worse than a fixed silence timeout. This is the missing Tamil model.

## Numbers

Sealed test set: **4,168 clips from 30 calls never trained on.** Threshold 0.5.

| build | accuracy | ROC-AUC | FP/N | size | p50 latency |
|---|---|---|---|---|---|
| `smart-turn-v3.2` zero-shot | 70.30% | 0.751 | – | 8.7 MB | – |
| **tiny int8** ← default | 83.71% | 0.905 | **7.94%** | **8.7 MB** | **83 ms** |
| **base int8** | **86.13%** | **0.921** | 9.17% | 21 MB | 143 ms |

Latency: 1 thread, batch 1, idle i5-12450H, model inference only (+~12 ms mel).
Pipecat defaults to one thread, which is why that is the column quoted.

`FP/N` is Smart Turn's own convention (FP/N + FN/N = error rate), not
FP/(FP+TN) — the two differ by ~3x.

For scale, Smart Turn's published table. **Those figures come from a different
benchmark on TTS-generated audio; this is real narrowband telephony, so the two
are not directly comparable:**

| | accuracy | FP/N |
|---|---|---|
| Hindi | 90.11% | 8.57% |
| **Tamil (base int8)** | **86.13%** | 9.17% |
| Bengali | 83.80% | 10.90% |
| **Tamil (tiny int8)** | 83.71% | **7.94%** |
| Marathi | 82.43% | 15.12% |

## Which one

**tiny.** Smart Turn v3 *is* whisper-tiny, so only the tiny fine-tune is a true
drop-in — and 8.7 MB against upstream's 8.68 MB. base buys **+2.4 points for
2.4x the size and 1.7x the latency**. Both are realtime.

## Files

| | |
|---|---|
{chr(10).join(f"| `{p}` | {d} |" for p, d in WEIGHTS)}
| `mel_filters.npz` | 80-mel filterbank, for the numpy feature path |
| `config.json` | thresholds and metadata |

## Use

Weights only — no torch, no transformers.

```bash
hf download {MODEL_REPO} --local-dir ./models
# weights download from HuggingFace on first use

# adapters (not yet on PyPI)
pip install 'smart-turn-livekit[livekit]'
```

```python
from smart_turn_livekit import SmartTurnDetector
turn_detection = SmartTurnDetector(model="smart-turn-tamil-tiny")
```

Raw ONNX: single input `input_features` of shape `(batch, 80, 800)`, single
output named `logits` that is **already a sigmoid**. Applying another one maps
[0,1] to [0.5,0.73] and every clip predicts `complete`.

## Thresholds

0.5 is inherited from the training loop and neither model peaks there — but
tuning it for accuracy on `dev` *lost* base 1.13 points on test, so **0.5 is
the shipped default.** The robust alternative targets a rate:

| operating point | tiny | base | effect |
|---|---|---|---|
| `inherited` (default) | 0.50 | 0.50 | the numbers above |
| `polite` | 0.75 | 0.92 | FP/N roughly halves, ~3 points of accuracy |

Both picked on `dev`, then measured once on test.

## Training data

[`{DATA_REPO}`](https://huggingface.co/datasets/{DATA_REPO}) —
**{len(rs):,} labelled clips from 116 real Tamil calls**, split by call so no
call appears in two splits. Test frozen from day one.

## Limitations

- **Narrowband telephone conversations.** Wideband or close-mic speech is out of
  distribution.
- **Tamil only.** No claim on other South Indian languages yet.
- **8 s window.** Only the last 8 seconds of audio are read.
- **Offline numbers.** Every figure here is on pre-cut clips, not a live call.
- **Code-switching** with English is common in this corpus and handled, but
  not measured as its own slice.

## Licence & credit

BSD-2-Clause, inherited from `pipecat-ai/smart-turn`. Trained on data derived
from **SPRING_INX Tamil R1** (CC BY 4.0), IIT Madras.

Full method, every negative result, and the four things that did not work:
<https://github.com/{OWNER}/TamilEOT>
"""


def dataset_card(rs: list[dict], counts: dict[str, int]) -> str:
    comp = sum(r["endpoint_bool"] == 1 for r in rs)
    disp = sum(bool(r["dispute"]) for r in rs)
    calls = len({r["call"] for r in rs})
    splits_yaml = "\n".join(
        f"  - name: {s}\n    num_examples: {counts[s]}" for s in SPLITS)
    files_yaml = "\n".join(
        f"  - split: {s}\n    path: data/{s}-*" for s in SPLITS)
    return f"""---
license: cc-by-4.0
language: [ta]
task_categories: [audio-classification]
tags: [end-of-turn, turn-detection, endpointing, voice-agent, tamil, smart-turn]
size_categories: [10K<n<100K]
configs:
- config_name: default
  data_files:
{files_yaml}
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: endpoint_bool
    dtype: bool
  - name: language
    dtype: string
  - name: dataset
    dtype: string
  - name: sid
    dtype: string
  - name: call
    dtype: string
  - name: split
    dtype: string
  - name: source
    dtype: string
  - name: harvest
    dtype: string
  - name: gap
    dtype: float32
  - name: prev_dur
    dtype: float32
  - name: n_samples
    dtype: int32
  - name: label_pipeline
    dtype: bool
  - name: llm_verdict
    dtype: bool
  - name: dispute
    dtype: bool
  - name: llm_conf
    dtype: float32
  splits:
{splits_yaml}
---

# tamil-eot-spring-inx

**{len(rs):,} labelled end-of-turn boundaries from {calls} real Tamil
telephone conversations.** Built so that Indic voice agents can use semantic
turn detection: the released Smart Turn model does not cover Tamil, and no open
Tamil end-of-turn dataset existed.

Same schema as `pipecat-ai`'s Smart Turn datasets — `endpoint_bool`,
`language`, `dataset` — so it trains with their loop unchanged.

## What is in it

| | |
|---|---|
| clips | **{len(rs):,}** — 16 kHz mono FLAC, <=8 s |
| complete / incomplete | {comp:,} / {len(rs) - comp:,} |
| source calls | {calls} |
| train / dev / test | {counts['train']:,} / {counts['dev']:,} / {counts['test']:,} |

**Split by call, never by clip.** 30 calls are sealed as test and have never
moved — byte-identical across every repack.

## Why the labels are trustworthy

The source ships **one audio file per speaker per call**, each leg separately
transcribed by a human. Which file the audio came from *is* the speaker label:
no diarization, no speaker-error rate. That is what made this buildable.

Turn boundaries come from Silero VAD, not from transcript timestamps — the
shipped segments are padded and cover ~107% of the call wall clock, so their
edges are not turn boundaries.

Completeness labels come from **`gemini-3.7-flash` listening to audio only**,
no transcript, no access to the pipeline's guess. Measured before use:
**97.5% agreement with a human** over 197 blind-listened clips, against 70.1%
for the acoustic pipeline it replaced. Seven models were compared on the same
clips; the full table is in the repo.

`label_pipeline` and `llm_verdict` are both kept on every row, with `dispute`
flagging the {disp:,} where they disagree. `endpoint_bool` follows the LLM —
measured and settled, not assumed. `endpoint_bool == llm_verdict` everywhere.

(The repo quotes 6,532 disputed rows; that is the `harvest == "core"` subset of
16,213. This ships core plus the 2,272-clip relaxed harvest.)

**Both classes carry identical trailing silence (200 ms of real audio).** If
positives ended with more silence than negatives, a model would learn to read
silence length instead of speech, score well offline, and collapse in
production.

## Fields

| field | meaning |
|---|---|
| `audio` | 16 kHz mono FLAC |
| `endpoint_bool` | **the label** — true = speaker finished |
| `language` | `tam` |
| `dataset` | `tamil_eot_spring_inx_r1` |
| `sid` | stable clip id: `<call>_<L\\|R>_<offset_ms>` |
| `call` | source call — group by this, never split across it |
| `source` | how the boundary was found: `change`, `hold_intra`, `change_midseg`, `hold_inter` |
| `harvest` | `core` or `relaxed` (the widened-gate second pass) |
| `gap` | silence at the boundary, seconds |
| `prev_dur` | duration of the preceding talk spurt, seconds |
| `n_samples` | true length before any padding |
| `label_pipeline` | the original acoustic label |
| `llm_verdict` | the labeller's verdict |
| `dispute` | the two disagree |
| `llm_conf` | labeller confidence |

## Load

```python
from datasets import load_dataset
ds = load_dataset("{DATA_REPO}")
ds["train"][0]["audio"]          # decoded waveform
ds["train"][0]["endpoint_bool"]  # the label
```

## Known limits

- **Narrowband telephone conversations**, one domain, ~36 h of source audio.
- **Positive-heavy** at {100 * comp / len(rs):.0f}:{100 - 100 * comp / len(rs):.0f}, not 50:50.
- **Label ceiling ~97.1%** — above that you are fitting labeller error.
- **`source` is not balanced**: `hold_intra` is 58% of the test split and is
  where the real end-of-turn problem lives. Report per-`source` accuracy
  against each bucket's own base rate, or it is unreadable.

## Licence & attribution — read this

Derived from **SPRING_INX Tamil R1**, released by IIT Madras under
**CC BY 4.0**. This derived set is redistributed under the same licence.
These are **not** our recordings and we do not claim ownership — the audio is
redistributable because CC BY 4.0 permits it, with attribution.

> SPRING Lab, Indian Institute of Technology Madras. *SPRING_INX Tamil R1.*
> CC BY 4.0. <https://asr.iitm.ac.in/SPRING_INX>

Method, every experiment, and the negative results:
<https://github.com/{OWNER}/TamilEOT>
"""


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", action="store_true", help="rewrite the cards, skip the heavy work")
    a = ap.parse_args()

    rs = rows()
    counts = {s: sum(r["split"] == s for r in rs) for s in SPLITS}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data").mkdir(parents=True, exist_ok=True)

    if not a.cards:
        print(f"model -> {MODEL_DIR}")
        for rel, _ in WEIGHTS:
            src, dst = MODELS / rel, MODEL_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"    {rel}  {mb(dst):.1f} MB")
        shutil.copy2(MODELS / "mel_filters.npz", MODEL_DIR / "mel_filters.npz")
        (MODEL_DIR / "config.json").write_text(json.dumps({
            "model_type": "smart-turn-tamil",
            "base_model": "pipecat-ai/smart-turn-v3.2",
            "language": "ta",
            "sample_rate": 16000,
            "window_seconds": 8,
            "n_mels": 80,
            "input_name": "input_features",
            "input_shape": [1, 80, 800],
            "output_name": "logits",
            "output_is_sigmoid": True,
            "default_variant": "tiny",
            "default_precision": "int8",
            "default_operating_point": "inherited",
            "thresholds": {
                "tiny": {"inherited": 0.50, "balanced": 0.34, "polite": 0.75},
                "base": {"inherited": 0.50, "balanced": 0.24, "polite": 0.92},
            },
        }, indent=2) + "\n", encoding="utf-8")

        print(f"\ndataset -> {DATA_DIR}")
        for s in SPLITS:
            print(f"  {s}: {counts[s]:,} clips")
            pack_split(s, [r for r in rs if r["split"] == s], DATA_DIR / "data")

    # Cards are **hand-maintained after the first generation.** Both have been
    # substantially rewritten in `dist/` since -- restructured, live-pipeline
    # results added, terminology settled -- and none of that is reflected in the
    # templates above. Overwriting would silently revert the published cards to
    # a months-old draft, which is not a failure that announces itself: the push
    # succeeds and the repo just gets worse.
    #
    # So: generate a card only when there is none. Delete the file to opt back
    # into the template.
    for name, card in (("model", model_card(rs)), ("dataset", dataset_card(rs, counts))):
        path = (MODEL_DIR if name == "model" else DATA_DIR) / "README.md"
        if path.exists():
            print(f"  {name} card exists, left alone -- edit {path} by hand")
        else:
            path.write_text(card, encoding="utf-8")
            print(f"  {name} card written from template -> {path}")

    def total(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

    print(f"\n  model   {total(MODEL_DIR):.2f} GB  -> {MODEL_DIR}")
    print(f"  dataset {total(DATA_DIR):.2f} GB  -> {DATA_DIR}")
    print(f"\n  next: review dist/hf/, then\n"
          f"    hf upload {DATA_REPO}  {DATA_DIR}  . --repo-type=dataset\n"
          f"    hf upload {MODEL_REPO} {MODEL_DIR} . --repo-type=model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
