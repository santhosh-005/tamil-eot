#!/usr/bin/env python3
"""Step 4c -- stage `santhosh-005/tamil-turns` for the Hub.

A **separate repo**, not a second config on `santhosh-005/tamil-eot`. The two
are different units of analysis -- 8 s windows with a binary label versus whole
turns with silence structure -- and `task_categories`, `tags` and
`size_categories` are repo-level on the Hub, so they can only ever describe
one of them. Those fields drive Hub search, and a turn-level corpus buried
under `audio-classification` tags is a corpus nobody finds. Keeping them apart
also leaves the artefact the paper cites frozen exactly as published.

The name is `tamil-turns`, not `tamil-eot-turns`: end-of-turn is one use of
this data. It also carries mid-turn pauses, overlap and interruption (a
negative `eot_gap`), and per-turn timing -- naming it for endpointing alone
would undersell the other three.

## What ships

All **4,774 `clean` turns** (see docs/turns_format.md for the six gates), not
just the subset eot-bench can score. `eot_bench_eligible` marks the 4,322 that
meet `livekit/eot-bench-data`'s release rules, so the benchmark subset is one
filter away while the fast handovers (`eot_gap < 0.2`, 298 rows) and long
lapses (`> 5.0`, 154 rows) stay available -- those are exactly the rows an
interruption or ASR-timing study wants and an endpointing benchmark does not.

## silence_spans follows eot-bench

`livekit/eot-bench` treats the **last** span as the end-of-turn:

    label = "eot" if span_index == len(silence_spans) - 1 else "hold"

So the published rows fold the end-of-turn back in, which is the one place
this differs from `data/gold/turns.jsonl`:

    local      silence_spans = mid-turn pauses only,  eot_gap separate
    published  silence_spans = mid-turn pauses + [the end-of-turn span]

The local separation exists so `24_verify_turns.py` can assert that every span
lies inside its turn -- an end-of-turn span never can. `eot_gap` ships as its
own column either way, so the two conventions always reconcile.

Spans are **rebased to the clip**: `iter_transformed_row` indexes audio as
`array[:timestamp * sr]`, so 0 must be the first audio sample, not call time.

Trailing audio is capped at `MAX_SILENCE`. For the 154 rows whose gap is longer,
the final span is the capped length while `eot_gap` keeps the true value; those
rows are `eot_bench_eligible = False`, so no benchmark consumer sees the gap.

    export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1
    python pipeline/25_pack_turns_hf.py          # ~5 min, writes ~1.0 GB
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from tamileot import audio  # noqa: E402
from tamileot.paths import GOLD, PROJECT, R1, require_corpus  # noqa: E402

REPO = "santhosh-005/tamil-turns"
OUT = PROJECT / "dist" / "hf" / "tamil-turns"
SPLITS = ("train", "dev", "test")
SHARD_BYTES = 450 * 1024 * 1024

# livekit/eot-bench-data's own release rules, from its dataset card.
MIN_EOT_SILENCE = 0.20
MAX_SILENCE = 5.00

SCHEMA = pa.schema([
    # --- the four eot-bench requires ---
    ("id", pa.string()),
    ("language", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("silence_spans", pa.list_(pa.struct([("start", pa.float32()),
                                          ("end", pa.float32())]))),
    # --- optional, read by the text-conditioned adapters ---
    ("messages", pa.list_(pa.struct([("role", pa.string()),
                                     ("content", pa.string())]))),
    # --- turn structure ---
    ("duration", pa.float32()),
    ("eot_gap", pa.float32()),
    ("speech_s", pa.float32()),
    ("word_rate", pa.float32()),
    ("n_spans", pa.int32()),
    ("n_absorbed", pa.int32()),
    ("pause_crosstalk", pa.float32()),
    ("speech_crosstalk", pa.float32()),
    ("transcript", pa.string()),
    ("conversation_id", pa.string()),
    ("speaker_id", pa.string()),
    ("turn_index", pa.int32()),
    ("split", pa.string()),
    ("eot_bench_eligible", pa.bool_()),
    # --- joined from the clips dataset where a labelled boundary lands here ---
    ("label", pa.int8()),
    ("source", pa.string()),
    ("sid", pa.string()),
])


def eligible(r: dict) -> bool:
    """Meets livekit/eot-bench-data's release rules."""
    return (MIN_EOT_SILENCE <= r["eot_gap"] <= MAX_SILENCE
            and all(s["end"] - s["start"] <= MAX_SILENCE for s in r["silence_spans"]))


def trail_for(r: dict) -> float:
    return min(r["eot_gap"], MAX_SILENCE)


def spans_for(r: dict) -> list[dict]:
    """Mid-turn pauses plus the end-of-turn, rebased so 0 is the clip start."""
    t0 = r["start_time"]
    out = [{"start": round(s["start"] - t0, 3), "end": round(s["end"] - t0, 3)}
           for s in r["silence_spans"]]
    d = round(r["end_time"] - t0, 3)
    out.append({"start": d, "end": round(d + trail_for(r), 3)})
    return out


def messages_for(r: dict, history: dict[str, list[dict]]) -> list[dict]:
    """**Exactly one** `assistant` message: the other leg's preceding turn.

    This matches `livekit/eot-bench-data`, where all 400 Hindi rows carry a
    single `role: assistant` message -- never a `user` one, never two. The
    harness supplies the user side itself, appending the current speaker's
    progressively-revealed words in `build_messages`.

    Shipping the full history instead (56 turns on the median row) would hand
    Tamil far more context than any other language in the benchmark and make
    the numbers incomparable, which is the whole reason for going upstream.
    """
    for p in reversed(history[r["conversation_id"]]):
        if (p["turn_index"] < r["turn_index"]
                and p["speaker_id"] != r["speaker_id"]
                and p["transcript"].strip()):
            return [{"role": "assistant", "content": p["transcript"]}]
    return []


def flac_bytes(base: str, side: str, a: float, b: float) -> bytes:
    wav = R1 / "Audio" / f"{base}_{side}.wav"
    x = audio.read_window(wav, a, b)
    buf = io.BytesIO()
    sf.write(buf, (x * 32767.0).astype("int16"), audio.SR,
             format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def pack_split(rs: list[dict], history: dict, out: Path, split: str) -> None:
    planned, size, cur = [], 0, []
    for r in rs:
        side = "Left" if r["speaker_id"].endswith("_L") else "Right"
        b = flac_bytes(r["conversation_id"], side,
                       r["start_time"], r["end_time"] + trail_for(r))
        cur.append((r, b))
        size += len(b)
        if size >= SHARD_BYTES:
            planned.append(cur)
            size, cur = 0, []
    if cur:
        planned.append(cur)

    for i, chunk in enumerate(planned):
        path = out / f"{split}-{i:05d}-of-{len(planned):05d}.parquet"
        col = lambda k: [r[k] for r, _ in chunk]  # noqa: E731
        tbl = pa.Table.from_pydict({
            "id": col("id"),
            "language": col("language"),
            "audio": [{"bytes": b, "path": f"{r['id']}.flac"} for r, b in chunk],
            "silence_spans": [spans_for(r) for r, _ in chunk],
            "messages": [messages_for(r, history) for r, _ in chunk],
            "duration": col("duration"),
            "eot_gap": col("eot_gap"),
            "speech_s": col("speech_s"),
            "word_rate": col("word_rate"),
            "n_spans": col("n_spans"),
            "n_absorbed": col("n_absorbed"),
            "pause_crosstalk": col("pause_crosstalk"),
            "speech_crosstalk": col("speech_crosstalk"),
            "transcript": col("transcript"),
            "conversation_id": col("conversation_id"),
            "speaker_id": col("speaker_id"),
            "turn_index": col("turn_index"),
            "split": col("split"),
            "eot_bench_eligible": [eligible(r) for r, _ in chunk],
            "label": col("label"),
            "source": col("source"),
            "sid": col("sid"),
        }, schema=SCHEMA)
        pq.write_table(tbl, path, compression="zstd")
        print(f"    {path.name}  {len(chunk):,} turns  "
              f"{path.stat().st_size / 1e6:.0f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card-only", action="store_true")
    a = ap.parse_args()
    require_corpus(R1)

    rows = [json.loads(l) for l in
            (GOLD / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    history: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        history[r["conversation_id"]].append(r)
    for v in history.values():
        v.sort(key=lambda r: r["turn_index"])

    keep = [r for r in rows if r["clean"]]
    keep.sort(key=lambda r: (r["conversation_id"], r["turn_index"]))
    n = Counter(r["split"] for r in keep)
    n_el = sum(1 for r in keep if eligible(r))

    print(f"turns.jsonl {len(rows):,} -> clean {len(keep):,}")
    print(f"  eot_bench_eligible {n_el:,}  "
          f"(fast handover < {MIN_EOT_SILENCE}s: "
          f"{sum(1 for r in keep if r['eot_gap'] < MIN_EOT_SILENCE):,}, "
          f"long lapse > {MAX_SILENCE}s: "
          f"{sum(1 for r in keep if r['eot_gap'] > MAX_SILENCE):,})")
    for s in SPLITS:
        sel = [r for r in keep if r["split"] == s]
        print(f"  {s:5s} {n[s]:5,d} turns  {len({r['conversation_id'] for r in sel}):3d} calls"
              f"  {sum(1 for r in sel if r['label'] is not None):5,d} labelled"
              f"  {sum(1 for r in sel if eligible(r)):5,d} eligible")

    if not a.card_only:
        (OUT / "data").mkdir(parents=True, exist_ok=True)
        for f in (OUT / "data").glob("*.parquet"):
            f.unlink()
        print(f"\nwriting -> {OUT / 'data'}")
        for s in SPLITS:
            pack_split([r for r in keep if r["split"] == s], history,
                       OUT / "data", s)

    (OUT / "README.md").write_text(card(keep, n, n_el), encoding="utf-8")
    print(f"  card -> {OUT / 'README.md'}")

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9
    print(f"\n  {OUT.name}: {size:.2f} GB")
    print(f"\n  next: review {OUT}, then\n"
          f"    hf upload {REPO} {OUT} . --repo-type=dataset")
    return 0


def card(keep: list[dict], n: Counter, n_el: int) -> str:
    """The dataset card.

    Section coverage deliberately mirrors the `tamil-eot` card, so a reader
    moving between the two finds the same things in the same order: source and
    how it was built, a schema table, a decode workaround, limitations, and the
    same attribution and citation blocks. The two cards cross-link each other
    in their opening paragraphs.
    """
    calls = len({r["conversation_id"] for r in keep})
    lab = sum(1 for r in keep if r["label"] is not None)
    hours = sum(r["duration"] + trail_for(r) for r in keep) / 3600
    pauses = sum(len(r["silence_spans"]) for r in keep)
    files = "\n".join(f"  - split: {s}\n    path: data/{s}-*" for s in SPLITS)
    el = lambda sp: sum(1 for r in keep if r["split"] == sp and eligible(r))  # noqa: E731
    return f"""---
license: cc-by-4.0
language: [ta]
task_categories: [audio-classification, automatic-speech-recognition]
tags:
- turn-taking
- end-of-turn
- turn-detection
- endpointing
- conversational
- overlap
- interruption
- voice-agent
- tamil
- 'arxiv:2609.05631'
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files:
{files}
---

# tamil-turns

**{len(keep):,} conversational turns** from {calls} real Tamil telephone calls —
one speaker's continuous hold of the floor, with their own mid-turn pauses kept
separate from the silence that ends the turn.

> **Companion dataset.**
> [**`santhosh-005/tamil-eot`**](https://huggingface.co/datasets/santhosh-005/tamil-eot)
> is the same source calls cut as **8 s windows with a binary complete /
> incomplete label** — use that one to *train an end-of-turn classifier*.
> **This** dataset keeps **whole turns with their silence structure** — use it
> for turn-taking, mid-turn pauses, overlap and interruption, or timing
> analysis. Same corpus, same call-level split, different unit of analysis.

| | |
|---|---|
| turns | {len(keep):,} — 16 kHz mono FLAC |
| source calls | {calls} |
| audio | {hours:.1f} h |
| mid-turn pauses | {pauses:,} |
| train / dev / test | {n['train']:,} / {n['dev']:,} / {n['test']:,} |
| turn ends with a joined label | {lab:,} |

```python
from datasets import load_dataset
ds = load_dataset("santhosh-005/tamil-turns", split="test")
```

`datasets` ≥ 4 needs `torchcodec` to decode the `audio` column. To avoid that
dependency, read the FLAC bytes directly:

```python
import io, soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("santhosh-005/tamil-turns", split="test").cast_column("audio", Audio(decode=False))
x, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
```

## What it is good for

| | |
|---|---|
| **end-of-turn / endpointing** | the last `silence_spans` entry is the real end-of-turn |
| **mid-turn pauses** | every earlier span is the speaker hesitating, not finishing |
| **overlap and interruption** | `speech_crosstalk` measures the other leg talking across the turn; a negative `eot_gap` means they started before this speaker stopped |
| **ASR / timing** | `speech_s`, `word_rate`, `n_spans` and the span list give per-turn timing |

## Structure

**Split by call, not by turn** — no call appears in two splits, and the split is
the same one `tamil-eot` uses, so the two datasets agree on which calls are held
out.

| field | meaning |
|---|---|
| `id` | `<call>_<L\|R>_<turn_index>` |
| `audio` | 16 kHz mono FLAC — **that speaker's leg only, never a mixdown** |
| `language` | `ta` |
| `silence_spans` | `{{start, end}}` in clip time. **The last is the end-of-turn**; earlier ones are mid-turn pauses |
| `messages` | the other speaker's preceding turn |
| `duration` | speech portion, before the end-of-turn silence |
| `eot_gap` | true length of the final silence |
| `speech_s` | seconds actually spoken (`duration` minus the pauses) |
| `word_rate` | `n_words / speech_s` |
| `n_spans`, `n_absorbed` | VAD spans in the turn; backchannels folded in |
| `pause_crosstalk`, `speech_crosstalk` | other-leg speech inside a pause / across the whole turn |
| `transcript` | words apportioned to the turn — **approximate**, see *Considerations* |
| `conversation_id`, `speaker_id`, `turn_index` | sort by `turn_index` to rebuild a call |
| `split` | `train` / `dev` / `test`, **by call** |
| `eot_bench_eligible` | final silence ≥ 0.2 s and no span > 5.0 s ({n_el:,} rows) |
| `label`, `source`, `sid` | joined from `tamil-eot` where a labelled boundary lands on this turn's end; `null` otherwise |

| split | turns | eligible |
|---|---|---|
| train | {n['train']:,} | {el('train'):,} |
| dev | {n['dev']:,} | {el('dev'):,} |
| test | {n['test']:,} | {el('test'):,} |

`silence_spans` follows the layout used by
[`livekit/eot-bench`](https://github.com/livekit/eot-bench) — last span is the
end-of-turn, earlier ones are holds — so the rows are consumable by that kind of
harness without conversion. `eot_bench_eligible` marks the rows meeting its
usual release rules; the other {len(keep) - n_el:,} are fast handovers and long
lapses, kept because interruption and timing work wants exactly those.

## Dataset creation

### Source

Derived from **SPRING_INX Tamil R1** (SPRING Lab, IIT Madras, CC BY 4.0), which
ships each call as **one audio file per speaker**, separately transcribed. That
makes the speaker label structural rather than inferred — no diarization, and no
speaker-error rate to propagate. It is also what makes turn construction
possible at all: which file the audio came from *is* the speaker.

### How turns were built

Turns come from **floor occupancy** over VAD spans confirmed against the human
transcript, never from a turn classifier. Maximal same-side runs, with
backchannels absorbed (< 3 words **and** < 1.0 s) and a split at any pause that
is not a hesitation — over 2.0 s with the floor empty, or one the other speaker
talks across.

Six gates then decide `clean`, and **only clean turns are published**: the turn
opens and closes a transcript segment (so it never starts or ends mid-word), the
other speaker is not talking across it, the handover is not overlapped, at least
1.0 s of speech, and at least 1.0 word per second of speech.

That gate is strict on purpose. Of 21,596 raw turns, {len(keep):,} are clean —
spot-listening the ungated set turned up fragments cut mid-word and "pauses"
that were really the other speaker holding the floor.

### Labels

This dataset is **structural, not labelled** — `silence_spans`, `eot_gap` and
the floor layout are derived from audio and transcript, not from anyone's
judgement, so no labelling was needed for them.

Where a labelled boundary from `tamil-eot` lands exactly on a turn's end, that
label is joined in ({lab:,} of {len(keep):,} turns). Those labels come from an
audio LLM that agreed with a human listener on **97.5%** of 197 blind-listened
clips; the labeller bake-off and its limits are documented on the
[`tamil-eot` card](https://huggingface.co/datasets/santhosh-005/tamil-eot).
`label` is `null` on the remaining turns rather than guessed.

## Considerations

- **One domain** — narrowband two-party telephone conversations, task-oriented.
- **`transcript` is approximate.** SPRING-INX ships segment-level transcripts,
  so words are apportioned across VAD speech seconds rather than force-aligned.
  No word-level timings ship, and `transcript` must not be treated as one.
- **Turns ending in overlap are excluded.** The `clean` gate requires
  `eot_gap >= 0`, so turns the other speaker cut into are not here. They exist
  in the pipeline and can be rebuilt from it.
- **Trailing audio is capped at {MAX_SILENCE:.0f} s.** For the
  {sum(1 for r in keep if r['eot_gap'] > MAX_SILENCE):,} rows with a longer gap
  the final span is the capped length while `eot_gap` keeps the true value; all
  of those are `eot_bench_eligible = False`.
- **Turn length is long-tailed.** Most turns are a few seconds, but genuine
  monologues run to several minutes — real, not a segmentation artefact.
- Audio is conversational speech from a public corpus; no personally identifying
  metadata is included beyond what SPRING_INX ships.

## Licensing and citation

The **dataset** — turn construction, the quality gates, and the splits — is by
[santhosh-005](https://github.com/santhosh-005), released under **CC BY 4.0**.

```bibtex
@article{{tamileot,
  author  = {{Santhoshkumar V}},
  title   = {{TamilEOT: A Dataset and Model for Semantic End-of-Turn
             Detection in Tamil Telephone Speech}},
  journal = {{arXiv preprint arXiv:2609.05631}},
  year    = {{2026}},
  url     = {{https://arxiv.org/abs/2609.05631}}
}}
```

The **source audio** is from SPRING_INX Tamil R1 and remains the work of SPRING
Lab, IIT Madras. It is redistributable here under CC BY 4.0, with attribution:

> SPRING Lab, Indian Institute of Technology Madras. *SPRING_INX Tamil R1.*
> CC BY 4.0. <https://asr.iitm.ac.in/SPRING_INX>

## Related

| | |
|---|---|
| [`santhosh-005/tamil-eot`](https://huggingface.co/datasets/santhosh-005/tamil-eot) | the same calls as 8 s clips with a binary end-of-turn label — **for training a classifier** |
| [`santhosh-005/smart-turn-tamil`](https://huggingface.co/santhosh-005/smart-turn-tamil) | the end-of-turn model trained on `tamil-eot` |
| [`smart-turn-livekit`](https://pypi.org/project/smart-turn-livekit/) | that model as a LiveKit Agents plugin |
| [github.com/santhosh-005/tamil-eot](https://github.com/santhosh-005/tamil-eot) | how all of it was built, including the experiments that failed |
"""


if __name__ == "__main__":
    raise SystemExit(main())
