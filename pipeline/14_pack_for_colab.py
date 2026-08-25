#!/usr/bin/env python3
"""Step 2b -- pack the dataset into a few large files for Drive + Colab.

Uploading 16,216 loose wavs to Google Drive is slow, and reading them back
through a Colab Drive mount is far slower still -- per-file overhead dominates
and a training epoch spends its life in FUSE round-trips. So the whole corpus
goes into one FLAC per split plus a JSONL index.

FLAC because it is lossless and this audio is narrowband telephony, which
compresses to about a third: **4.15 GB of PCM becomes ~1.4 GB**, and the
samples that come back are bit-identical. Anything lossy would silently change
what the model hears relative to every measurement already taken.

Every clip occupies exactly CLIP_SAMPLES samples in the stream, so clip *i* is
just `[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES]` -- no offset table to get
wrong. The seven clips shorter than 8 s are right-padded, matching what
WhisperFeatureExtractor does, and their true length is in the index so the
preprocessing can be reproduced exactly.

    python pipeline/14_pack_for_colab.py
    python pipeline/14_pack_for_colab.py --verify      # decode and compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tamileot.paths import CLIPS, GOLD  # noqa: E402

SR = 16_000
CLIP_SAMPLES = 8 * SR
OUT = GOLD.parent / "colab"

# Carried through to the HF dataset so the upstream schema is a rename away.
LANGUAGE = "tam"
DATASET_TAG = "tamil_eot_spring_inx_r1"


SAMPLES = "samples_llm.jsonl"   # set by --samples


def rows_for(split: str) -> list[dict]:
    rows = [json.loads(l) for l in (GOLD / SAMPLES).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("llm_ok") and r["split"] == split]
    return sorted((r for r in rows if (CLIPS / split / f"{r['sid']}.wav").exists()),
                  key=lambda r: r["sid"])


def pack(split: str) -> tuple[int, int]:
    rows = rows_for(split)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"tamileot_{split}.flac"
    short = 0
    with sf.SoundFile(path, "w", samplerate=SR, channels=1,
                      format="FLAC", subtype="PCM_16") as out:
        for k, r in enumerate(rows, 1):
            x, sr = sf.read(CLIPS / split / f"{r['sid']}.wav", dtype="int16")
            assert sr == SR, f"{r['sid']}: {sr} Hz"
            r["n_samples"] = int(len(x))
            if len(x) < CLIP_SAMPLES:
                short += 1
                x = np.concatenate([x, np.zeros(CLIP_SAMPLES - len(x), np.int16)])
            out.write(x[:CLIP_SAMPLES])
            if k % 2000 == 0:
                print(f"    {k:,}/{len(rows):,}")
    return len(rows), short


def index(splits: list[str]) -> Path:
    path = OUT / "tamileot_index.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for split in splits:
            for i, r in enumerate(rows_for(split)):
                p = CLIPS / split / f"{r['sid']}.wav"
                f.write(json.dumps({
                    "sid": r["sid"], "split": split, "i": i,
                    "n_samples": int(sf.info(p).frames),
                    # Smart Turn's own schema, so the HF conversion is a rename.
                    "endpoint_bool": int(r["label"]),
                    "language": LANGUAGE, "dataset": DATASET_TAG,
                    # kept so the label policy stays a decision, not a fact
                    "label_pipeline": int(r["label_pipeline"]),
                    "llm_verdict": int(r["llm_verdict"]),
                    "dispute": bool(r["dispute"]),
                    "llm_conf": float(r.get("llm_conf", 0.0)),
                    "source": r["source"], "call": r["base"],
                    "gap": r["gap"], "prev_dur": r["prev_dur"],
                    # which pass produced the row (`core` / `relaxed`), so a
                    # run can be scored on the frozen benchmark separately.
                    # Provenance, never a feature: unavailable at inference.
                    "harvest": r.get("harvest", "core"),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"  index: {n:,} rows -> {path.name}")
    return path


def verify(splits: list[str], per_split: int = 40) -> int:
    """Decode the packed stream and compare against the original wavs.

    FLAC is lossless, so any mismatch here is an indexing bug -- an off-by-one
    in the stream would shift every label onto the wrong clip and would not
    show up until the model mysteriously failed to learn.
    """
    idx: dict[str, list[dict]] = {}
    for l in (OUT / "tamileot_index.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        idx.setdefault(r["split"], []).append(r)
    bad = 0
    for split in splits:
        rows = idx[split]
        data, sr = sf.read(OUT / f"tamileot_{split}.flac", dtype="int16")
        assert sr == SR
        exp = len(rows) * CLIP_SAMPLES
        ok_len = len(data) == exp
        print(f"  {split:6s} {len(rows):6,d} clips   stream {len(data):,} samples "
              f"{'OK' if ok_len else f'MISMATCH (expected {exp:,})'}")
        bad += not ok_len
        pick = np.linspace(0, len(rows) - 1, min(per_split, len(rows))).astype(int)
        for j in pick:
            r = rows[j]
            got = data[j * CLIP_SAMPLES:(j + 1) * CLIP_SAMPLES]
            ref, _ = sf.read(CLIPS / split / f"{r['sid']}.wav", dtype="int16")
            n = r["n_samples"]
            if not np.array_equal(got[:n], ref[:n]) or got[n:].any():
                print(f"    MISMATCH {r['sid']} (i={j})")
                bad += 1
    print(f"  -> {'all sampled clips are bit-identical' if not bad else f'{bad} PROBLEMS'}")
    return 1 if bad else 0


def main() -> int:
    global SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--samples", default="samples_llm.jsonl",
                    help="row file to pack; samples_llm_all.jsonl adds the 03c harvest")
    a = ap.parse_args()
    SAMPLES = a.samples
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]

    if a.verify:
        print("verifying packed audio against the original clips")
        return verify(splits)

    total = 0
    for split in splits:
        print(f"packing {split} ...")
        n, short = pack(split)
        mb = (OUT / f"tamileot_{split}.flac").stat().st_size / 1e6
        total += mb
        print(f"  {n:,} clips, {short} right-padded, {mb:,.0f} MB")
    index(splits)

    print(f"\n  {total:,.0f} MB of audio in {len(splits)} files  ->  {OUT}")
    print("  upload that whole folder to Drive; run --verify first if you like")
    return verify(splits)


if __name__ == "__main__":
    raise SystemExit(main())
