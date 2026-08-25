#!/usr/bin/env python3
"""Step 0 -- fetch the third-party weights the pipeline needs.

Three files, none of them ours, none of them in the repo:

  silero_vad.onnx           step 02 runs it over all 232 legs
  smart-turn-v3.2-cpu.onnx  step 13 scores it zero-shot -- the number to beat
  smart-turn-v3.0.onnx      the same, for the earlier release

Every download is checked against a recorded SHA256. That is not ceremony: the
zero-shot baseline is the figure this whole project is measured against, and a
silently different VAD or a silently different upstream checkpoint would move
it without anything failing. A hash mismatch here is a real finding -- report
it rather than deleting the expected value.

    python pipeline/00_fetch_models.py
    python pipeline/00_fetch_models.py --check    # verify, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot.paths import MODELS  # noqa: E402

# Silero pin a commit rather than a tag: the v5.1 tag carries a *different*
# 2,327,524-byte file, so "silero v5" alone does not identify a graph.
SILERO_COMMIT = "bfdc0193023f121ea5b3cc7b176dbed570a68a59"

WANTED = [
    {
        "name": "silero_vad.onnx",
        "url": f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_COMMIT}"
               "/src/silero_vad/data/silero_vad.onnx",
        "sha256": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        "note": "Silero VAD v5, MIT. Used by step 02.",
    },
    {
        "name": "smart-turn-v3.2-cpu.onnx",
        "hf": ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"),
        "sha256": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
        "note": "upstream Smart Turn v3.2, int8. The zero-shot baseline.",
    },
    {
        "name": "smart-turn-v3.0.onnx",
        "hf": ("pipecat-ai/smart-turn-v3", "smart-turn-v3.0.onnx"),
        "sha256": "07a133aba31e2d0b523f17f8c2e4e65efe6d8f685efd12ca4fe21ebf4e798991",
        "note": "upstream Smart Turn v3.0, fp32.",
    },
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: dict) -> Path:
    dest = MODELS / spec["name"]
    if dest.exists():
        return dest
    if "hf" in spec:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit("pip install huggingface_hub") from None
        repo, fname = spec["hf"]
        print(f"  {spec['name']:26s} <- hf:{repo}")
        src = Path(hf_hub_download(repo_id=repo, filename=fname))
        dest.write_bytes(src.read_bytes())
    else:
        print(f"  {spec['name']:26s} <- {spec['url'][:58]}...")
        with urllib.request.urlopen(spec["url"], timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify what is on disk; download nothing")
    a = ap.parse_args()

    bad = 0
    for spec in WANTED:
        dest = MODELS / spec["name"]
        if not dest.exists():
            if a.check:
                print(f"  {spec['name']:26s} MISSING")
                bad += 1
                continue
            dest = fetch(spec)

        got = sha256(dest)
        want = spec["sha256"]
        mb = dest.stat().st_size / 1e6
        if not want:
            print(f"  {spec['name']:26s} {mb:6.1f} MB  sha256 {got[:16]}  (unpinned)")
        elif got == want:
            print(f"  {spec['name']:26s} {mb:6.1f} MB  ok")
        else:
            print(f"  {spec['name']:26s} {mb:6.1f} MB  SHA MISMATCH\n"
                  f"      expected {want}\n"
                  f"      got      {got}\n"
                  "      Upstream changed the file. Every number measured against it is\n"
                  "      now measured against something else -- do not silently re-pin.")
            bad += 1

    print(f"\n  models in {MODELS}")
    if bad:
        print(f"  {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
