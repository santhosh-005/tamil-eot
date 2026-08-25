"""Filesystem layout. Everything else imports paths from here.

Only one path has to be supplied from outside: where the SPRING_INX Tamil
corpus is unpacked. It is ~36 h of audio under a CC BY 4.0 licence that has to
be downloaded from SPRING Lab, IIT Madras, so it cannot live in this repo.

    export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1

Everything else is derived from the repo root, so a clone works with no
configuration until a step actually needs the audio.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

# The corpus. R1 is what this project is built on -- 116 stereo telephone calls,
# one speaker per channel. R2 was audited and rejected: see
# docs/archive/DATASET_AUDIT_SPRING_INX_R2.md.
R1 = Path(os.environ.get("SPRING_INX_R1", PROJECT.parent / "SPRING_INX_Tamil_R1"))
R2 = Path(os.environ.get("SPRING_INX_R2", PROJECT.parent / "SPRING_INX_Tamil_R2"))

MODELS = PROJECT / "models"
SILERO_ONNX = MODELS / "silero_vad.onnx"

DATA = PROJECT / "data"
MANIFEST = DATA / "manifest"
VAD = DATA / "vad"            # one .npy of frame probabilities per recording
GOLD = DATA / "gold"          # label tables
CLIPS = DATA / "clips"        # extracted 8 s wav windows
REPORTS = PROJECT / "reports"

# Label tables, text-stripped and gzipped, committed to the repo.
#
# `data/` is build output and gitignored -- 7.9 GB of audio and caches. But the
# label tables are the contribution, they are small, and without them a clone
# cannot run a single analysis step. So a redistributable copy ships here: same
# rows, same labels, minus the transcript-derived fields (`text`, `next_text`,
# `llm_reason`). No analysis step reads those, and the published HuggingFace
# dataset does not carry them either.
#
# `labelling.read_lines()` prefers `data/gold/` and falls back to here, so a
# fresh clone works and a full local run is never shadowed by the shipped copy.
LABELS = PROJECT / "labels"

for _d in (MODELS, MANIFEST, VAD, GOLD, CLIPS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def require_corpus(root: Path = R1) -> Path:
    """Fail with the fix, not with a FileNotFoundError forty lines in."""
    if not (root / "Audio").is_dir():
        raise SystemExit(
            f"SPRING_INX corpus not found at {root}\n"
            "  export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1\n"
            "  SPRING Lab, IIT Madras -- https://asr.iitm.ac.in/  (CC BY 4.0)\n"
            "  paper: https://arxiv.org/abs/2310.14654"
        )
    return root
