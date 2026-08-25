"""Readers for the Kaldi-style files SPRING_INX ships.

The releases split every table across train/dev/eval, but those splits are
utterance-level random draws from the same recordings (511 of 545 R2
recordings appear in both train and eval), so they are useless as splits and
we always read the union. See docs/archive/DATASET_AUDIT_SPRING_INX_R2.md section 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "dev", "eval")


@dataclass(frozen=True, slots=True)
class Segment:
    """One row of `segments`, joined with its `text`.

    Kaldi order is `<utt_id> <reco_id> <start> <end>` -- the second field is
    the *recording* id, not a speaker id. utt2spk is degenerate in both
    releases (utt_id == spk_id for 100% of rows), so no speaker information
    ships with the corpus.
    """

    utt: str
    reco: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def nwords(self) -> int:
        return len(self.text.split())


def read_text(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in SPLITS:
        p = root / split / "text"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
            elif line.strip():  # utterance with empty transcript
                out.setdefault(line.strip(), "")
    return out


def read_segments(root: Path) -> list[Segment]:
    text = read_text(root)
    seen: set[str] = set()
    segs: list[Segment] = []
    for split in SPLITS:
        p = root / split / "segments"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) != 4 or f[0] in seen:
                continue
            seen.add(f[0])
            segs.append(Segment(f[0], f[1], float(f[2]), float(f[3]), text.get(f[0], "")))
    segs.sort(key=lambda s: (s.reco, s.start))
    return segs


def by_recording(segs: list[Segment]) -> dict[str, list[Segment]]:
    out: dict[str, list[Segment]] = {}
    for s in segs:
        out.setdefault(s.reco, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: (s.start, s.end))
    return out
