"""Corpus structure: which recordings are what, and how the stereo call legs pair up.

The one family that matters for step 1a is `ta_IN_*_{Left,Right}`: 116 real
telephone conversations shipped as one audio file per speaker, each leg separately
transcribed with its own timestamps. Which file the audio came from *is* the
speaker label, so this family needs no diarization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .kaldi import Segment, by_recording, read_segments

LEFT, RIGHT = "Left", "Right"
SIDES = (LEFT, RIGHT)


def family(reco: str) -> str:
    """Coarse family label derived from the recording id."""
    if reco.startswith("ta_IN_"):
        return "IN_stereo"
    if reco.startswith("ta_CRESC_"):
        return "CRESC"
    parts = reco.split("_")
    if len(parts) > 2 and parts[1] == "SHA1P":
        return f"SHA1P_{parts[2]}"  # C, C2, C3, M
    return "other"


def is_conversational(reco: str) -> bool:
    fam = family(reco)
    return fam != "SHA1P_M" and fam != "other"


def stereo_base(reco: str) -> tuple[str, str] | None:
    """('ta_IN_10102450_20230112', 'Left') for a stereo leg, else None."""
    if not reco.startswith("ta_IN_"):
        return None
    base, _, side = reco.rpartition("_")
    return (base, side) if side in SIDES else None


@dataclass(slots=True)
class StereoCall:
    """One call, with both legs' transcript segments kept separate."""

    base: str
    legs: dict[str, list[Segment]] = field(default_factory=dict)
    wavs: dict[str, Path] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(self.legs.get(s) and self.wavs.get(s) for s in SIDES)

    @property
    def speech_dur(self) -> float:
        return sum(s.dur for side in SIDES for s in self.legs.get(side, ()))

    @property
    def span(self) -> float:
        ends = [s.end for side in SIDES for s in self.legs.get(side, ())]
        return max(ends) if ends else 0.0

    def timeline(self) -> list[tuple[Segment, str]]:
        """Both legs merged into one speaker-tagged sequence, sorted by start."""
        ev = [(s, side) for side in SIDES for s in self.legs.get(side, ())]
        ev.sort(key=lambda x: (x[0].start, x[0].end))
        return ev


def load_stereo_calls(root: Path) -> dict[str, StereoCall]:
    """Assemble every `ta_IN_` call that has both legs on disk and in `segments`."""
    recs = by_recording(read_segments(root))
    calls: dict[str, StereoCall] = {}
    for reco, segs in recs.items():
        pair = stereo_base(reco)
        if pair is None:
            continue
        base, side = pair
        call = calls.setdefault(base, StereoCall(base))
        call.legs[side] = segs
        wav = root / "Audio" / f"{reco}.wav"
        if wav.exists():
            call.wavs[side] = wav
    return {b: c for b, c in calls.items() if c.complete}


def leg_reco(base: str, side: str) -> str:
    return f"{base}_{side}"


def other(side: str) -> str:
    return RIGHT if side == LEFT else LEFT
