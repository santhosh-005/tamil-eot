"""The labelling question, and the human labels it is scored against.

The pipeline's `hold_intra` class agreed with a human listener 44.4% of the
time -- below chance -- because "a pause fell inside a transcript segment" is
not a judgement about completeness. Structural, textual and metadata fixes were
all tried and all failed (AUC 0.637 from every feature the pipeline computes,
combined). What replaced it is something that listens.

This module holds the parts that must not drift: the prompt, its identity hash,
and the readers for the 197 blind-listened clips that every labeller is scored
against. `pipeline/11_label.py` runs the question at scale;
`pipeline/12_bakeoff.py` rescores the comparison offline. Both import from
here rather than carrying their own copy.

The cache is keyed on `PROMPT_ID`, so editing `PROMPT` starts a clean run
instead of silently mixing verdicts from two different questions.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .paths import GOLD, LABELS, REPORTS

# Audio only, never the transcript: the shipped transcripts carry substantial
# word errors and mis-cut boundaries, and the pipeline's own label is withheld
# so the model cannot simply agree with it.
PROMPT = """You are an expert Speech Science and Spoken Tamil Dialogue System evaluator. Your task is to analyze the provided Tamil audio clip (and its optional transcript) to classify its End-of-Turn (EOT) status for a Voice AI system.

TASK DEFINITION:
- "complete": The speaker has fully finished their conversational turn and yielded the floor. It is safe for the Voice AI to take over and speak a full response without interrupting.
- "incomplete": The speaker is mid-thought, setting context, holding the floor, or using a dependent linguistic marker. The Voice AI must NOT interrupt.

CRITICAL TAMIL CLASSIFICATION RULES:

1. MARK AS "incomplete" IF:
   - Dependent Suffixes: Verb ends in conditional/concessive/temporal suffixes like "-aalum" (e.g., irundhaalum = "even though..."), "-na/-nna" (e.g., kondu vandheengana = "if you bring..."), or "-bodhu" (e.g., sollum-bodhu, irukkum-bodhu = "when/while...").
   - Topic/Context Setters: Ends on clitics used to set context or elicit backchannels like "-ena", "-aana", "-la" (e.g., vandhena..., saami kumbiduvaangalla...).
   - Verbal Participles & Connectors: Ends on a participle like "paathu" (choosing/looking for...), "poittu...", or dangling connectors like "adhanala..." (so...), "munnaadi..." (before...), "aprom..." (after...).
   - Hypothetical Setters: Ends on premise setters like "vachukonga..." (suppose...).
   - Speech Repairs & Disfluencies: Speaker self-corrects mid-speech ("sorry ma'am, 30 illai..."), or pre-announces dictation cut off before digits ("Visa number...").
   - Open-Ended Lists: Itemized lists starting with "Adhukkappram..." lacking a wrap-up verb ("kudunga") or particle ("avvalavuthaan").

2. MARK AS "complete" IF:
   - Terminal Finite Verbs: Sentence ends on a finite verb (e.g., sonna, thara mudiyum, check pannikkiren, potturunga, call panren) accompanied by falling pitch.
   - Direct Questions & Question Tags: Direct questions or tags explicitly expecting an answer (e.g., "vaayppu irukkaa?", "pottaanaa?", "undaa?", "theriyumaa?").
   - Colloquial Idiomatic Closers: Self-contained conversational wrap-up idioms (e.g., "neenga vera!", "naanga kaali!").
   - Nominal Predicates & Elliptical Answers: Finished nominal statements ("Property-na kammi thaan"), standalone affirmations ("Aamaa"), or direct short answers to a previous question ("Ondrai vayasu kuzhandhaikki").

ACOUSTIC & PROSODIC GUIDANCE:
- Analyze the pitch contour at the final 300-500ms of the audio: Falling pitch = complete; Flat/Rising continuation tone or trailing breath = incomplete.
- Prioritize Tamil discourse grammar and acoustic pitch over silent pause length.

Reply with JSON only, no markdown:
{"verdict": "complete" or "incomplete", "confidence": 0.0 to 1.0, "reason": "under 8 words"}"""

PROMPT_ID = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]


def human_labels() -> dict[str, int]:
    """sid -> 1 complete / 0 incomplete, from the blind listening pass.

    `answers.tsv` carries the clip order and its provenance; `sheet.tsv` carries
    the verdicts, edited by hand. Kept apart so a re-listen changes one file and
    every downstream number is re-derived from it.
    """
    ans = {int(r["idx"]): r for r in
           csv.DictReader((REPORTS / "qa/answers.tsv").open(encoding="utf-8"), delimiter="\t")}
    sheet = {int(r["idx"]): (r["verdict"] or "").strip().lower() for r in
             csv.DictReader((REPORTS / "qa/sheet.tsv").open(encoding="utf-8"), delimiter="\t")}
    out = {}
    for i, a in ans.items():
        v = sheet.get(i)
        if v in ("complete", "incomplete"):
            sid = f"{a['call']}_{a['side'][0]}_{int(round(float(a['t']) * 1000)):08d}"
            out[sid] = 1 if v == "complete" else 0
    return out


def qa_index() -> dict[str, int]:
    """sid -> the 1..198 clip number, so an output sheet lines up with the wavs."""
    out = {}
    for r in csv.DictReader((REPORTS / "qa/answers.tsv").open(encoding="utf-8"), delimiter="\t"):
        sid = f"{r['call']}_{r['side'][0]}_{int(round(float(r['t']) * 1000)):08d}"
        out[sid] = int(r["idx"])
    return out


def load_rows(name: str = "samples.jsonl") -> list[dict]:
    """Read one of the row files under `data/gold/`.

    Scoring against the human labels always reads `samples.jsonl`, whatever
    else is being labelled: the 197 human labels are keyed to sids in that file,
    and scoring a labeller against a set it cannot cover would report agreement
    on nothing.
    """
    return [json.loads(l) for l in read_lines(name)]


def read_lines(name: str) -> list[str]:
    """Lines of one row file: `data/gold/` first, then the shipped `labels/`.

    The shipped copies are gzipped -- 15 MB of JSONL compresses under 2, and a
    repo carrying its own label tables should not carry them at twice the size
    they need. `gzip` is stdlib, so this costs no dependency.
    """
    live = GOLD / name
    if live.exists():
        return live.read_text(encoding="utf-8").splitlines()
    packed = LABELS / (name + ".gz")
    if packed.exists():
        import gzip
        return gzip.decompress(packed.read_bytes()).decode("utf-8").splitlines()
    plain = LABELS / name
    if plain.exists():
        return plain.read_text(encoding="utf-8").splitlines()
    raise FileNotFoundError(
        f"{name} not found. Looked in {live}, {packed}, {plain}.\n"
        "Run the pipeline from step 01, or restore `labels/` from the repo."
    )


def cache_for(model: str) -> Path:
    """One verdict cache per (model, prompt).

    These used to be separate files for the validation pass and the bulk run,
    which meant the 197 validation clips got paid for twice: once to prove the
    model, once again as members of the pool. Same question, same model, same
    answer -- so it is the same cache, and a bulk run reports its agreement on
    whichever human-labelled clips it happens to cover, for free.
    """
    return GOLD / f"llm_{model}_{PROMPT_ID}.jsonl"
