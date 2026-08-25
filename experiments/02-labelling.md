# 02 — Are the pipeline's labels right? ❌ No

**Question.** The pipeline derives labels from call structure. Does a Tamil
speaker agree with them?

**Why it matters.** Every number in this project rests on that one claim. It is
cheap to state and nobody should believe it without evidence.

**Setup.** 197 clips, stratified by `(source, label)`, blind protocol — no
label, no metadata, audio only. One listen each.

---

## Result

| class | grounded in | agreement |
|---|---|---|
| `change` (positive) | the other person took the floor | **95.9%** |
| `hold_intra` (negative) | a pause fell inside a transcript segment | **44.4%** |
| overall | | 70.1% |

**44.4% is below chance.** The negative class was worse than a coin.

## Why

"A pause fell inside a transcript segment" is **not a judgement about
completeness**. It is a judgement about where a human transcriber pressed
enter. The positive class works because it is grounded in *recorded behaviour* —
the other speaker heard the turn end and took the floor. The negative class had
no such witness.

## ❌ What was tried before giving up on it

Structural, textual and metadata fixes — all of them, combined:

| | |
|---|---|
| every feature the pipeline computes | **AUC 0.637** |

Not rescuable by feature engineering. What was left is something that
**listens** → [03 — relabelling](03-relabelling.md).

## ⚠️ The confound this exposed

`prev_dur` separated the classes at **d = −0.33**. `hold_intra` required a pause
*inside* a segment, so the negative class was drawn from longer utterances by
construction — and it leaked into the audio as voiced fraction (d = −0.46, the
largest acoustic effect measured).

The gate was deciding the label, so the gate's bias rode along with it. Fixed
only by relabelling, not by reweighting.

## The question being asked — Q1, not Q2

Two defensible questions diverge on *"finished a sentence, then started
another"*:

| | question | who asks it |
|---|---|---|
| **Q1** | is this **syntactically finished**? | the 197 human labels, our prompt, Smart Turn's data |
| **Q2** | did the speaker **intend to continue**? | a pause/gap oracle, OpenETD, VAP |

**Our old negatives answered Q2.** OpenETD labels real data exactly as this
pipeline did and got **94.0%** human agreement on its Pause class where this got
44.4% — because they asked Q2 and scored against Q2.

**Re-wording to Q2 was ruled out.** Fine-tuning Smart Turn requires Q1; mixing a
Q2 corpus into a Q1-pretrained model teaches two different questions. At $5.69,
relabelling was cheaper than the ambiguity.

## ⚠️ The 197 labels are not truth

One human, one listen, one isolated 8 s clip, no future audio. **Three were
wrong** — every one of seven models contradicted them in the same direction;
re-listened 2026-08-24, the models were right.

`reports/qa/sheet.tsv` holds the corrected verdicts, and every downstream number
is re-derived from that file rather than hardcoded.

---

`reports/model_bakeoff.md`, `docs/archive/PHASE1A_RESULTS.md` §7 ·
`python pipeline/10_qa_sample.py`
