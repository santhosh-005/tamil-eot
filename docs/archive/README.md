# Archive

Superseded documents, kept because each records *why* a decision was made and
several are cited from code comments. Current state is
[`experiments/`](../../experiments/README.md) and
[`docs/dataset.md`](../dataset.md).

**Numbers here are pre-correction and left as written.** Where they disagree
with `experiments/`, `experiments/` is right.

| file | what it is | cited by |
|---|---|---|
| `DATASET_AUDIT_SPRING_INX_R1.md` | audit of the 116 stereo calls, and why R1 | `pipeline/01_manifest.py`, `docs/dataset.md` |
| `DATASET_AUDIT_SPRING_INX_R2.md` | audit of R2's 403 conversational recordings — why not R2 | `src/tamileot/kaldi.py`, `src/tamileot/paths.py` |
| `PHASE1A_RESULTS.md` | build report for the original gold set. Self-flagged superseded in its own §9 — the negative class described here was withdrawn | `pipeline/07_relax_funnel.py`, `experiments/02-labelling.md` |
| `RESEARCH_LABELLING.md` | how EOT training data gets labelled anywhere, why no Dravidian language had a model, and the licence position on LLM-assisted labelling | — |

Removed as duplicative once `experiments/` and `docs/pitfalls.md` existed:
`HANDOFF_PHASE1.md`, `IMPL_PHASE1_DATA.md`, `RESULT1.md`. Their two pieces of
unique content — the Q1/Q2 label-semantics decision and the `transformers` 5.x
export traps — moved to `experiments/02-labelling.md` and `docs/pitfalls.md`.
