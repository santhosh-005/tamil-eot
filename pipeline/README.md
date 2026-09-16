# pipeline/

Corpus → dataset → model → release. Linear, numbered, each step writing a file
the next one reads. Every step is idempotent and safe to re-run except where
noted.

```bash
export SPRING_INX_R1=/path/to/SPRING_INX_Tamil_R1
pip install -r requirements.txt
python pipeline/00_fetch_models.py
```

## Steps

| # | step | reads | writes | needs |
|---|---|---|---|---|
| 00 | `fetch_models` | — | `models/*.onnx` | network |
| 01 | `manifest` | corpus | `manifest/stereo_calls.json` | corpus |
| 02 | `vad` | corpus | `vad/*.npy` | corpus, all cores |
| 03 | `gold` | vad, transcripts | `gold/samples.jsonl` | |
| 04 | `refresh_text` | `samples.jsonl` | *same file, in place* | |
| 05 | `split_extract` | `samples.jsonl` | `manifest/split.json`, `clips/` | corpus |
| 06 | `cut_heldback` | `samples.jsonl` | `clips/` | corpus |
| 07 | `relax_funnel` | `samples.jsonl`, `split.json` | `gold/samples_relaxed.jsonl` | |
| 08 | `cut_relaxed` | `samples_relaxed.jsonl` | `clips/` | corpus |
| 09 | `verify` | any row file | `reports/verify_*` | |
| 10 | `qa_sample` | `samples.jsonl` | `reports/qa/blind/` | |
| 11 | `label` | `samples*.jsonl`, clips | `gold/samples_llm*.jsonl` | GCP |
| 12 | `bakeoff` | verdict caches | `reports/model_bakeoff.md` | |
| 13 | `baseline` | `samples_llm.jsonl` | `reports/baseline_zeroshot.md` | |
| 14 | `pack_for_colab` | rows + clips | `data/colab/*.flac` | |
| — | **train** | `data/colab/` | `best.pt` | Colab T4 |
| 15 | `export_ckpt` | `best.pt` | fp32 `.onnx` | torch |
| 16 | `evaluate` | `.onnx`, test split | `reports/finetune_results_*.md` | |
| 17 | `dispute_analysis` | rows, QA verdicts | `reports/dispute_analysis.md` | |
| 18 | `operating_point` | `.onnx`, dev split | `reports/operating_point.md` | |
| 19 | `quantize` | fp32 `.onnx` | int8 `.onnx` — **what ships** | |
| 20 | `label_policy` | rows, models | `reports/label_policy.md` | |
| 21 | `pack_for_hf` | models, clips | `dist/hf/` | pyarrow |
| 22 | `replay_live` | test split, corpus | `reports/replay_live.md` | livekit |
| 23 | `build_turns` | vad, `samples_llm_all.jsonl`, `split.json` | `gold/turns.jsonl` | corpus |
| 24 | `verify_turns` | `gold/turns.jsonl` | —, **exits non-zero on failure** | corpus |
| 25 | `pack_turns_hf` | `gold/turns.jsonl`, corpus | `dist/hf/tamil-turns/` | pyarrow |

Training is step 14½: `notebooks/train_smart_turn_tamil.ipynb`, on a Colab T4.
Upload `data/colab/` to Drive, run the notebook, bring `best.pt` back.

## Order is not the numbering you would guess

Step **07** reads `split.json`, which step **05** writes. The harvest has to
know which call each boundary belongs to before it can be labelled without
touching the sealed test split — so relaxation comes *after* the split, not
next to the gold build that it relaxes.

## Two steps that must not be re-run casually

**03 and 05 re-derive the sample set.** `_apportion()` feeds both the displayed
text and `Span.words`, which the ≥3-word backchannel gate consumes — so
changing it changes *which* boundaries become samples. A full rebuild silently
re-draws the 197 QA clips, every LLM verdict keyed by `sid`, the per-call
yields the split comes from, and 4.7 GB of cut audio. Step 04 exists precisely
so the text can be fixed without that.

**The test split has never moved.** 4,168 clips from 30 calls, byte-identical
across every repack, which is why every number in `experiments/` is on one
benchmark. Anything that re-draws it invalidates all of them at once. Step 23
reads `split.json` and carries it through per *call*, so `tamil-turns` agrees
with `tamil-eot` about what "test" means; step 24 asserts the 71/15/30 call
counts still hold.

## Steps 23–25 build a separate dataset

`turns` regroups the same speech into turn-level rows — one speaker's hold of
the floor, mid-turn pauses separated from the end-of-turn. It needs no new VAD
pass and no new labels, and **cuts no audio**: every row carries
`start_time`/`end_time`/`trail_s`, so step 25 slices the leg WAVs straight into
parquet rather than materialising 4.5 GB twice.

It ships as **`santhosh-005/tamil-turns`, a separate repo**, not a second config
on `tamil-eot`. `task_categories`, `tags` and `size_categories` are repo-level on
the Hub and drive its search, so one repo cannot describe both units of analysis
— and keeping them apart leaves the artefact the paper cites frozen as published.
Step 25's docstring carries the full reasoning.

Spec and the measured yield are in `docs/turns_format.md`. **Never publish
without step 24 passing** — it caught a bug that had silently deleted 169 minutes
of real speech.

## Credentials

| step | needs | how |
|---|---|---|
| 11 | GCP project + GCS bucket | `TAMILEOT_GCP_PROJECT`, `TAMILEOT_GCS_BUCKET`, `gcloud auth application-default login` |
| 21 | HuggingFace write token | `hf auth login` |

Nothing else touches a credential. Steps 12, 13, 16–20 run entirely offline
from the local caches.
