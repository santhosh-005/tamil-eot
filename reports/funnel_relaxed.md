# Funnel relaxation — what the rejected boundaries were actually worth

`03_gold.py` found **50,532** boundaries in the 116 calls and kept **16,216**
(32%). This is the second pass over the other 34,316.

## Result: 3,074 new clips, not the 8–12k first estimated

The first estimate was wrong, and the reason is worth recording. Lowering
`min_prev_dur` 1.00 → 0.50 releases 8,726 boundaries — but `drop_straddle` and
`drop_bargein` sit **after** that gate and catch 4,172 of them:

| | first pass | relaxed | Δ |
|---|---|---|---|
| `drop_short_prev` | 14,902 | 6,176 | −8,726 |
| `drop_straddle` | 6,174 | 10,100 | **+3,926** |
| `drop_bargein` | 311 | 557 | **+246** |

A boundary released by one gate is not a sample; it still has to survive the
rest. Estimating a gate's yield without running the gates behind it
overcounts, here by 3×.

## What was harvested

```
relaxed pass produced 20,283 samples
  already in samples.jsonl  16,216   untouched
  strict-gate drift            993   text refresh, excluded
  NEW from relaxation        3,074   <=
```

| source | n | gate label |
|---|---|---|
| `change` | 1,083 | complete |
| `change_midseg` | 290 | complete |
| `hold_intra` | 1,309 | incomplete |
| `hold_inter` | 392 | incomplete |
| **complete** | **1,373** | |
| **incomplete** | **1,701** | |

Splits, assigned from the recorded `data/manifest/split.json` by call:
train 1,987 / dev 285 / test 802. 115 of 116 calls contributed.

**All 16,216 recorded sids reproduce under the relaxed gates** (0 missing), so
the relaxation is a strict superset and the two sets stay comparable.

### The 993 excluded as drift

`04_refresh_text.py` re-apportioned words *after* `samples.jsonl` was written,
so the current strict gates keep 993 clips the recorded file does not have —
`keep_pos` is 4,461 under today's code against 3,901 on disk. Those are a
code/data skew, not a relaxation result, and are excluded so the 3,074 figure
means only what it claims. `reports/funnel.json` was stale for the same reason and has been removed;
totals still reconcile to 11,654 `cand_change`.

## What was moved, and what was not

| gate | | why |
|---|---|---|
| `min_prev_dur` | 1.00 → **0.50** | The gate wanted "enough speech to read prosody", but `prev_dur` is the final talk-spurt — the clip is still 8 s of context regardless. The real reason for 1.00 was the `prev_dur` confound, and relabelling by listening already severed that (d −0.33 → −0.09). |
| `neg_max_pause` | 2.00 → **5.00** | A speaker who pauses 3 s and resumes is the hardest `incomplete` there is. The gate dropped all 377 because a long pause *might* mean they finished — a label question, which now has a labeller. Capped at 5 s; the tail runs to 80 s and an 80 s pause is a new turn. |

Not moved:

| pool | n | why it stays rejected |
|---|---|---|
| `drop_neg_pause_short` | 6,152 | **Geometry leak.** The speaker who paused is the one who resumes, on the same leg, so a pause shorter than `TRAIL_S` puts their resumed speech *inside* the trailing window and hands the model the answer. Recovering these needs `TRAIL_S` changed for **both** classes and every one of the 16,216 existing clips re-cut — which would also move the test set and invalidate every prior number. |
| `drop_straddle` | 10,100 | Genuine collision; `gap` is measured to a later span and means nothing. |
| `drop_bargein` | 557 | Same. |
| `drop_pos_backchannel` | 4,516 | Would yield ~4.5k more **complete**. The set is already 65% complete; wrong class. |
| `drop_pos_gap_long` | 2,417 | Also complete. Same reason. |

A floor below 0.50 was swept and rejected — 0.30 yields +938 more but admits
~1-syllable VAD fragments as speech offsets, and the class ratio barely moves:

| floor | new | complete | incomplete | ratio after |
|---|---|---|---|---|
| 0.50 | 3,074 | 1,373 | 1,701 | 1.60:1 |
| 0.40 | 3,603 | 1,650 | 1,953 | 1.58:1 |
| 0.30 | 4,012 | 1,860 | 2,152 | 1.57:1 |
| 0.20 | 4,218 | 1,933 | 2,285 | 1.55:1 |

## Geometry check

3,074 clips cut, 3,024 at the full 8 s (98%), median 8.00 s. Peak amplitude in
the final 200 ms: **median 0.0021, p95 0.0225** — the trailing window is
silence, so trailing-silence length still carries no label information.

Clips on disk reconcile exactly: 11,995 + 2,325 + 4,970 = **19,290** =
16,216 + 3,074. No sid collides with an existing clip.

## What this is worth — honestly

> **Outcome: +0.03 points, not the +0.5 to +1.2 predicted below.** Test went
> 83.54% → 83.57% (AUC 0.898 → 0.901) on the frozen split. The prediction below
> assumed the +4.15/doubling slope held, and at whisper-tiny's capacity it did
> not — tiny had already saturated. The same rows were worth more once the
> encoder changed. Kept as written so the miss is legible.

+3,074 clips is **+19%** on 16,216, and only +2,272 of it is train+dev.

The learning curve measured one doubling of training data (50% → 100%) as
**+4.15 accuracy points**. +22.7% train data is 0.30 doublings, so the expected
gain is **+0.5 to +1.2 points** — 83.54% → roughly **84–84.7%**, and less than
that if the curve's concavity at the 100% end is real.

**This does not reach 90%.** Closing 83.5 → 90 needs (90 − 83.5) / 4.15 ≈ 1.6
doublings ≈ **3× the training data**, about 20,000 more train rows. The funnel
had 2,272 of them. The remaining 722 recordings are still the only path to the
rest.

The funnel is still worth doing: it is the cheapest data that exists — no
diarization, no ASR, no new audio, ~₹300 of labelling.

## Labelled — and the balance hypothesis is refuted

2,272 train+dev clips through Vertex AI batch, `gemini-3.7-flash`, prompt
`ca216b25` (identical to the original run, so the labels are comparable).
1,987 + 285 clips, **0 failures**, **$0.81 / ₹71**.

The relaxation was proposed to fix a 1.83:1 complete:incomplete imbalance. It
did not. It made it marginally **worse**:

| | complete | incomplete | ratio |
|---|---|---|---|
| gate labels promised | 989 | 1,283 | — |
| **what the labeller heard** | **1,576** | **696** | — |
| existing set | 10,493 | 5,720 | 1.83:1 |
| **combined** | **12,069** | **6,416** | **1.88:1** |

Because the "incomplete" pools are mostly not incomplete:

| source | n | gate said complete | labeller said complete | flipped |
|---|---|---|---|---|
| `change` | 759 | 100% | 83% | 131 (17%) |
| `hold_intra` | 998 | 0% | **56%** | 562 (56%) |
| `change_midseg` | 230 | 100% | 80% | 45 (20%) |
| `hold_inter` | 285 | 0% | **71%** | 201 (71%) |

A short final talk-spurt followed by a pause is usually a **short complete
answer** — "ஆமா", "சரி", a number — not somebody pausing mid-thought. Lowering
`min_prev_dur` therefore harvests positives, not negatives. The 56%
`hold_intra` flip rate reproduces the 53% measured on the original set, so this
is a stable property of the class and not a labeller artifact.

`change` flipped 17.3% against 7.4% on the original set: a floor change after
only 0.5–1.0 s of speech is a materially less reliable positive than one after
a full utterance. Both labels are on every row, so `--label-policy core-safe`
still reverses it.

**Conclusion: the funnel bought data, not balance.** Which is fine — the
balance was never the problem. `BCEWithLogitsLoss` already carries a per-batch
`pos_weight` that equalises the two classes' contribution exactly, and
`hold_intra`, the bucket that decides the score, was already 1.16:1.

## The confound did not come back

`min_prev_dur` was the gate holding the `prev_dur` confound shut, so the real
test was whether real labels re-open it. They do not:

| set | n | `prev_dur` d | `gap` d |
|---|---|---|---|
| existing | 16,213 | −0.11 | +0.44 |
| new | 2,272 | **+0.17** | +0.24 |
| **combined** | **18,485** | **−0.11** | +0.36 |

The new rows' `prev_dur` effect has the **opposite sign**, so they dilute the
confound rather than reinforce it. Complete-rate by provenance is 64.7% core
against 69.4% relaxed — 4.7 points, not enough for `harvest` to act as a
shortcut even if it were available at inference, which it is not.

## Result: `data/gold/samples_llm_all.jsonl`

18,488 rows, 18,485 labelled. Checked on write: no sid overlap with the
existing set, no duplicate sids, and no call appearing in two splits.

| split | n | complete | incomplete | vs before |
|---|---|---|---|---|
| train | 11,992 | 7,908 | 4,084 | **+19.9%** |
| dev | 2,325 | 1,532 | 793 | +14.0% |
| test | 4,168 | 2,629 | 1,539 | **unchanged** |

The test set is byte-identical to the one every prior number was measured on,
so a retrain is directly comparable.

## Recommended: do not label the 802 test rows

Labelling train+dev only (**2,272 clips**) is both cheaper and methodologically
better. The 4,168-clip test set stays frozen, so any improvement is measured on
the unchanged benchmark and is unambiguous. The 802 test clips are cut to disk
but are in no index; nothing reads them unless asked.

## Reproduce

```
python pipeline/07_relax_funnel.py --dry-run     # report, write nothing
python pipeline/07_relax_funnel.py               # -> data/gold/samples_relaxed.jsonl
python pipeline/08_cut_relaxed.py                # -> data/clips/{train,dev,test}
python pipeline/11_label.py --rows samples_relaxed.jsonl --source all \
                               --split train --model gemini-3.7-flash
```

`samples.jsonl`, `samples_llm.jsonl` and `split.json` are never written by any
of this, on purpose: `03`/`04` re-derive the split, and re-running them would
invalidate the 197 human labels and every cached verdict keyed by `sid`.
