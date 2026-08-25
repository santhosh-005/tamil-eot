# 05 — Getting more out of the corpus

50,532 boundaries found. 16,216 kept by the first pass — 32%. What is in the
other 68%?

---

## ✅ Held-back classes unlocked — +2,908 clips

Two sources were labelled and tagged but never cut to audio, because the two
witnesses disagreed:

| source | n | why held back |
|---|---|---|
| `change_midseg` | 850 | floor changed, but the transcriber had the segment still running |
| `hold_inter` | 2,058 | pause straddles a segment edge — may be a finished sentence |

Held back because the *pipeline* had to decide the label alone. A labeller
measured at 97.5% against a human is the third witness → both unlocked.

**In the shipped dataset.**

## ⚠️ Funnel relaxation — +3,074 clips, +0.03 accuracy

34,316 rejected boundaries revisited. Two gates moved, both set when the
pipeline had to self-label:

| gate | was | now |
|---|---|---|
| `min_prev_dur` | 1.00 | 0.50 |
| `neg_max_pause` | 2.00 | 5.00 |

**Yield 3,074, not the 8–12k first estimated** — `drop_straddle` sits *after* the
gate and ate 4,172.

Made the balance **worse** (1.83:1 → 1.88:1): `hold_intra` flipped 56% to
complete, `hold_inter` 71%. A short final talk-spurt plus a pause is usually a
short **complete** answer, not a mid-thought pause.

**Test gain +0.03 points.** Confound check passed (`prev_dur` d unchanged at
−0.11). Kept — it is in the shipped 18,485 — but it is not a lever.

## ❌ Undisputed-only training

Train only where the pipeline and the LLM agree.

| | accuracy |
|---|---|
| full set | 82.10% |
| undisputed only | **76.99%** |

**Losing 40% of the data costs more than the label noise does.** −5.11 points.

## ❌ Class imbalance — no problem existed

**Premise:** 1.83:1 complete:incomplete is hurting accuracy.

**Reality:**

- `BCEWithLogitsLoss` already carries per-batch `pos_weight` ≈ 0.531 → the two
  classes contribute **exactly equally**.
- `hold_intra` — 58% of the data, the bucket that decides the score — was
  already **1.16:1**.
- The 1.83:1 comes from `change` at 12:1, which is **definitional**: speaker
  changed → turn ended.

Nothing to fix. Recorded because the hypothesis was plausible enough to cost a
day.

## ❌ Gates that stay closed

| pool | n | why not |
|---|---|---|
| `drop_neg_pause_short` | 6,152 | **geometry leak.** The speaker who paused resumes on the *same leg*; a pause shorter than `TRAIL_S` puts resumed speech inside the trailing window, handing the model the answer. Fixing means re-cutting all 18k clips — which moves the test set. |
| `prev_dur` floor < 0.50 | +938 | admits ~1-syllable VAD fragments as speech offsets |

## ❌ TTS training data

Considered and dropped. Smart Turn's own 271k-row corpus *is* TTS, so a TTS
Tamil set adds nothing that is not already reproducible in an afternoon — and
it would not carry the acoustics the model has to work in.

**Real narrowband telephony is the whole point of this dataset.**

---

## Where the 18,485 came from

| | |
|---|---|
| boundaries found | 50,532 |
| first pass kept | 16,216 |
| + held-back classes | already inside the 16,216, cut later |
| + funnel relaxation | +3,074 |
| − no usable LLM verdict | −3 |
| **shipped** | **18,485** |

| split | clips | complete | incomplete |
|---|---|---|---|
| train | 11,992 | 7,908 | 4,084 |
| dev | 2,325 | 1,532 | 793 |
| **test** | **4,168** | 2,629 | 1,539 |

Split by **call**, never by clip. 722 recordings ≈ 208 h left unused —
collection closed deliberately.

---

`reports/funnel_relaxed.md`, `reports/relabelling_relaxed.md`, `reports/gold_build.txt` ·
`python pipeline/07_relax_funnel.py` → `08_cut_relaxed.py`
