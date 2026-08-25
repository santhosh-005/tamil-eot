# Experiments

Everything measured, including what failed. One benchmark throughout: the
sealed test split, **4,168 clips from 30 calls**, byte-identical since day one.

| | meaning |
|---|---|
| ✅ | shipped — in the released artefact |
| ❌ | measured, negative, not shipped |
| ⚠️ | conclusion later reversed, or true only under stated conditions |
| 🔍 | open — recorded as headroom, not attempted |

`reports/` holds the machine-written outputs these docs cite. Regenerate any of
them by re-running the step named at the bottom of the file.

---

## The result

**70.30% → 86.13%.** Zero-shot Smart Turn v3.2 → whisper-base fine-tune, int8.

| doc | question | verdict |
|---|---|---|
| [01 — baseline](01-baseline.md) | what does released Smart Turn do on Tamil? | 70.30%, ROC-AUC 0.751 |
| [02 — labelling](02-labelling.md) | are the pipeline's labels right? | ❌ no — 44.4% on the negative class |
| [03 — relabelling](03-relabelling.md) | can an audio LLM do better? | ✅ 97.5% vs human, $5.69 |
| [04 — training](04-training.md) | what moves the number? | ✅ capacity only, +2.88 |
| [05 — data harvest](05-data-harvest.md) | is there more data in the corpus? | ⚠️ +2,908 real, +3,074 marginal |
| [06 — distillation](06-distillation.md) | can tiny inherit base's accuracy? | ❌ +0.41, inside noise |
| [07 — thresholds](07-thresholds.md) | is 0.5 the right cut? | ⚠️ no, but tuning it lost 1.13 pts |
| [08 — quantisation](08-quantisation.md) | can it be smaller and faster? | ✅ int8 dynamic free, static ❌ |
| [09 — live path](09-live-path.md) | does it survive real serving? | ✅ runs end-to-end; −2.60 pts, 92.2% coverage |

## Levers, ranked

Every training lever tried, on `dev` AUC or test accuracy as noted.

| lever | gain | |
|---|---|---|
| whisper-tiny → **whisper-base** | **+2.88 acc** | ✅ the only one that moved |
| distillation base → tiny | +0.41 acc | ❌ inside the 0.87 spread |
| funnel relaxation, +20% rows | +0.03 acc | ⚠️ marginal |
| learning-rate retune | +0.003 AUC | ⚠️ reversed once data grew |
| 50% → 100% of data (at tiny) | +0.001 AUC | ⚠️ flat — capacity-bound |
| undisputed-only training | **−5.11 acc** | ❌ |
| class rebalancing | — | ❌ no problem existed |
| threshold tuned on dev (base) | **−1.13 acc** | ❌ did not transfer |
| int8 **static** quantisation | **−12.76 acc** at base | ❌ destroys the ranking |
| int8 **dynamic** quantisation | −0.10 to +0.36 | ✅ free |

**Treat anything under ~1 point as noise.** Three base runs at identical config
and `SEED=0` scored 86.23 / 85.63 / 85.36 — SDPA's backward pass uses atomics,
so the seed does not pin the trajectory. [04 — training](04-training.md) §spread.

## Still open 🔍

| | |
|---|---|
| **Regularisation** | base hits 98.99% train accuracy. SpecAugment, dropout > 0.1, earlier stopping. Untried, one session. |
| **More data** | 722 recordings ≈ 208 h unused. At base capacity the last doubling was still worth +2.02 on `hold_intra`. Collection closed deliberately. |
| **Label-noise ceiling** | 97.1%. Above that you are fitting labeller error — 11 points of headroom remain. |
| **Short buffers** | mechanism real, effect not established. Measure offline by truncating test clips to 0.5/1/2/4 s. [09 — live path](09-live-path.md). |
