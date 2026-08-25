# Pitfalls

Things that bit this project, each once, each expensive. Kept because most of
them produce **confident, plausible, wrong** numbers rather than an error.

---

## Measurement

**Never read one metric.** tiny int8 static has the best FP/N in the whole
quantisation table (4.94%) and is the second-worst model.

**FP/N ≠ FP/(FP+TN).** They differ by ~3×. Smart Turn publishes the first, most
notebooks print the second. Mixing them makes a competitive model look broken.

**A latency number needs thread count and machine load attached.** Three wrong
figures shipped here before that rule; one produced a wrong shipping
recommendation (*base is too slow for realtime* — it is not).

**Seed is not enough.** SDPA's backward pass uses atomics. Three base runs at
identical config and `SEED=0`: 86.23 / 85.63 / 85.36. **Treat < 1 point as
noise.**

**Never pick a threshold on test.** base's dev-argmax lost 1.13 points; the test
sweep said 0.65, dev said 0.24.

**ONNX Runtime is reproducible to ~±0.3%.** Different GEMM kernels by batch
size → different float summation order → clips on the threshold flip. Compare
against a baseline computed *in the same session*.

**A live-only number is unfalsifiable.** Score the same rows the known-good way
in the same run. Two harness bugs were caught exactly this way; both looked like
model failures. **Never report a new measurement against a figure remembered
from a doc.**

## Labels

> **`samples.jsonl` and `samples_llm_all.jsonl` both have a `label` field and
> they agree on 62.3% of test.** The first is rule-derived; the second is the
> label the model was trained and evaluated on. Score against the wrong one and
> `tiny` reads **58% instead of 84%** — plausible enough to publish.
>
> Every scorer loads `samples_llm_all.jsonl` filtered to `llm_ok`.

**The 197 human labels are one listen to an isolated clip, not truth.** Three
were wrong. `pipeline/12_bakeoff.py` re-derives every downstream number from
`reports/qa/sheet.tsv` when one changes.

**A subset accuracy without its own base rate is unreadable.** The `disputed`
bucket is 93.9% single-class and covers 38% of test — both shipped models score
*above* their overall accuracy there and *below* its majority baseline.

## Models

**ONNX output named `logits` is already a sigmoid.** Applying another maps
[0,1] → [0.5, 0.73] and every clip predicts `complete`.

**Whisper's encoder is built for 30 s, this is 8 s.** The positional table has
1500 rows; 8 s of mel gives 400. Set `config.max_source_positions = 400`
*before* constructing `WhisperEncoder`, and load with
`ignore_mismatched_sizes=True`.

**The truncated position table is not refilled with sinusoids on
`transformers` 5.x** — it is left random, max abs diff 1.09 from pretrained.
Copy the first 400 rows explicitly and assert equality afterwards.

**`transformers` 5.x breaks upstream's model class.** `from_pretrained` calls
`mark_tied_weights_as_initialized`, which needs state only `self.post_init()`
sets — and upstream's `__init__` never calls it. Add it *before* their manual
head init so `std=0.1` still wins. Upstream pins 4.48.2, so they never hit this.

**`torch.onnx.export` needs `dynamo=False`**, or it pulls in `onnxscript`. The
legacy path gives 32.1 MB, which matches the released fp32 checkpoint — a free
architecture check.

**Features must match `WhisperFeatureExtractor(chunk_length=8)` with
`do_normalize=True`** — not the Whisper default. Getting it wrong produces
plausible but meaningless output.

**int8 *static* costs −4 points at tiny and −12.76 at base** (AUC 0.922 →
0.792). Dynamic is the free one.

## Serving

**LiveKit VAD `min_silence_duration` is a floor of 0.25, not a ceiling of 0.2.**
It fails loudly at session start.

**The silent kill is `supports_language()` returning False.** If STT reports a
code the model does not claim, the detector is skipped and the agent quietly
falls back to fixed VAD timing — the exact behaviour this project exists to
replace.

**Adapter latency ≈ 120 ms, not the 83 ms in the tables.** That figure is
model-only; live adds mel (~12 ms) and the thread handoff. Different spans, both
right — never mix them.

**A control arm has to sit on the same commit path.** `turn_detection="vad"`
waits on STT; the detector arm gets a pre-landed transcript on half its turns.
Comparing them measures LiveKit's plumbing, not the model.

**`e2e_latency` only exists for turns that got a reply.** A healthy 3.3 s median
sat on top of a call where 11 of 19 turns waited > 10 s and some got nothing.
Always read it next to the reply count.

**`agent_false_interruption` reads 0 more often than it should.** It only fires
when the agent actually started talking. The commoner damage — turn committed
mid-sentence, utterance arriving in five pieces — is invisible to it.
Fragmentation per utterance is the working proxy.

## Pipeline

**Never re-run steps 03 or 05 casually.** They re-derive the sample set, which
re-draws the 197 QA clips, every LLM verdict keyed by `sid`, the split, and
4.7 GB of cut audio. Step 04 exists so text can be fixed without that.

**`/content` is ephemeral.** Cost two Colab checkpoints — an LR-sweep winner and
a whole base run. The notebook copies to Drive on every improvement, and
`pipeline/15_export_ckpt.py` exports on CPU so the last cell of a notebook is
never the only path to a shippable artefact.

**dev → test transfer is poor.** A dev sweep predicted +1.05; test delivered
−0.22. ±2.3 points of sampling error on n≈1,300. **A 1-point dev win is noise.**
