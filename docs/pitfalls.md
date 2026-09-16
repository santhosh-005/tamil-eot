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

**`quantize_dynamic` quantises Conv by default, and that breaks the model.**
It rewrites the two encoder convs as `ConvInteger`, for which onnxruntime's CPU
provider has no kernel before 1.24 — `NOT_IMPLEMENTED : ConvInteger(10)`. The
package pinned `onnxruntime>=1.16`, so the published default weights could not
be loaded at all on most of the range it claimed to support, and the failure is
at session construction, nowhere near the quantisation step that caused it.

Upstream's own int8 build is the tell: it carries `Conv: 2` and zero
`ConvInteger`. Pass `op_types_to_quantize=["MatMul"]`. Costs +1.6 MB and leaves
two layers in float32; max output deviation from fp32 is 0.0035.

**Diff the op types against a known-good model.** `collections.Counter(n.op_type
for n in onnx.load(p).graph.node)` on ours versus upstream's showed the whole
bug in one line — 2 `ConvInteger` where upstream had 2 float `Conv`. Much faster
than reading a NOT_IMPLEMENTED trace.

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

## Publishing

**Never leave results in the scratchpad.** Two hours of evaluation runs --
outputs, metrics, the venv, a patched dependency clone -- were wiped when the
session-scoped scratchpad was cleared, and none of it could be re-derived from
disk. Same lesson as `/content` is ephemeral: write anything expensive into
`reports/` as it is produced, not at the end.

**Set `HF_HUB_DISABLE_XET=1` for large uploads.** `upload_folder` on the xet
chunked path ran **6 hours and ~13 GB of egress for a 646 MB payload and
committed nothing**. The same files over classic LFS took 4.5 minutes at
~2.3 MB/s. Nothing errored; the repo simply stayed empty while bytes moved.

**The tell is cumulative tx exceeding the payload size.** A slow uplink and a
retry loop look identical from the outside -- both show steady throughput and
an empty repo. Only the total transferred separates them. Check it before
concluding "slow network".

**Never background an upload with buffered stdout.** Redirected Python stdout
is block-buffered, so the task output file sat at 0 bytes for the whole 6 hours
and the progress bars never surfaced. Use `python -u`, and upload file by file
so each commit lands independently and partial progress survives a kill.

**Verify the push, do not assume it.** Compare the repo's LFS `sha256` against
the local file, then `load_dataset` with a clean `HF_HOME` -- a cached load
will succeed against a half-uploaded repo and tell you nothing.

## Pipeline

**A merge that rebuilds from the source list loses everything merged before
it.** The turn builder absorbs backchannel runs by joining the runs either side
of them. Written as `merged[-1] = out[i-1] + out[i+1]` it is correct for one
absorption and wrong for two in a row — `[A, b1, B, b2, C]` absorbs twice in a
single pass, and the second rebuild drops `A`. It deleted **4,572 spans** —
3,317 of them too long to be backchannels, **169 minutes of real speech**,
including one continuous 10.37 s utterance. Mean turn duration moved 0.9 s and
total audio 4 h, and nothing raised. Extend the accumulator
(`merged[-1] = merged[-1] + out[i+1]`), never re-derive it.

**Assert a partition, not a spot-check.** That bug was invisible to every
per-row check — durations, ordering, span containment all passed, because the
surviving rows were individually well-formed. It was caught in one line by
asking whether *every input span* lands in exactly one output row. When a step
regroups data rather than filtering it, totality is the check that finds the
bugs the plausible-looking output hides.

**"Inside the interval" is not "belongs to".** Two speakers genuinely overlap,
so a whole other-side turn can sit inside this one's time range without having
been folded into it. Conflating the two put 641 non-backchannels into the
absorbed count. Membership has to be decided by onset and by what no other row
claims — and the onset bound has to be inclusive, because both legs can open on
the same VAD frame (one R1 call does: two people saying hello at once).

**Dedup across a span edge only when the spans touch.** Joining a run's
apportioned text drops a word repeated at a span boundary, because a word
straddling two VAD spans is rounded into both. Applied unconditionally it also
deletes genuine repetition: `ta_IN_10102885_20230404_L_0078` is "Hello", a
**2.75 s pause**, then "Hello" again — two different transcript segments —
rendered as one "Hello". Gate the dedup on the gap being under ~50 ms.

**A row can contradict itself across rounding.** `clean` was decided on an
unrounded `word_rate` while the row shipped the value rounded to 2 dp, so a
turn at 0.9973 was excluded by a gate while displaying `word_rate: 1.0`. Decide
gates on the number that ships, not the one before rounding.

**A long pause is an end-of-turn, not a hesitation.** A 42.8 s turn whose ten
pauses were `[8.45, 0.67, 0.64, 0.38, ...]` shipped that 8.45 s silence as a
mid-turn hesitation. It is the opposite: the speaker finished, nobody answered,
and they resumed -- the transcriber closed a segment at 310.95 and opened the
next at 319.02. `LAPSE_S` is 2.0 s, matching `rules.NEG_MAX_PAUSE`.

**Split, do not drop.** Both defects found by ear could have been filtered out
of `clean`. Splitting the row instead turns one bad turn into two good ones
*plus* an extra end-of-turn example: the clean count went **up**, 4,277 to
4,774, while the gates got stricter.

**A gate that only inspects pauses is blind to a turn with no pauses.**
`ta_IN_10102663_20230123_R_0057` is 1.63 s cut off mid-sentence, sitting wholly
inside the other leg's `[613.60, 620.58]` -- the speaker tried to take the
floor, failed, gave up. `pause_crosstalk` was 0.0 because there were no pauses,
so every pause-based check passed it. Measure crosstalk across the whole turn,
not only its gaps -- and exclude absorbed backchannels first, or 14.9% of clean
rows fail for a benign "mm".

**Listen before believing a structural metric.** Every check passed on a table
where **31.6% of rows opened mid-word** and **half the rows with a pause had
the other speaker talking through it**. Totality, ordering and containment
cannot see either one — the rows are well-formed, they are just not turns. Both
were found in a 10-clip spot-listen, and only then became measurable.

**Never re-run steps 03 or 05 casually.** They re-derive the sample set, which
re-draws the 197 QA clips, every LLM verdict keyed by `sid`, the split, and
4.7 GB of cut audio. Step 04 exists so text can be fixed without that.

**`/content` is ephemeral.** Cost two Colab checkpoints — an LR-sweep winner and
a whole base run. The notebook copies to Drive on every improvement, and
`pipeline/15_export_ckpt.py` exports on CPU so the last cell of a notebook is
never the only path to a shippable artefact.

**dev → test transfer is poor.** A dev sweep predicted +1.05; test delivered
−0.22. ±2.3 points of sampling error on n≈1,300. **A 1-point dev win is noise.**
