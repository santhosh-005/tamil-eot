# How EOT training data actually gets labelled

Research pass answering five questions: what the standard labelling methods are,
whether human listening scales, what Smart Turn really did, why no Dravidian
language has an EOT model, and how Gemini/OpenAI handle Tamil prosody without
Tamil labels.

Every claim below is sourced or measured. Where I checked a hypothesis and it
turned out false, it says so.

---

## 1. The six methods, and who uses which

| method | label comes from | who ships it | needs |
|---|---|---|---|
| **Synthetic TTS** | text you wrote as complete or truncated | **Smart Turn** (all 23 langs) | sentence corpus + TTS |
| **Pause/Gap oracle** | did the same speaker resume, or the other one | **OpenETD** (300 h) | stereo or diarized dialogue |
| **LLM distillation** | a bigger model's judgement | **LiveKit** (7B→0.5B) | teacher that knows the language |
| **Human annotation** | a native listener | Smart Turn's eval + v3.1 gains | native speakers, time |
| **Self-supervised (VAP)** | *nothing* — predicts future voice activity | Ekstedt & Skantze, academic | stereo dialogue audio only |
| **Subtitle-derived** | subtitle segment boundaries | Thai EOT paper (YODAS) | captioned video in the language |

Two things to notice.

**Only VAP is genuinely label-free.** Everything else derives the label from
text or from behaviour. Smart Turn's labels are text-derived — a complete
sentence versus one the LLM truncated. That matters for us: text-derived labels
are not a compromise, they are the industry standard.

**Nobody hand-labels at scale.** Human annotation is used for evaluation sets
and for targeted top-ups, never for the bulk. Smart Turn's v3.1 human
contributions came from three companies whose business is collecting audio.

---

## 2. Smart Turn: your guess was right, with one correction

The v2 pipeline, verbatim from Daily's own write-up:

```
agentlans/high-quality-multilingual-sentences   (text corpus, 50 languages)
  -> Gemini 2.5 Flash             classify + discard bad sentences (50-80% thrown away)
  -> Gemini Flash                 truncate mid-stream, append filler words  = INCOMPLETE
  -> Google Chirp 3 HD TTS        synthesize both classes
```

Chirp 3 was chosen specifically because *"pronunciation of filler words is
excellent, and ending a sentence with a comma generally causes the model to use
the correct intonation."* That is exactly the mechanism you described.

`smart-turn-data-v3.2-train` is **271,000 rows across 23 languages** — about
11.8k per language — and the `synthetic` column is `true` on most of them.

**The correction.** Synthetic alone tops out lower than people assume. v3.1
added *human-recorded* English and Spanish from Liva AI, Midcentury and MundoAI:

| | v3.0 | v3.1 (8 MB) | v3.1 (32 MB) |
|---|---|---|---|
| English | 88.3% | **94.7%** | 95.6% |
| Spanish | 86.7% | **90.1%** | 91.0% |

Daily's own explanation: *"synthetic data often lacks the natural variability
and subtle cues in actual human speech."*

So the shape of the field is: **synthetic gets a language to ~88%. Human
conversational audio takes it to ~95%.** Smart Turn has 271k of the first and
is publicly asking for the second. We have 37 hours of the second.

Your objection to Bulbul-generated training data was not wrong, but it was
aimed at the wrong target. Synthetic is how every one of those 23 languages
exists at all. The right position is not "no TTS" — it is "TTS for coverage,
real calls for the last 7 points and for the eval set."

---

## 3. Why no Tamil? I tested every explanation. They are all false.

Smart Turn's 23: Arabic, Bengali, Chinese, Danish, Dutch, German, English,
Finnish, French, **Hindi**, Indonesian, Italian, Japanese, Korean, **Marathi**,
Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian,
Vietnamese. Three Indic languages, all Indo-Aryan. **Zero Dravidian.**
LiveKit's eot-bench: 14 languages, Hindi, no Tamil. Same line.

Your hypothesis was that Tamil TTS isn't good enough to run the synthetic
recipe. I checked both inputs the recipe needs.

**Hypothesis A — no Tamil TTS. False.** Chirp 3 HD — the exact model Smart Turn
used — supports `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`, plus hi/bn/mr/gu. It has
SSML, `[pause long]` markup and `speaking_rate`. All four Dravidian languages
were available the whole time.

**Hypothesis B — no Tamil sentence corpus. False, and it is not close.** I
queried the actual dataset Smart Turn built from:

```
en  59,134     vi  46,801     bn  32,800     hi  31,932
ml  26,794     tr  25,285     ta  23,863     mr  20,624
```

**Tamil has 23,863 clean sentences — more than Marathi and Turkish, both of
which shipped.** Malayalam has more than both too. Telugu and Kannada are the
only ones genuinely absent from that corpus.

So the honest answer to "is that why nobody has tried": **no.** Nothing
technical blocked it. Smart Turn is an open project where language coverage
tracks whoever shows up with data — Liva AI is a voice-data company, and the
Indic languages that landed are the ones somebody cared enough to build.

That is worth sitting with. The gap is not a research problem. It is a nobody
did it problem, and the project explicitly invites contributions.

---

## 4. Is human listening workable? No.

Measured against your own throughput — 198 clips in about an hour:

| | clips | listening hours |
|---|---|---|
| `hold_intra` alone | 9,406 | **47.5** |
| same, 3 annotators for agreement | 28,218 | **142.5** |

Forty-seven hours alone is six weeks of evenings with quality decaying the
whole way. Through an Indian annotation vendor at 3 raters it is roughly
$500–900 plus weeks of coordination, and you would still be validating their
work against your own ears.

For comparison, Gemini 3 Flash on the same clips — audio bills at 32 tokens per
second and $1.00 per million:

| | clips | cost |
|---|---|---|
| `hold_intra` | 9,406 | **$4.67** |
| + held-back holds | 11,465 | $5.69 |
| every clip in the gold set | 16,216 | **$8.04** |

Even if that estimate is off by 5×, it is under $50.

**Money was never the constraint on this project. The only question that ever
mattered is whether the labeller is right** — which is why the 197 human labels
are worth more as a validation set than they ever were as a labelling effort.
Human listening is the right tool for the eval set and useless for the pool.

---

## 5. How Gemini/OpenAI/Grok handle Tamil without Tamil EOT labels

Three parts, in order of how much they explain.

**They do not have a Tamil EOT model.** There is no Tamil turn classifier
inside Gemini Live. There is one multilingual audio-native model whose
turn-taking falls out of understanding what was said. OpenAI's semantic VAD is
described as *"a semantic classifier... based on the words they have uttered"*
— it scores the probability the user is done and stretches the timeout when
that probability is low. It is language understanding, not a per-language
prosody model.

**Scale does the rest.** Google's USM was pretrained on **12 million hours of
unlabelled YouTube audio across 300+ languages** with BEST-RQ, then fine-tuned
on captions in 73 languages averaging under 3k hours each. Nobody labelled turn
boundaries in Tamil. The model heard enough Tamil to represent it, and the LLM
on top already reads Tamil from text pretraining. LiveKit says this outright
about their own detector: the Qwen base *"is pre-trained on multilingual
corpora, which encodes knowledge of global formats and reduces the need for
language-specific datasets."*

**And the premise is wrong anyway — they absolutely do have Tamil speakers.**
India is the world's annotation hub. Scale AI, Surge, Appen and Karya all run
Tamil-language annotation. Frontier labs buy Tamil evaluation data routinely.
They just don't spend it on a dedicated EOT classifier, because they don't need
one.

**Where that leaves the project.** The claim "Tamil turn-taking needs Tamil
data" needs to be defended, not assumed. Two pieces of evidence say it holds:

- **Cross-lingual transfer measurably fails.** The multilingual VAP paper
  (LREC-COLING 2024) found a monolingual VAP model *"does not make good
  predictions when applied to other languages"*; only multilingual training
  recovers it, and the multilingual model identifies the input language at
  F1 **99.99%**. Turn-taking prediction is language-conditioned in practice.
- **Tamil breaks the main universal cue.** Final lengthening is widespread
  across languages, but the 25-language DoReCo study found it varies
  specifically in languages with a **phonological vowel length contrast** —
  and Tamil has exactly that (கல் *kal* vs கால் *kaal*). A detector leaning on
  final lengthening has a harder job in Tamil than in Hindi.

That is a real argument. It is not "Tamil is neglected", it is "the cue
generalises worse here, and the measurement exists."

---

## 6. Is Gemini the only option? No — and there is a licence problem

Yes, using Gemini to label is knowledge distillation via pseudo-labelling. It
is standard: **LiveKit's shipping turn detector was built exactly this way**,
fine-tuning Qwen2.5-7B as a teacher and distilling into the 0.5B student.

### The licence problem — read this before spending money

Gemini API Additional Terms, verbatim:

> "You may not use the Services to develop models that compete with the
> Services (e.g., Gemini API or Google AI Studio)."

> "You also may not attempt to reverse engineer, extract or replicate any
> component of the Services, including the underlying data or models."

My read: an 8M-parameter binary classifier is not plausibly a competitor to the
Gemini API. But Gemini Live *does* sell turn detection, and Google's
enforcement language covers *suspected* violation. The plan is to publish this
dataset CC BY 4.0 and put it in front of a company with lawyers. Ambiguity is a
cost here.

Two things reduce it, and both are cheap:

1. **Pay.** On the paid tier Google does not train on your prompts or
   responses; on the free tier it does. For $8 of inference this is not a
   decision.
2. **Frame the provenance accurately.** Gemini produces a binary verdict on
   audio *you own*, not content. PuIblish it as human-validated with
   LLM-assisted labelling, with the measured agreement rate against the 197
   human labels attached. That is both true and standard practice.

### The alternatives, honestly ranked

**Open-weight audio teachers do not cover Tamil.** Qwen3-Omni: Apache 2.0, 19
speech input languages, includes Urdu, **no Tamil**. Voxtral: Apache 2.0, 13
languages, includes Hindi, **no Tamil**. The open ecosystem's Indic audio
coverage stops at Hindi, same wall as everything else.

**ASR + text LLM is the real second option, and it is fully open.** Split the
job the way LiveKit originally did:

```
Tamil ASR with word timestamps   (IndicConformer 600M, or Whisper large-v3 — both open)
  -> accurate text at the cut point
  -> Tamil-capable open text LLM judges completeness
```

Worth testing, and now partly tested. §7 of `PHASE1A_RESULTS.md` ruled the
textual fix out on a reason that did not hold: the shipped transcripts are
accurate at the **word** level. The defect was our **cut** — `_apportion()`
spread words evenly over elapsed time, while the median transcript segment is
21% silence, so a fifth of the words were assigned to intervals with no speech.
That is fixed (VAD-weighted apportionment); it changed the final three words on
**38.1%** of all samples.

Re-tested on the corrected cut: **no improvement.** A leave-one-out vote on
word-final suffixes went 72.1% → 72.6%, both below the 76.1% majority baseline,
even though the last word moved on 32% of the QA clips. The cheap textual route
is now properly dead rather than merely suspected.

What is still open is the expensive version — a Tamil-capable LLM reading the
corrected text, rather than a 2-character suffix heuristic. And the expensive
half was never ASR, since the words already exist: **what is missing is
word-level timing**, i.e. forced alignment over a transcript you already have.
Validates against the same 197 labels, free.

**VAP is the wildcard.** Self-supervised, zero labels, input is a stereo
waveform — which is precisely what the 116 calls are. It cannot produce a
"complete/incomplete" label, but it can produce an independent second witness
for the negatives, orthogonal to any LLM, with no licence question at all.
Filed as a later option, not a substitute.

---

## 7. The finding that reframes our 43.4%

OpenETD's real portion is labelled the same way our pipeline is: diarize, then
any silence over 200 ms is a **Pause** if the same speaker resumes and a **Gap**
if the other one does. That is our `hold_intra` and `change` exactly.

They validated it against humans on 96 clips:

| | OpenETD (English) | ours (Tamil) |
|---|---|---|
| Pause / `hold_intra` | **94.0%** | **43.4%** |
| Gap / `change` | 76.1% | **95.9%** |

Nearly inverted. Ours is better on one class and catastrophically worse on the
other, which rules out "our pipeline is broken" as a general explanation — a
broken pipeline does not produce 95.9%.

The difference is **the question the annotator was asked.** OpenETD defines
Pause as *"the speaker is silent but intends to continue talking"* — a
judgement about intent, checkable from the recording. We asked *"was the
utterance finished?"* — a judgement about syntax. Those two answers diverge on
exactly one case: **the speaker finished a sentence and then started another
one.** OpenETD's annotator marks that Pause and agrees with the automatic
label. Our listener marks it complete and disagrees.

Our own data says that case is everywhere. I measured it: **28% of the QA
clips' transcript text contains sentence-final punctuation *internally*** — a
transcript segment routinely holds more than one sentence, which is precisely
the failure `PHASE1A_RESULTS.md` §5 predicted before the listening pass. And
agreement falling as the pause lengthens (48.5% → 42.4% → 39.4%) is the
signature of the same thing.

**So there are two defensible label semantics, and we have to pick one on
purpose:**

- **Q1 "was it syntactically finished?"** — what our 197 labels answer, and
  what Smart Turn's synthetic data means (complete sentence vs truncated one).
  Fine-tuning Smart Turn requires this. Our negatives do not currently answer it.
- **Q2 "did the speaker intend to continue?"** — what the pause/gap oracle
  answers, what OpenETD and VAP model, and what actually determines whether an
  agent should speak. Our negatives already answer this, at roughly OpenETD's
  quality.

One correction to my own earlier reasoning. §7 argued no model can separate
"finished and stopped" from "finished and continued" because the audio is
identical. **That is overstated.** A speaker who intends to continue produces a
non-final boundary tone — a continuation rise — at a syntactically complete
clause. That difference is real, is what turn-yielding-cue research measures,
and is why VAP works at all. Our listener, hearing an isolated 8-second clip
with no future audio, was in a much worse position to hear it than a model
trained on thousands of examples would be.

Which means the negative class may be less broken than the headline number
says. It is mislabelled **for Q1**. It has never been tested against Q1's
alternative.

---

## 8. What I would do

Ordered by information gained per rupee.

1. **Finish the 40–60 clip paired test** on `gemini-3-flash-preview` — 2–3 days
   of free quota, settles whether the model-quality gap is real (currently
   p = 0.250, not established).
2. **Re-run the 197-clip validation asking Q2** — *"is the speaker going to
   keep talking?"* instead of *"was that finished?"*. Same clips, same models,
   different prompt. Costs about ten cents. If agreement on `hold_intra` jumps
   from 43% toward OpenETD's 94%, the 9,406 negatives are salvageable as-is and
   the whole relabelling job disappears.
3. **Test the ASR + text-LLM teacher** against the same 197. Fully open licence,
   no Gemini dependency, and it is a fair test of the textual fix that the
   shipped transcripts were too dirty to give.
4. **Then** commit the ~$8 to whichever labeller won, on the paid tier, test
   split first.
5. **Separately, run the official Chirp 3 Tamil synthetic recipe.** Both inputs
   are confirmed available. It is a weekend and tens of dollars, it produces
   the ~12k samples that make Tamil exist in Smart Turn at all, and it is
   directly mergeable upstream — which is a public, verifiable, citable result
   with Sarvam's name nowhere near it and their attention very much on it.
   The real calls then become what they are actually scarce for: the eval set
   and the last seven accuracy points.

---

## Sources

Smart Turn — [v2 blog](https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/),
[v3 blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/),
[v3.1 blog](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/),
[repo](https://github.com/pipecat-ai/smart-turn),
[contribution guide](https://github.com/pipecat-ai/smart-turn/blob/main/docs/data_generation_contribution_guide.md),
[v3.2 train set](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
[sentence corpus](https://huggingface.co/datasets/agentlans/high-quality-multilingual-sentences).
TTS — [Chirp 3 HD](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd).
LiveKit — [turn detector v1.0](https://livekit.com/blog/solving-end-of-turn-detection),
[original transformer post](https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection),
[eot-bench](https://github.com/livekit/eot-bench).
OpenETD — [Speculative End-Turn Detector](https://arxiv.org/abs/2503.23439).
VAP — [Ekstedt & Skantze 2022](https://arxiv.org/abs/2205.09812),
[Multilingual VAP](https://aclanthology.org/2024.lrec-main.1036/).
Thai EOT — [arXiv 2510.04016](https://arxiv.org/abs/2510.04016).
Scale — [Google USM](https://arxiv.org/abs/2303.01037).
Prosody — [Final Lengthening in 25 languages](https://www.sciencedirect.com/science/article/pii/S0095447022000547).
Terms — [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms).
Open audio models — [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct),
[Voxtral](https://mistral.ai/news/voxtral/).
OpenAI — [Realtime VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad).
Sarvam — [LiveKit integration docs](https://docs.sarvam.ai/api/integration/build-voice-agent-with-live-kit).
