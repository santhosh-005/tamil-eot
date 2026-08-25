# Label tables

The labels for all 18,485 turn boundaries, plus the raw verdicts from the seven
audio LLMs in the bake-off. **1.1 MB gzipped.**

`data/` is build output and gitignored — it holds 7.9 GB of clips, VAD caches
and mel caches, all reproducible from the corpus. But the labels are the part of
this project that is not reproducible from anything else, and without them a
clone cannot run a single analysis step. So they ship here.

| | rows | what |
|---|---|---|
| `samples.jsonl.gz` | 17,209 | boundaries with the **rule-derived** label |
| `samples_llm.jsonl.gz` | 16,216 | the same rows relabelled by `gemini-3.7-flash` |
| `batch_<model>_ca216b25.jsonl.gz` | 197 / 18,488 | one file per labeller: verdict, confidence, reason |

`ca216b25` identifies the prompt. The two 18k files are the bulk run; the
197-row files are the bake-off against the human listening pass.

## What was removed

`text`, `next_text` and `llm_reason` — everything derived from the corpus
transcript. No analysis step reads them, and the published
[HuggingFace dataset](https://huggingface.co/datasets/santhosh-005/tamil-eot)
does not carry them either, so this keeps the two releases consistent.

Everything that decides a label is here: `label`, `label_pipeline`,
`llm_verdict`, `llm_conf`, `dispute`, `source`, `harvest`, `gap`, `prev_dur`,
`split`, `sid`.

## Reading them

`tamileot.labelling.read_lines()` prefers `data/gold/` and falls back here, so a
full local pipeline run is never shadowed by these copies:

```python
from tamileot import labelling
rows = labelling.load_rows("samples_llm.jsonl")
```

Or directly — they are plain gzipped JSONL:

```bash
zcat labels/samples_llm.jsonl.gz | head -1 | python -m json.tool
```

Licence: CC BY 4.0, as derived work from SPRING_INX Tamil R1 (SPRING Lab, IIT
Madras). The audio itself is not redistributed here.
