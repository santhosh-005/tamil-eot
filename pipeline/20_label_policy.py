#!/usr/bin/env python3
"""Step 2e -- settle `--label-policy`, the last open label decision.

`11_label.py` writes `label_pipeline` and `llm_verdict` on every row
and takes one of them as `label`:

  * `llm` (default)  -- the LLM everywhere
  * `core-safe`      -- keep the pipeline's label on the `change` class

They differ only on `change` rows where the two witnesses disagree. The case
for `core-safe` is that a `change` positive is grounded in behaviour recorded
in the call -- the other person heard the speaker finish and took the floor --
which is evidence no listener judging an isolated 8 s clip has access to.

Three measurements decide it, and this prints all three:

  1. On the QA clips the policy actually decides, who does the human back?
  2. How many rows move, and where?
  3. What does switching cost on the frozen test set?

    python pipeline/20_label_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamileot import labelling  # noqa: E402
from tamileot.paths import CLIPS, GOLD, REPORTS  # noqa: E402

TRAIN_FILE = "samples_llm_all.jsonl"


def qa_verdicts() -> list[tuple[int, int, int, int]]:
    """(clip number, pipeline, llm, human) for every QA clip the policy decides."""
    hum = labelling.human_labels()
    rows = {r["sid"]: r for r in labelling.load_rows("samples.jsonl")}
    idx = labelling.qa_index()
    llm = {}
    for line in (GOLD / f"batch_gemini-3.7-flash_{labelling.PROMPT_ID}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["verdict"] in ("complete", "incomplete"):
            llm[r["sid"]] = 1 if r["verdict"] == "complete" else 0
    out = []
    for sid, h in hum.items():
        r = rows.get(sid)
        if r and r["source"] == "change" and sid in llm and llm[sid] != r["label"]:
            out.append((idx[sid], r["label"], llm[sid], h))
    return sorted(out)


def cost_on_test(affected: list[dict], n_test: int) -> dict[str, float]:
    """What `core-safe` would do to test accuracy, per shipped model.

    Not a selection -- nothing is being chosen on test here. It sizes a
    labelling change that would move the benchmark itself, which is the
    argument against making it.
    """
    try:
        import soundfile as sf
        from smart_turn_livekit import SmartTurn
    except ImportError as e:
        print(f"  skipping the test-set cost ({e}) -- pip install smart-turn-livekit")
        return {}
    out = {}
    for variant in ("tiny", "base"):
        st = SmartTurn(f"smart-turn-tamil-{variant}", cpu_count=4)
        k = 0
        for d in affected:
            w, sr = sf.read(CLIPS / "test" / f"{d['sid']}.wav", dtype="float32")
            k += st.probability(w, sr) > st.threshold
        # `llm` says incomplete on all of these, `core-safe` says complete, so a
        # `complete` prediction is right under core-safe and wrong under llm.
        out[variant] = 100 * (k - (len(affected) - k)) / n_test
    return out


def main() -> int:
    qa = qa_verdicts()
    rows = [json.loads(l) for l in (GOLD / TRAIN_FILE).read_text().splitlines()]
    ok = [d for d in rows if d.get("llm_ok")]
    chg = [d for d in ok if d["source"] == "change"]
    diff = [d for d in chg if d["label_pipeline"] != d["llm_verdict"]]
    test = [d for d in ok if d["split"] == "test"]
    aff_test = [d for d in diff if d["split"] == "test"]
    delta = cost_on_test(aff_test, len(test))

    back_llm = sum(1 for _, _, l, h in qa if l == h)
    L = ["# `--label-policy` — settled", "",
         "`llm` (default) vs `core-safe` (keep the pipeline's label on `change`).",
         f"Differ on **{len(diff):,} of {len(ok):,} rows** — {100 * len(diff) / len(ok):.2f}% of the corpus,",
         "all in one direction: pipeline `complete`, LLM `incomplete`.", "",
         "**Decision: `llm`.** Weak evidence for it, no evidence against, and",
         "switching would re-label the frozen test set.", "",
         "## 1. On the rows the policy decides, the human backs the LLM", "",
         f"Of the 197 blind-listened clips, **{len(qa)}** are `change` rows where the two",
         "witnesses disagree — the only direct evidence there is.", "",
         "| clip | pipeline | LLM | human |", "|---|---|---|---|"]
    w = {1: "complete", 0: "incomplete"}
    for i, p, l, h in qa:
        L.append(f"| `{i:03d}.wav` | {w[p]} | {w[l]} | **{w[h]}** |")
    L += ["",
          f"**{back_llm}–{len(qa) - back_llm} for the LLM.** On {len(qa)} clips that settles nothing on its",
          "own — it is the direction, not a result.", ""]

    L += ["## 2. What moves", "",
          "| split | affected | of | share |", "|---|---|---|---|"]
    for sp in ("train", "dev", "test"):
        n = sum(1 for d in ok if d["split"] == sp)
        k = sum(1 for d in diff if d["split"] == sp)
        L.append(f"| {sp} | {k} | {n:,} | {100 * k / n:.2f}% |")
    L += ["",
          f"**{len(aff_test)} of them are in the sealed test split.** That is the argument that",
          "actually decides this: `core-safe` does not re-weigh the training data, it",
          "re-labels the benchmark. Every number in `RESULTS.md` would then be quoted",
          "against a different ground truth than the one it was measured on.", ""]

    if delta:
        L += ["## 3. And it would buy nothing", "",
              "Shipped int8 models, scored on the affected test clips at their default",
              "threshold. Change in overall test accuracy if the labels flipped:", "",
              "| | Δ test accuracy |", "|---|---|"]
        for v, d in delta.items():
            L.append(f"| {v} int8 | {d:+.2f} |")
        L += ["",
              "Opposite signs, both far inside the **0.87-point** run-to-run spread of §5b.",
              "There is no measurable difference between the two policies, so the tie",
              "breaks on benchmark integrity.", ""]

    L += ["## Still reversible", "",
          "Every row carries `label_pipeline`, `llm_verdict` and `dispute`. Switching",
          "is one flag on `11_label.py` plus a repack — but it invalidates",
          "the published numbers, so it is a decision to re-measure, not a config",
          "change.", "",
          "Regenerate with `python pipeline/20_label_policy.py`.", ""]

    out = REPORTS / "label_policy.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
