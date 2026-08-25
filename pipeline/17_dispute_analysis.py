#!/usr/bin/env python3
"""Step 2d -- what the `dispute` flag is actually doing to the numbers.

Two questions the headline accuracy cannot answer:

1. **How good are the labels?** The 197 human-labelled QA clips are the only
   ground truth in the project. Scoring the LLM against them *split by dispute
   status* gives the label accuracy on each bucket, which weighted by the test
   split's dispute rate is the real ceiling on measured accuracy.

2. **Is the model actually good on disputed clips?** It scores higher there
   than on agreed ones, which looks like a strength. It is not: the disputed
   bucket is 93.9% `complete`, so the majority-class policy scores 93.9% and
   the model is *below* it. Any subset accuracy that isn't compared to its own
   base rate is unreadable.

    python pipeline/17_dispute_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot import labelling  # noqa: E402
from tamileot.paths import REPORTS  # noqa: E402

BOOL = {"complete": 1, "incomplete": 0}


def main() -> int:
    with (REPORTS / "qa/bakeoff_all_models_ca216b25.csv").open(encoding="utf-8-sig") as f:
        human = {r["sid"]: BOOL[r["you"]] for r in csv.DictReader(f) if r["you"] in BOOL}

    rows = labelling.load_rows("samples_llm.jsonl")
    qa = {d["sid"]: d for d in rows if d["sid"] in human}
    assert len(qa) == len(human), f"{len(human) - len(qa)} QA clips did not join"
    test = [d for d in rows if d.get("llm_ok") and d["split"] == "test"]

    L = ["# What `dispute` is doing to the numbers", "",
         f"`dispute` marks the {sum(bool(d.get('dispute')) for d in rows):,} clips where the gap-based",
         "pipeline and the LLM disagreed. It is not an incidental flag — it splits the",
         "corpus into two populations with different label quality *and* different class",
         "balance, and every aggregate number hides that.", "",
         "## 1. Label accuracy, from the 197 human labels", "",
         "| subset | n | % of QA | LLM vs human | pipeline vs human |", "|---|---|---|---|---|"]

    stat = {}
    for key, want in (("agree", False), ("disputed", True)):
        s = [d for d in qa.values() if bool(d.get("dispute")) == want]
        llm = sum(d["llm_verdict"] == human[d["sid"]] for d in s) / len(s)
        pipe = sum(d["label_pipeline"] == human[d["sid"]] for d in s) / len(s)
        stat[key] = llm
        L.append(f"| `{key}` | {len(s)} | {100*len(s)/len(qa):.1f}% | "
                 f"**{100*llm:.1f}%** | {100*pipe:.1f}% |")
    allllm = sum(d["llm_verdict"] == human[d["sid"]] for d in qa.values()) / len(qa)
    L.append(f"| **all** | {len(qa)} | 100.0% | **{100*allllm:.1f}%** | "
             f"{100*sum(d['label_pipeline'] == human[d['sid']] for d in qa.values())/len(qa):.1f}% |")

    dr = sum(bool(d.get("dispute")) for d in test) / len(test)
    ceil = (1 - dr) * stat["agree"] + dr * stat["disputed"]
    L += ["", f"The test split is **{100*dr:.1f}% disputed**, so the labels it is scored against",
          f"are roughly `{1-dr:.2f}×{100*stat['agree']:.1f} + {dr:.2f}×{100*stat['disputed']:.1f}` = "
          f"**{100*ceil:.1f}%** accurate.", "",
          f"That is the ceiling on any measured accuracy. **90% sits {100*ceil-90:.1f} points below it** —",
          "label quality is not what is holding the model back.", "",
          "## 2. Every subset accuracy needs its own base rate", "",
          "| subset | n | % complete | majority baseline |", "|---|---|---|---|"]

    for key, want in (("agree", False), ("disputed", True)):
        s = [d for d in test if bool(d.get("dispute")) == want]
        c = sum(d["label"] for d in s)
        L.append(f"| `{key}` | {len(s):,} | {100*c/len(s):.1f}% | **{100*max(c, len(s)-c)/len(s):.1f}%** |")
    c = sum(d["label"] for d in test)
    L.append(f"| **all** | {len(test):,} | {100*c/len(test):.1f}% | "
             f"{100*max(c, len(test)-c)/len(test):.1f}% |")

    L += ["", "The disputed bucket is near-single-class. Both shipped models score above",
          "their overall accuracy here and still **below** the majority baseline — see",
          "the `vs baseline` column in `reports/finetune_results_*.md`. A subset",
          "accuracy without its own base rate is unreadable, and this one covers 38%",
          "of the test set.", "",
          "Those clips are acoustically adversarial positives: the pipeline called them",
          f"`incomplete` on gap and prosody, and both the human and the LLM overrule",
          f"it ~{100*stat['disputed']:.0f}% of the time. The per-batch `pos_weight` (≈0.59 at 63% positives)",
          "pushes the model toward `incomplete` — wrong for all of them."]

    out = REPORTS / "dispute_analysis.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
