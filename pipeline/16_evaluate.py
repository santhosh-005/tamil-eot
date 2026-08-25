#!/usr/bin/env python3
"""Step 2c -- score the fine-tuned model, in Smart Turn's own units.

Two things this exists to fix about reading the training output:

1. **The FPR conventions differ.** Smart Turn's published tables report FP/N
   and FN/N -- they sum to the error rate. The notebook reports FP/(FP+TN),
   the standard rate. On this test set 28.9% standard is 10.6% in their units,
   and confusing the two makes a competitive model look like a broken one.

2. **The 0.5 threshold is inherited, not chosen.** With ROC-AUC 0.884 the
   ranking is good; where the cut goes is a separate decision, and in a voice
   agent a false `complete` (talking over the user) is not worth the same as a
   little added latency. The sweep below is the actual product decision.

    python pipeline/16_evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from tamileot.paths import GOLD, MODELS, REPORTS  # noqa: E402

# Smart Turn v3.2's published benchmark: accuracy, FPR (FP/N), FNR (FN/N).
PUBLISHED = {
    "English": (94.26, 3.75, 1.99), "overall (all languages)": (92.63, 4.73, 2.64),
    "Hindi": (90.11, 8.57, 1.32), "Bengali": (83.80, 10.90, 5.30),
    "Marathi": (82.43, 15.12, 2.45), "Vietnamese": (79.38, 8.86, 11.75),
}


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
            "acc": (tp + tn) / n,
            "fpr_pub": fp / n, "fnr_pub": fn / n,          # Smart Turn's units
            "fpr_std": fp / max(fp + tn, 1), "fnr_std": fn / max(fn + tp, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    o = np.argsort(-p); ys = y[o]
    tps, fps = np.cumsum(ys == 1), np.cumsum(ys == 0)
    return float(np.trapezoid(tps / max(tps[-1], 1), fps / max(fps[-1], 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smart-turn-tamil.onnx")
    ap.add_argument("--split", default="test")
    # Two models now ship (tiny drop-in, base high-accuracy). A fixed output
    # path meant whichever ran last silently replaced the other's report.
    ap.add_argument("--out", default="finetune_results_tiny.md")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (GOLD / "samples_llm.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("llm_ok") and r["split"] == a.split]
    feats = np.ascontiguousarray(np.load(GOLD / f"melcache_{a.split}.npy"), dtype=np.float32)
    assert len(feats) == len(rows), f"mel cache is {len(feats)}, rows are {len(rows)}"
    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])

    sess = ort.InferenceSession(str(MODELS / a.model), providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    p = np.concatenate([sess.run(None, {nm: feats[i:i + 64]})[0].reshape(-1)
                        for i in range(0, len(feats), 64)])
    assert 0.0 <= p.min() and p.max() <= 1.0, "output is not a probability"

    m = confusion(y, (p > 0.5).astype(int))
    A = auc(y, p)

    L = ["# Fine-tuned Smart Turn — Tamil", "",
         f"`{a.model}` on the sealed {a.split} split: **{m['n']:,} clips from "
         f"{len({r['base'] for r in rows})} calls**, never trained on.", "",
         "## Two FPR conventions — check which one a number is in", "",
         "Smart Turn's published tables report **FP/N and FN/N**, which sum to the",
         "error rate. The training notebook reports **FP/(FP+TN)**, the standard rate.",
         f"They are not the same number: this model is **{100*m['fpr_std']:.1f}% standard** and",
         f"**{100*m['fpr_pub']:.1f}% in Smart Turn's units.** Compared like for like:", "",
         "| | accuracy | FPR (FP/N) | FNR (FN/N) |", "|---|---|---|---|",
         f"| **Tamil (this model)** | **{100*m['acc']:.2f}%** | **{100*m['fpr_pub']:.2f}%** | "
         f"**{100*m['fnr_pub']:.2f}%** |"]
    for k, (acc_, fpr_, fnr_) in PUBLISHED.items():
        L.append(f"| {k} | {acc_:.2f}% | {fpr_:.2f}% | {fnr_:.2f}% |")
    # Stated by comparison, not by assertion -- the ranking moves with the model.
    beat_acc = [k for k, v in PUBLISHED.items() if 100 * m["acc"] > v[0]]
    beat_fpr = [k for k, v in PUBLISHED.items() if 100 * m["fpr_pub"] < v[1]]
    L += ["",
          f"Tamil is above **{', '.join(beat_acc) or 'none'}** on accuracy and "
          f"**{', '.join(beat_fpr) or 'none'}** on false-positive rate. The published",
          "rows come from a different benchmark on TTS-generated audio and this one is",
          "real narrowband telephone conversation, so read them for scale, not as a",
          "like-for-like comparison.", ""]

    L += ["## Against the zero-shot baseline", "",
          "| | accuracy | FPR (std) | ROC-AUC |", "|---|---|---|---|",
          "| `smart-turn-v3.2` zero-shot | 70.30% | 36.39% | 0.751 |",
          f"| **fine-tuned** | **{100*m['acc']:.2f}%** | {100*m['fpr_std']:.2f}% | **{A:.3f}** |",
          f"| delta | **{100*m['acc']-70.30:+.2f}** | {100*m['fpr_std']-36.39:+.2f} | {A-0.751:+.3f} |",
          "",
          "| | predicted complete | predicted incomplete |", "|---|---|---|",
          f"| **actually complete** | {m['tp']:,} | {m['fn']:,} |",
          f"| **actually incomplete** | {m['fp']:,} | {m['tn']:,} |", ""]

    # --- the operating point is a product decision, not a default ------------
    L += ["## Choosing the threshold", "",
          f"ROC-AUC is **{A:.3f}** — the ranking is good. 0.5 is inherited from the",
          "training loop, not chosen. In a voice agent a false `complete` interrupts the",
          "user; a false `incomplete` costs a little latency. Those are not equal, so",
          "the cut belongs wherever the product wants it.", "",
          "| threshold | accuracy | FPR (std) | FNR (std) | FPR (FP/N) |",
          "|---|---|---|---|---|"]
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.96, 0.05):
        mm = confusion(y, (p > t).astype(int))
        if mm["acc"] > best_acc:
            best_acc, best_t = mm["acc"], t
        L.append(f"| {t:.2f} | {100*mm['acc']:.2f}% | {100*mm['fpr_std']:.2f}% | "
                 f"{100*mm['fnr_std']:.2f}% | {100*mm['fpr_pub']:.2f}% |")

    # the point where interrupting the user is held to 10% of pauses
    tgt = next((t for t in np.arange(0.30, 0.999, 0.01)
                if confusion(y, (p > t).astype(int))["fpr_std"] <= 0.10), None)
    bb = confusion(y, (p > best_t).astype(int))
    L += ["",
          f"Best accuracy is **{100*best_acc:.2f}% at threshold {best_t:.2f}** "
          f"(FPR {100*bb['fpr_std']:.1f}% std).",
          "Pick it on `dev`, never on this split — quoted here only as a ceiling."]
    if tgt is not None:
        mt = confusion(y, (p > tgt).astype(int))
        L += ["",
              f"To hold interruptions at 10% of pauses, threshold **{tgt:.2f}**: accuracy "
              f"{100*mt['acc']:.2f}%, FPR {100*mt['fpr_std']:.1f}% std "
              f"({100*mt['fpr_pub']:.1f}% FP/N), FNR {100*mt['fnr_std']:.1f}%.",
              "That is the trade a live agent would actually take."]

    # A subset accuracy without its own base rate is unreadable: the disputed
    # bucket is 93.9% `complete`, so both shipped models score *below* the
    # majority-class policy there while it looks like their best result.
    # Hence the `majority baseline` and `vs baseline` columns below.
    def subsets(title: str, keys, mask_of) -> None:
        L.extend(["", f"## By {title}", "",
                  "| | n | accuracy | majority baseline | vs baseline | FPR (std) |",
                  "|---|---|---|---|---|---|"])
        for s in keys:
            k = mask_of(s)
            if not k.any():
                continue
            mm = confusion(y[k], (p[k] > 0.5).astype(int))
            pos = int(y[k].sum())
            base = max(pos, mm["n"] - pos) / mm["n"]
            d = 100 * mm["acc"] - 100 * base
            L.append(f"| `{s}` | {mm['n']:,} | {100*mm['acc']:.2f}% | {100*base:.2f}% | "
                     f"**{d:+.2f}** | "
                     + (f"{100*mm['fpr_std']:.2f}% |" if mm["fp"] + mm["tn"] else "– |"))

    subsets("source", ("change", "hold_intra", "change_midseg", "hold_inter"),
            lambda s: src == s)
    disp = np.array([bool(r.get("dispute")) for r in rows])
    subsets("label agreement", ("agree", "disputed"),
            lambda s: disp == (s == "disputed"))

    out = REPORTS / a.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
