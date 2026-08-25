#!/usr/bin/env python3
"""Step 3 -- choose the shipping thresholds, on `dev`.

`16_evaluate.py` sweeps the threshold on **test** and prints the best one.
That number is a ceiling, not a choice: picking it and then shipping it is
selecting on the benchmark, and the accuracy you quote afterwards is no longer
an out-of-sample estimate.

So the thresholds the package ships are picked here, on `dev`, and only then
reported against test. Two operating points, because a voice agent has two
sensible ones:

  balanced  the dev-best accuracy
  polite    the tightest threshold whose dev FPR (standard, FP/(FP+TN)) is
            <= 10% -- i.e. the agent interrupts the user on at most 1 in 10
            pauses. This is the one most products actually want; accuracy is
            the wrong objective when the two errors cost different amounts.

    python pipeline/18_operating_point.py
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from tamileot import audio  # noqa: E402
from tamileot.paths import CLIPS, GOLD, MODELS, REPORTS  # noqa: E402

# The published plugin, so the features scored here are the ones served live.
#   pip install smart-turn-livekit
from smart_turn_livekit.features import log_mel  # noqa: E402

# Picked on fp32. int8 dynamic is what ships, but it agrees with fp32 on 97.6%
# (tiny) / 98.6% (base) of clips and lands within 0.36 / 0.10 accuracy points
# (reports/quantisation.md), so the thresholds carry. `--precision int8`
# re-picks on the shipped build if that ever stops being true.
MODELS_TO_SCORE = {
    "fp32": {"tiny": "smart-turn-tamil-tiny/smart-turn-tamil.onnx",
             "base": "smart-turn-tamil-base/smart-turn-tamil.onnx"},
    "int8": {"tiny": "smart-turn-tamil-tiny/smart-turn-tamil-int8-dynamic.onnx",
             "base": "smart-turn-tamil-base/smart-turn-tamil-int8-dynamic.onnx"},
}
FPR_BUDGET = 0.10


def _one(job: tuple[str, str]) -> np.ndarray:
    return log_mel(audio.read(Path(job[1])))


def rates(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p > t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"acc": (tp + tn) / len(y), "fpr_std": fp / max(fp + tn, 1),
            "fnr_std": fn / max(fn + tp, 1), "fpr_pub": fp / len(y), "fnr_pub": fn / len(y)}


def feats_for(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached on disk -- 2,325 clips is ~90 s of CPU and this gets re-run."""
    rows = [json.loads(l) for l in (GOLD / "samples_llm.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("llm_ok") and r["split"] == split]
    cache = GOLD / f"melcache_{split}.npy"
    if cache.exists():
        X = np.load(cache, mmap_mode="r")
        assert len(X) == len(rows), f"{cache} is {len(X)}, rows are {len(rows)}"
    else:
        jobs = [(r["sid"], str(CLIPS / split / f"{r['sid']}.wav")) for r in rows]
        with Pool() as pool:
            X = np.stack(pool.map(_one, jobs, chunksize=32))
        np.save(cache, X)
        print(f"  cached {cache.name}  {X.shape}")
    return np.ascontiguousarray(X, dtype=np.float32), np.array([r["label"] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--precision", default="fp32", choices=list(MODELS_TO_SCORE),
                    help="which build to pick on; int8 tracks fp32 within noise")
    a = ap.parse_args()
    out_name = a.out or (f"operating_point_{a.precision}.md"
                         if a.precision != "fp32" else "operating_point.md")

    Xd, yd = feats_for("dev")
    Xt, yt = feats_for("test")
    print(f"dev {len(yd):,}   test {len(yt):,}   precision {a.precision}")

    picked, L = {}, [
        f"# Shipping thresholds — picked on `dev`, on the **{a.precision}** build", "",
        "The threshold sweeps in `reports/finetune_results_*.md` run on **test** and",
        "report a ceiling. These are the thresholds the package actually ships, chosen",
        "on `dev` and only then measured on test — so the test column below is an",
        "out-of-sample estimate rather than the maximum of a sweep.", "",
        f"`polite` is the tightest threshold holding **dev FPR <= {100*FPR_BUDGET:.0f}%**",
        "(standard FP/(FP+TN)): the agent talks over the user on at most one pause in",
        "ten. `balanced` maximises dev accuracy.", ""]

    for name, rel in MODELS_TO_SCORE[a.precision].items():
        sess = ort.InferenceSession(str(MODELS / rel), providers=["CPUExecutionProvider"])
        nm = sess.get_inputs()[0].name
        run = lambda X: np.concatenate([sess.run(None, {nm: X[i:i + 64]})[0].reshape(-1)
                                        for i in range(0, len(X), 64)])
        pd_, pt_ = run(Xd), run(Xt)

        grid = np.arange(0.01, 1.00, 0.01)
        balanced = float(max(grid, key=lambda t: rates(yd, pd_, t)["acc"]))
        polite = next((float(t) for t in grid if rates(yd, pd_, t)["fpr_std"] <= FPR_BUDGET), 0.99)
        picked[name] = {"balanced": round(balanced, 2), "polite": round(polite, 2)}

        L += [f"## {name}", "",
              "| operating point | threshold | dev acc | dev FPR | **test acc** | test FPR | test FP/N |",
              "|---|---|---|---|---|---|---|"]
        for op, t in (("balanced", balanced), ("polite", polite), ("inherited 0.5", 0.5)):
            d, s = rates(yd, pd_, t), rates(yt, pt_, t)
            L.append(f"| `{op}` | {t:.2f} | {100*d['acc']:.2f}% | {100*d['fpr_std']:.1f}% | "
                     f"**{100*s['acc']:.2f}%** | {100*s['fpr_std']:.1f}% | {100*s['fpr_pub']:.2f}% |")
        L.append("")
        print(f"{name}: balanced {balanced:.2f}  polite {polite:.2f}")

    if a.precision == "fp32":                    # the picked-on build owns the file
        (GOLD / "thresholds.json").write_text(json.dumps(picked, indent=2) + "\n")
    L += ["## Read the `balanced` row against `inherited 0.5` before using it", "",
          "Tuning the threshold for accuracy on dev is not reliably worth doing here.",
          "The accuracy-vs-threshold curve is flat near its top and dev is 2,325 clips,",
          "so the argmax is largely sampling noise — and it can transfer *negatively*:",
          "if `balanced` scores below `inherited 0.5` on test above, that is the",
          "threshold having been fitted to dev.", "",
          "`polite` is the more robust of the two, because it targets a rate rather",
          "than an argmax. Expect its dev FPR budget to slip by a few points on test;",
          "the direction holds, the level does not.", "",
          "**The package therefore defaults to 0.5**, which is also what every headline",
          "number in `RESULTS.md` is quoted at. `polite` is one argument away.", "",
          "---", "", "Generated by `pipeline/18_operating_point.py`. The package reads the",
          "same values from `smart_turn_livekit.models.MODEL_REGISTRY`; duplicated",
          "there as literals so the package installs without this repo.", "",
          "```json", json.dumps(picked, indent=2), "```", ""]
    (REPORTS / out_name).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  -> {REPORTS / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
