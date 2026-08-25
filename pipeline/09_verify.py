#!/usr/bin/env python3
"""Step 1a.5 -- try to break the gold set before a model does.

A turn-detection set is easy to build and easy to build *wrong*: if the two
classes differ in any trivial way the clips carry -- loudness, clip length,
how much silence trails the speech -- a model will learn that instead of
prosody, score beautifully offline, and fall apart in a live call.

So this does not report accuracy. It attacks:

  1. leakage      -- no call may appear in two splits.
  2. geometry     -- clip length and trailing silence must not separate classes.
  3. shortcut     -- fit a deliberately dumb model on cheap global features
                     (energy, duration, spectral tilt). It has no access to
                     prosody or words, so it *should* be near chance. High AUC
                     here means the label is readable off an artefact.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tamileot import audio, rules  # noqa: E402
from tamileot.paths import CLIPS, GOLD, REPORTS  # noqa: E402

FEATS = ["dur", "rms_all", "rms_last200", "rms_last500", "rms_pre500",
         "peak", "zcr", "tilt", "p90_over_p50", "voiced_frac"]


def features(x: np.ndarray) -> np.ndarray:
    n = len(x)
    last2, last5 = x[-int(0.2 * audio.SR):], x[-int(0.5 * audio.SR):]
    pre5 = x[-int(0.7 * audio.SR): -int(0.2 * audio.SR)]

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt((a.astype(np.float64) ** 2).mean())) if len(a) else 0.0

    fr = 512
    m = n // fr
    e = np.sqrt((x[: m * fr].reshape(m, fr).astype(np.float64) ** 2).mean(1)) if m else np.zeros(1)
    p50, p90 = np.percentile(e, 50), np.percentile(e, 90)
    sp = np.abs(np.fft.rfft(x[-int(1.0 * audio.SR):] * np.hanning(min(n, int(audio.SR)))))
    half = len(sp) // 2
    return np.array([
        n / audio.SR, rms(x), rms(last2), rms(last5), rms(pre5),
        float(np.abs(x).max()),
        float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
        float(np.log1p(sp[half:].sum()) - np.log1p(sp[:half].sum())),
        float(p90 / (p50 + 1e-9)),
        float((e > max(p90 * 0.1, 1e-4)).mean()),
    ], dtype=np.float64)


def load_features(rows: list[dict], cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Cheap global features, cached by sid.

    Reading 16k eight-second clips and running an FFT over each is a couple of
    minutes; the features depend only on the audio, never on the label, so a
    relabelling does not invalidate them. That makes it cheap to re-run this
    against several label policies, which is the whole point of doing so.
    """
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        have = {s: v for s, v in zip(z["sid"], z["X"])}
    X, y, grp, missing, fresh = [], [], [], 0, 0
    for r in rows:
        f = have.get(r["sid"])
        if f is None:
            p = CLIPS / r["split"] / f"{r['sid']}.wav"
            if not p.exists():
                missing += 1
                continue
            f = features(audio.read(p))
            have[r["sid"]] = f
            fresh += 1
        X.append(f)
        y.append(r["label"])
        grp.append(r["split"])
    if fresh:
        np.savez_compressed(cache, sid=np.array(list(have)),
                            X=np.vstack(list(have.values())))
    return np.vstack(X), np.array(y), np.array(grp), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples.jsonl",
                    help="samples_llm.jsonl to verify the relabelled set")
    ap.add_argument("--sources", default="core", choices=["core", "all"],
                    help="'all' includes change_midseg and hold_inter, which only "
                         "carry a usable label once an LLM has judged them")
    ap.add_argument("--save", action="store_true",
                    help="also write the transcript and a machine-readable summary")
    args = ap.parse_args()

    # Everything printed is also kept, so a relabelling report can quote the
    # attack results instead of a human copying numbers between files.
    lines: list[str] = []
    summary: dict = {"samples": args.samples, "sources": args.sources}

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    path = GOLD / args.samples
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    # A relabelled file marks rows the labeller actually judged. Those are the
    # only ones whose label means anything in it, whatever `source` says.
    relabelled = any("llm_ok" in r for r in rows[:1])
    if relabelled:
        rows = [r for r in rows if r.get("llm_ok")]
    if args.sources == "core":
        rows = [r for r in rows if rules.is_core(r)]
    out(f"{len(rows):,} usable samples from {path.name}"
          + ("  [LLM-relabelled]" if relabelled else "  [pipeline labels]")
        + f"  sources={args.sources}\n")

    # --- 1. leakage --------------------------------------------------------
    where = defaultdict(set)
    for r in rows:
        where[r["base"]].add(r["split"])
    bad = {b: s for b, s in where.items() if len(s) > 1}
    out("1. SPLIT LEAKAGE")
    out(f"   calls appearing in more than one split: {len(bad)}"
          + ("   <- clean" if not bad else f"   {list(bad)[:5]}"))
    ids = [r["sid"] for r in rows]
    out(f"   duplicate sample ids: {len(ids) - len(set(ids))}")

    # --- load audio --------------------------------------------------------
    X, y, grp, missing = load_features(rows, GOLD / "verify_features.npz")
    if missing:
        out(f"   WARNING: {missing} clips missing on disk")

    # --- 1b. metadata confound --------------------------------------------
    # Not everything that separates the classes is a bug in the audio. The
    # gates themselves impose structure: `hold_intra` needs a pause inside one
    # transcript segment, so it is drawn from longer utterances than `change`,
    # which sits at a segment end. That is a confound we created, and phase 2
    # should sample to match on it rather than pretend it is not there.
    out("\n1b. METADATA CONFOUND (induced by the gates, not by the audio)")
    out(f"   {'field':14s} {'complete':>12s} {'incomplete':>12s} {'std-gap':>9s}")
    for fld in ("prev_dur", "gap"):
        a = np.array([r[fld] for r in rows if r["label"] == 1])
        b = np.array([r[fld] for r in rows if r["label"] == 0])
        d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-12)
        summary[f"d_{fld}"] = round(float(d), 3)
        out(f"   {fld:14s} {a.mean():12.3f} {b.mean():12.3f} {d:9.2f}"
              + ("  <-- match on this when training" if abs(d) > 0.3 else ""))
    out("   (`gap` is metadata only -- it is never inside the clip, so a model"
          "\n    cannot read it; `prev_dur` leaks into the clip as voiced fraction.)")

    # --- 2. geometry -------------------------------------------------------
    out("\n2. CLIP GEOMETRY (must not separate the classes)")
    out(f"   {'feature':14s} {'complete':>12s} {'incomplete':>12s} {'std-gap':>9s}")
    for i, name in enumerate(FEATS):
        a, b = X[y == 1, i], X[y == 0, i]
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
        d = (a.mean() - b.mean()) / pooled          # Cohen's d
        flag = "  <-- suspicious" if abs(d) > 0.8 else ""
        summary[f"d_{name}"] = round(float(d), 3)
        out(f"   {name:14s} {a.mean():12.4f} {b.mean():12.4f} {d:9.2f}{flag}")

    # --- 3. shortcut probe -------------------------------------------------
    tr, te = grp == "train", grp == "test"
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    auc = roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])
    base = max(y[te].mean(), 1 - y[te].mean())
    acc = (clf.predict(sc.transform(X[te])) == y[te]).mean()
    out("\n3. SHORTCUT PROBE  (dumb global features, held-out calls)")
    out(f"   AUC {auc:.3f}   accuracy {acc:.3f}   majority-class {base:.3f}")
    verdict = ("clean: the label is not readable from crude acoustics"
               if auc < 0.65 else
               "WEAK LEAK: inspect the features above" if auc < 0.80 else
               "STRONG LEAK: the set is broken, do not train on it")
    out(f"   -> {verdict}")
    top = np.argsort(-np.abs(clf.coef_[0]))[:4]
    out("   strongest crude features: "
          + ", ".join(f"{FEATS[i]}({clf.coef_[0][i]:+.2f})" for i in top))

    summary.update(n=len(rows), auc=round(float(auc), 3), acc=round(float(acc), 3),
                   majority=round(float(base), 3), verdict=verdict.split(":")[0],
                   n_complete=int((y == 1).sum()), n_incomplete=int((y == 0).sum()))
    if args.save:
        stem = f"{Path(args.samples).stem}_{args.sources}"
        (REPORTS / f"verify_{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (REPORTS / f"verify_{stem}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        out(f"\n  -> {REPORTS / f'verify_{stem}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
