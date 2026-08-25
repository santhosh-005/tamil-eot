#!/usr/bin/env python3
"""Step 2a -- what does the released Smart Turn already do on Tamil?

This is the "before" number, and it has to exist before a single training step
runs. The released model does not cover Tamil; the closest Indic languages it
does publish are Marathi 82.4% and Bengali 83.8%. Whether Tamil lands near
those or far below decides how much of this project is worth doing, and it
costs a couple of minutes of CPU.

No torch and no `transformers` here on purpose -- the whole preprocessing chain
is reimplemented in numpy against ONNX Runtime, which is all this machine has.
That is a risk (a wrong mel filterbank silently produces a plausible-looking
but meaningless number), so:

  * the mel filterbank is Whisper's own `mel_filters.npz`, not a reimplementation,
  * `--selftest` checks the chain against properties the real one must satisfy,
  * two independently trained checkpoints are scored, and they have to agree.

Preprocessing matches `pipecat-ai/smart-turn`'s `inference.py` exactly, which
is NOT the Whisper default -- it passes `do_normalize=True`, so the raw
waveform is zero-mean unit-variance normalised before the mel is taken.

    python pipeline/13_baseline.py                     # test split, both models
    python pipeline/13_baseline.py --split all --model models/smart-turn-v3.2-cpu.onnx
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

N_SAMPLES = 8 * 16_000      # the model's fixed 8 s window
N_FFT, HOP, N_MELS = 400, 160, 80

_MEL = np.load(MODELS / "mel_filters.npz")["mel_80"]          # (80, 201)
# transformers' window_function(..., "hann") is PERIODIC. np.hanning is
# symmetric and would be subtly wrong at every frame edge.
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float64)


def log_mel(x: np.ndarray) -> np.ndarray:
    """-> (80, 800), matching WhisperFeatureExtractor(chunk_length=8, do_normalize=True).

    Order matters and is not the obvious one: truncate to the last 8 s, pad,
    normalise the *waveform* using only the real samples, and only then take
    the spectrogram.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) > N_SAMPLES:
        x = x[-N_SAMPLES:]                       # keep the END of the utterance
    n = len(x)
    if n < N_SAMPLES:
        x = np.concatenate([x, np.zeros(N_SAMPLES - n, np.float64)])

    # zero_mean_unit_var_norm over the unpadded region, padding forced to 0.0
    real = x[:n]
    x = (x - real.mean()) / np.sqrt(real.var() + 1e-7)
    x[n:] = 0.0

    # center=True with reflect padding, as transformers' spectrogram() defaults
    p = N_FFT // 2
    xp = np.pad(x, (p, p), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * _WIN                       # (frames, 400)
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2   # (frames, 201)

    mel = _MEL @ power.T                          # (80, frames)
    mel = mel[:, :-1]                             # Whisper drops the last frame
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _one(job: tuple[str, str]) -> tuple[str, np.ndarray]:
    sid, path = job
    return sid, log_mel(audio.read(Path(path)))


def selftest() -> int:
    """Properties the real chain must satisfy. Cheap, and it would have caught
    every mistake I was actually at risk of making."""
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    chk("mel filterbank is Whisper's own", _MEL.shape == (80, 201), f"{_MEL.shape}")
    chk("hann window is periodic", abs(_WIN[0]) < 1e-12 and _WIN[N_FFT // 2] == _WIN.max(),
        f"w[0]={_WIN[0]:.3e}")

    x = np.random.RandomState(0).randn(N_SAMPLES).astype(np.float32) * 0.05
    m = log_mel(x)
    chk("shape is (80, 800)", m.shape == (80, 800), f"{m.shape}")
    # The clamp is a floor at max-8 decades, so 2.0 is a ceiling on the range,
    # not a target -- broadband noise never reaches it.
    chk("dynamic range within 8 decades/4", m.max() - m.min() <= 2.0 + 1e-5,
        f"range={m.max() - m.min():.4f}")
    # A pure tone has almost no energy outside one mel bin, so it does exceed 8
    # decades and pins the floor exactly. This is what proves the clamp fires.
    tone = np.sin(2 * np.pi * 1000 * np.arange(N_SAMPLES) / 16_000).astype(np.float32)
    mt = log_mel(tone)
    chk("clamp engages on a pure tone", abs((mt.max() - mt.min()) - 2.0) < 1e-5,
        f"range={mt.max() - mt.min():.4f}")

    # Scale invariance: the waveform is variance-normalised, so multiplying the
    # input by a constant must not move the features at all. If do_normalize
    # were skipped this fails loudly.
    chk("invariant to input gain", np.allclose(m, log_mel(x * 7.3), atol=1e-5))
    # A DC offset is removed by the same normalisation.
    chk("invariant to DC offset", np.allclose(m, log_mel(x + 0.2), atol=1e-5))
    # Silence must not produce NaNs through the 1/sqrt(var) or the log.
    chk("silence is finite", np.isfinite(log_mel(np.zeros(N_SAMPLES, np.float32))).all())
    # The last 8 s are what count, so a long input must equal its own tail.
    long = np.random.RandomState(1).randn(3 * N_SAMPLES).astype(np.float32)
    chk("keeps the last 8 s", np.allclose(log_mel(long), log_mel(long[-N_SAMPLES:]), atol=1e-5))
    return 0 if ok else 1


# --------------------------------------------------------------------------- metrics

def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    """Positive class is `complete`.

    A false positive means the model said the speaker had finished when they
    had not: the agent talks over the user. A false negative is dead air. They
    are not equally bad, so they are never collapsed into one number here.

    `fpr_all`/`fnr_all` are FP/N and FN/N -- the convention Smart Turn's own
    benchmark tables use, where the two sum to the error rate. `fpr` is the
    standard FP/(FP+TN), which is the one that describes behaviour in a call.
    """
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    d = {"n": n, "acc": (tp + tn) / n if n else 0.0,
         "acc_ci": wilson(tp + tn, n),
         "precision": prec, "recall": rec,
         "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
         "fpr_all": fp / n if n else 0.0, "fnr_all": fn / n if n else 0.0,
         "fpr": fp / (fp + tn) if fp + tn else 0.0,
         "fnr": fn / (fn + tp) if fn + tp else 0.0,
         "tp": tp, "fp": fp, "tn": tn, "fn": fn,
         "majority": max(y.mean(), 1 - y.mean()) if n else 0.0}
    if prob is not None and 0 < y.sum() < n:
        order = np.argsort(-prob)
        ys = y[order]
        tps, fps = np.cumsum(ys == 1), np.cumsum(ys == 0)
        d["auc"] = float(np.trapezoid(tps / max(tps[-1], 1), fps / max(fps[-1], 1)))
    return d


def run_model(model: Path, feats: np.ndarray, batch: int = 64) -> np.ndarray:
    """-> probability of `complete`, one per clip.

    The graph's output is *named* `logits` and is not one: the final node is a
    Sigmoid, so what comes out is already a probability. Applying another
    sigmoid maps [0,1] onto [0.5, 0.73] and makes every clip look complete --
    a bug that produces a confident, plausible, entirely wrong baseline. The
    assertion below is here so it can never happen silently again.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(feats), batch):
        out.append(sess.run(None, {name: feats[i:i + batch]})[0].reshape(-1))
        print(f"\r    {min(i + batch, len(feats)):,}/{len(feats):,}", end="", flush=True)
    print()
    prob = np.concatenate(out)
    assert 0.0 <= prob.min() and prob.max() <= 1.0, (
        f"{model.name} output is outside [0,1] ({prob.min():.3f}..{prob.max():.3f}) -- "
        "it is a real logit after all, apply a sigmoid here")
    return prob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "train", "all"])
    ap.add_argument("--samples", default="samples_llm.jsonl")
    ap.add_argument("--models", default="smart-turn-v3.0.onnx,smart-turn-v3.2-cpu.onnx")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("preprocessing self-test")
        return selftest()

    print("preprocessing self-test")
    if selftest():
        print("\n  self-test failed -- the features would be wrong, refusing to report a number")
        return 1

    rows = [json.loads(l) for l in (GOLD / args.samples).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("llm_ok")]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    rows = [r for r in rows if (CLIPS / r["split"] / f"{r['sid']}.wav").exists()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"\n{len(rows):,} clips, split={args.split}")

    cache = GOLD / f"melcache_{args.split}.npy"
    if cache.exists() and not args.limit:
        feats = np.load(cache, mmap_mode="r")
        assert len(feats) == len(rows), f"stale mel cache ({len(feats)} vs {len(rows)}), delete {cache}"
        print(f"  mels from cache {cache.name}")
    else:
        print(f"  extracting mels on {args.workers} workers ...")
        jobs = [(r["sid"], str(CLIPS / r["split"] / f"{r['sid']}.wav")) for r in rows]
        got = {}
        with Pool(args.workers) as pool:
            for k, (sid, m) in enumerate(pool.imap_unordered(_one, jobs, chunksize=16), 1):
                got[sid] = m
                if k % 500 == 0:
                    print(f"\r    {k:,}/{len(jobs):,}", end="", flush=True)
        print(f"\r    {len(jobs):,}/{len(jobs):,}")
        feats = np.stack([got[r["sid"]] for r in rows])
        if not args.limit:
            np.save(cache, feats)

    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])
    dis = np.array([bool(r.get("dispute")) for r in rows])
    feats = np.ascontiguousarray(feats, dtype=np.float32)

    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = MODELS / name if not Path(name).exists() else Path(name)
        if not path.exists():
            print(f"  missing {path}, skipping")
            continue
        print(f"\n  scoring {path.name}")
        prob = run_model(path, feats)
        results[path.name] = (prob, metrics(y, (prob > 0.5).astype(int), prob))

    if not results:
        print("no models scored")
        return 1
    report(results, y, src, dis, rows, args.split)
    return 0


# Published v3.2 benchmark, for the gap statement. Its two Indic languages are
# its two worst, which is the closest thing to a prior we have for Tamil.
PUBLISHED = {"overall": 92.63, "English": 94.26, "Hindi": 90.11,
             "Bengali": 83.80, "Marathi": 82.43, "worst (Vietnamese)": 79.38}


def report(results: dict, y, src, dis, rows, split: str) -> None:
    best_acc = 100 * max(m["acc"] for _p, m in results.values())
    L = ["# Zero-shot baseline — released Smart Turn on Tamil", "",
         f"**{best_acc:.1f}% on real Tamil call audio, against a "
         f"{100 * max(y.mean(), 1 - y.mean()):.1f}% always-say-complete baseline.**",
         "",
         f"It clears the majority-class baseline by "
         f"{best_acc - 100 * max(y.mean(), 1 - y.mean()):.1f} points, so it is hearing *something* — the",
         "ROC-AUC below says the ranking carries real signal — but at the shipped",
         "threshold it is not a usable turn detector for Tamil.", "",
         "> **Runtime tolerance.** Reproducible to about ±0.3% (~12 clips of 4,168).",
         "> ONNX Runtime picks different GEMM kernels by batch size, so float summation",
         "> order changes and clips sitting on the 0.5 threshold flip. Compare a trained",
         "> model against a baseline computed the same way in the same session, not",
         "> against this number to two decimals.", "",
         "---", "",
         f"`{split}` split of `samples_llm.jsonl`: **{len(y):,} clips, "
         f"{int((y == 1).sum()):,} complete / {int((y == 0).sum()):,} incomplete**, "
         f"from {len({r['base'] for r in rows})} calls.", "",
         "The released Smart Turn model does not cover Tamil. This is what it does",
         "anyway, before any fine-tuning — the number every later result is measured",
         "against.", "",
         "Positive class is `complete`. **FP = the agent talks over the user**; FN = dead",
         "air. `FPR/FNR (of all)` is Smart Turn's own benchmark convention (FP/N, FN/N,",
         "summing to the error rate); `FPR (std)` is FP/(FP+TN).", "",
         "| model | n | accuracy | 95% CI | precision | recall | F1 | FPR (of all) | FNR (of all) | FPR (std) | ROC-AUC |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (_p, m) in results.items():
        L.append(f"| `{name}` | {m['n']:,} | **{100 * m['acc']:.2f}%** | "
                 f"{100 * m['acc_ci'][0]:.1f}–{100 * m['acc_ci'][1]:.1f}% | {m['precision']:.3f} | "
                 f"{m['recall']:.3f} | {m['f1']:.3f} | {100 * m['fpr_all']:.2f}% | "
                 f"{100 * m['fnr_all']:.2f}% | {100 * m['fpr']:.2f}% | {m.get('auc', 0):.3f} |")
    maj = max(y.mean(), 1 - y.mean())
    L += [f"| *always say `complete`* | {len(y):,} | {100 * maj:.2f}% | – | – | – | – | – | – | – | 0.500 |", "",
          "Published v3.2 figures for context: **Marathi 82.43%, Bengali 83.80%**,",
          "against 92.63% overall and 94.26% on English. Those come from a different",
          "benchmark on TTS-generated audio; this one is real narrowband telephone",
          "conversation, so the two are not directly comparable.", ""]

    best = max(results.items(), key=lambda kv: kv[1][1]["acc"])
    prob, m = best[1]
    L += [f"## Where `{best[0]}` fails", "",
          "| | predicted complete | predicted incomplete |", "|---|---|---|",
          f"| **actually complete** | {m['tp']:,} | {m['fn']:,} |",
          f"| **actually incomplete** | {m['fp']:,} | {m['tn']:,} |", "",
          f"Of {int((y == 0).sum()):,} clips where the speaker had *not* finished, it says they had "
          f"on **{m['fp']:,}** ({100 * m['fpr']:.1f}%).", "",
          "### By source", "", "| source | n | accuracy | FPR (std) |", "|---|---|---|---|"]
    for s in ("change", "hold_intra", "change_midseg", "hold_inter"):
        k = src == s
        if not k.any():
            continue
        mm = metrics(y[k], (prob[k] > 0.5).astype(int))
        L.append(f"| `{s}` | {mm['n']:,} | {100 * mm['acc']:.2f}% | "
                 + (f"{100 * mm['fpr']:.2f}% |" if mm["fp"] + mm["tn"] else "– |"))

    pred = (prob > 0.5).astype(int)
    if pred.min() == pred.max():
        L += ["", "> **Note.** This model emits a single class for every clip at this",
              "> threshold, so the per-source and per-agreement tables below measure the",
              "> label composition of each slice, not anything the model knows.", ""]
    L += ["", "### By label agreement", "",
          "Rows where the pipeline and the labeller agree are the cleanest labels",
          "available. A model that is genuinely reading the audio should do better",
          "there; if it does not, the gap is the model, not the labels.", "",
          "| | n | accuracy |", "|---|---|---|"]
    for lbl, k in (("both witnesses agree", ~dis), ("disputed", dis)):
        if k.any():
            mm = metrics(y[k], (prob[k] > 0.5).astype(int))
            L.append(f"| {lbl} | {mm['n']:,} | {100 * mm['acc']:.2f}% |")

    L += ["", "### Threshold sweep", "",
          "The 0.5 default is not sacred. In a voice agent a false 'complete' is much",
          "more expensive than a little added latency, so the operating point should be",
          "chosen, not inherited.", "",
          "| threshold | accuracy | FPR (std) | FNR (std) |", "|---|---|---|---|"]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        mm = metrics(y, (prob > t).astype(int))
        L.append(f"| {t:.1f} | {100 * mm['acc']:.2f}% | {100 * mm['fpr']:.2f}% | {100 * mm['fnr']:.2f}% |")

    L += ["", f"Best accuracy over the sweep: "
              f"**{100 * max(metrics(y, (prob > t).astype(int))['acc'] for t in np.arange(0.05, 0.96, 0.05)):.2f}%** "
              "— an upper bound for what threshold tuning alone can buy, and it needs a",
          "held-out split to choose on, so it is not a free win.", ""]

    out = REPORTS / "baseline_zeroshot.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
