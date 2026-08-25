#!/usr/bin/env python3
"""Step 4 -- int8 quantisation, because the released model is quantised and ours was not.

`smart-turn-v3.2-cpu.onnx` ships at 8.6 MB against our 32 MB fp32 tiny, same
architecture. Inspecting it explains the gap: 78 `QuantizeLinear` / 114
`DequantizeLinear` nodes and uint8/int8 initialisers. Upstream ships int8 and we
were shipping fp32, so "drop-in replacement" was true of the graph signature and
false of the size and latency.

Two modes, both measured rather than assumed:

  dynamic  weights int8, activations quantised at run time. No calibration.
  static   weights and activations int8, ranges calibrated from real audio.
           Usually faster, and the one that can lose accuracy.

**Calibration uses `dev`, never `test`.** Calibrating on the benchmark is
selecting on it -- the same mistake `18_operating_point.py` exists to avoid.

    python pipeline/19_quantize.py                 # both variants, both modes
    python pipeline/19_quantize.py --variants tiny
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402

from tamileot.paths import GOLD, MODELS, REPORTS  # noqa: E402

VARIANTS = {"tiny": "smart-turn-tamil-tiny", "base": "smart-turn-tamil-base"}
N_CALIB = 256


class MelReader(CalibrationDataReader):
    """Real Tamil mels from `dev`. Random noise would calibrate the ranges to
    a distribution the model never sees and quietly cost accuracy."""

    def __init__(self, X: np.ndarray, name: str):
        self.name, self.i, self.X = name, 0, X

    def get_next(self):
        if self.i >= len(self.X):
            return None
        x = self.X[self.i:self.i + 1]
        self.i += 1
        return {self.name: np.ascontiguousarray(x, dtype=np.float32)}


def _sess(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def score(path: Path, X: np.ndarray, y: np.ndarray) -> dict:
    """Two passes on purpose, with different session settings.

    Accuracy wants throughput: batch 64, all cores. Latency wants realism:
    batch 1, one thread, the shape a live agent actually runs. Measuring both
    from one single-threaded session makes the accuracy pass ~10x slower for no
    gain -- 4,168 clips x 6 builds on one core, which is how the first version
    of this script ran for an hour.
    """
    s = _sess(path, 0)                      # 0 = ORT picks, i.e. all cores
    nm = s.get_inputs()[0].name
    p = np.concatenate([s.run(None, {nm: X[i:i + 64]})[0].reshape(-1)
                        for i in range(0, len(X), 64)])
    pred = (p > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    o = np.argsort(-p); ys = y[o]
    tps, fps = np.cumsum(ys == 1), np.cumsum(ys == 0)
    auc = float(np.trapezoid(tps / max(tps[-1], 1), fps / max(fps[-1], 1)))

    s1 = _sess(path, 1)                     # batch 1, one thread: the agent's shape
    one = np.ascontiguousarray(X[:1])
    for _ in range(5):
        s1.run(None, {nm: one})
    ts = []
    for _ in range(40):
        t0 = time.perf_counter(); s1.run(None, {nm: one}); ts.append((time.perf_counter() - t0) * 1000)
    return {"acc": (tp + tn) / len(y), "auc": auc, "fpr_pub": fp / len(y), "fnr_pub": fn / len(y),
            "p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
            "mb": path.stat().st_size / 1e6, "probs": p}


def rows_for(split: str) -> np.ndarray:
    rows = [json.loads(l) for l in (GOLD / "samples_llm.jsonl").read_text().splitlines()]
    return np.array([r["label"] for r in rows if r.get("llm_ok") and r["split"] == split])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="tiny,base")
    ap.add_argument("--out", default="quantisation.md")
    ap.add_argument("--requantise", action="store_true",
                    help="rebuild the int8 graphs instead of reusing them")
    a = ap.parse_args()

    Xt = np.ascontiguousarray(np.load(GOLD / "melcache_test.npy", mmap_mode="r"), dtype=np.float32)
    yt = rows_for("test")
    Xd = np.load(GOLD / "melcache_dev.npy", mmap_mode="r")
    calib = np.ascontiguousarray(Xd[np.linspace(0, len(Xd) - 1, N_CALIB).astype(int)],
                                 dtype=np.float32)
    print(f"test {len(yt):,}   calibration {len(calib)} clips from dev")

    L = ["# int8 quantisation", "",
         "Upstream's `smart-turn-v3.2-cpu.onnx` is int8 (78 `QuantizeLinear` nodes,",
         "uint8/int8 initialisers) and ours was fp32 — 8.6 MB against 32 MB for the",
         "same architecture. This closes that gap and measures what it costs.", "",
         f"Activations calibrated on **{N_CALIB} `dev` clips**, never test. Latency is",
         "single-threaded, batch 1, on an i5-12450H — the shape a live agent runs.", ""]

    for v in a.variants.split(","):
        v = v.strip()
        work = MODELS / VARIANTS[v]
        src = work / "smart-turn-tamil.onnx"
        dyn = work / "smart-turn-tamil-int8-dynamic.onnx"
        st = work / "smart-turn-tamil-int8.onnx"

        # Quantising is the slow half and its output is deterministic, so a
        # rerun to fix the *measurement* should not redo it. --requantise forces.
        if a.requantise or not (dyn.exists() and st.exists()):
            prep = work / "_prep.onnx"
            quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
            quantize_dynamic(str(prep), str(dyn), weight_type=QuantType.QInt8)
            nm = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]) \
                .get_inputs()[0].name
            quantize_static(str(prep), str(st), MelReader(calib, nm),
                            quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
            prep.unlink(missing_ok=True)
            Path(str(prep) + ".data").unlink(missing_ok=True)
        else:
            print(f"  {v}: reusing existing int8 graphs")

        outs = {"fp32": src, "int8 dynamic": dyn, "int8 static": st}

        L += [f"## {v}", "",
              "| build | size | p50 | p95 | accuracy | AUC | FP/N | vs fp32 |",
              "|---|---|---|---|---|---|---|---|"]
        base = None
        for label, p in outs.items():
            m = score(p, Xt, yt)
            if base is None:
                base, agree = m, "—"
            else:
                agree = (f"{100*m['acc']-100*base['acc']:+.2f} acc, "
                         f"{100*((m['probs'] > .5) == (base['probs'] > .5)).mean():.1f}% same")
            # One decimal under 10 MB: whether tiny int8 is 8.7 or 9 MB is the
            # whole comparison against upstream's 8.68, and `.0f` loses it.
            mb = f"{m['mb']:.1f}" if m["mb"] < 10 else f"{m['mb']:.0f}"
            L.append(f"| {label} | {mb} MB | {m['p50']:.0f} ms | {m['p95']:.0f} ms | "
                     f"{100*m['acc']:.2f}% | {m['auc']:.3f} | {100*m['fpr_pub']:.2f}% | {agree} |")
            print(f"  {v:5s} {label:14s} {mb:>5s}MB  {m['p50']:6.1f}ms  "
                  f"{100*m['acc']:.2f}%  AUC {m['auc']:.3f}")
        L.append("")

    (REPORTS / a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  -> {REPORTS / a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
