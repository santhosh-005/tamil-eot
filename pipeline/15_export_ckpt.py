#!/usr/bin/env python3
"""Step 2b -- turn a Colab `.pt` state dict into the shippable ONNX, locally.

The notebook's §6 does this, but §6 runs last and is the first thing a Colab
usage limit kills. Three trained checkpoints came off the last session with no
ONNX beside them, and `16_evaluate.py` only reads ONNX -- so weights that
were paid for sat unmeasured.

Nothing about the export needs a GPU. This reconstructs the same model class,
loads the state dict **strictly** (which is what actually verifies the config
guessed below is the right one -- a wrong `d_model` or layer count fails on
shape, it does not silently produce a worse model), exports the same graph, and
checks the ONNX against torch before writing.

Needs `transformers` for the `WhisperEncoder` class, which is not in
`requirements.txt` because nothing else in the pipeline imports it:

    pip install transformers
    python pipeline/15_export_ckpt.py data/colab/lr_5e-05.pt --out base-auc.onnx
    python pipeline/16_evaluate.py --model base-auc.onnx --out x.md

`--against` re-exports a checkpoint whose ONNX already exists and compares the
two probability vectors. That is the control for this script: if a local export
of `smart-turn-tamil-base/best.pt` does not reproduce the shipped graph, no
number produced here is trustworthy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

# The only step in the pipeline that needs torch, so it is not in
# requirements.txt -- everything else runs against ONNX on a CPU box.
try:
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402
    from torch.nn.functional import softmax  # noqa: E402
    from transformers import WhisperConfig, WhisperPreTrainedModel  # noqa: E402
    from transformers.models.whisper.modeling_whisper import WhisperEncoder  # noqa: E402
except ImportError as e:
    raise SystemExit(
        f"{e}\n"
        "  step 15 is the one step that needs torch:\n"
        "    pip install 'torch>=2.4' 'transformers>=4.44'"
    ) from None

from tamileot.paths import GOLD, MODELS  # noqa: E402

# openai/whisper-{tiny,base} encoder geometry, written out rather than fetched:
# the export must work with no network, and `load_state_dict(strict=True)`
# below rejects a wrong guess on shape.
GEOM = {
    "tiny": dict(d_model=384, encoder_layers=4, encoder_attention_heads=6,
                 encoder_ffn_dim=1536),
    "base": dict(d_model=512, encoder_layers=6, encoder_attention_heads=8,
                 encoder_ffn_dim=2048),
}


class SmartTurnV3Model(WhisperPreTrainedModel):
    """Verbatim from the notebook §3, which is verbatim from upstream."""

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        config.max_source_positions = 400        # 8 s, not 30 s
        self.encoder = WhisperEncoder(config)
        h = config.d_model
        self.pool_attention = nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.post_init()

    def forward(self, input_features):
        z = self.encoder(input_features=input_features).last_hidden_state
        w = softmax(self.pool_attention(z), dim=1)
        return torch.sigmoid(self.classifier((z * w).sum(dim=1))).view(-1)


def build(sd: dict) -> SmartTurnV3Model:
    """Pick the encoder size from the weights themselves, not from a flag."""
    h = sd["classifier.0.weight"].shape[1]
    size = next((k for k, v in GEOM.items() if v["d_model"] == h), None)
    assert size, f"d_model {h} is neither whisper-tiny (384) nor -base (512)"
    g = GEOM[size]
    cfg = WhisperConfig(num_mel_bins=80, max_source_positions=400,
                        decoder_layers=g["encoder_layers"],
                        decoder_attention_heads=g["encoder_attention_heads"],
                        decoder_ffn_dim=g["encoder_ffn_dim"], **g)
    m = SmartTurnV3Model(cfg)
    m.load_state_dict(sd, strict=True)     # the real config check
    print(f"  {size}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
          f"{len(sd)} tensors loaded strict")
    return m.eval()


def probs(path: Path, X: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    nm = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {nm: X[i:i + 64]})[0].reshape(-1)
                           for i in range(0, len(X), 64)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", default=None, help="name under models/ (default: <ckpt>.onnx)")
    ap.add_argument("--against", default=None,
                    help="existing onnx under models/ to compare probabilities with")
    a = ap.parse_args()

    out = MODELS / (a.out or f"{a.ckpt.stem}.onnx")
    assert not out.exists(), f"{out} exists -- refusing to overwrite a shipped model"

    m = build(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    torch.onnx.export(m, torch.randn(1, 80, 800), str(out), dynamo=False,
                      input_names=["input_features"], output_names=["logits"],
                      opset_version=18,
                      dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}})

    # 512 real clips, not random noise: an export bug that only shows on
    # in-distribution input is exactly the kind that survives a smoke test.
    X = np.ascontiguousarray(np.load(GOLD / "melcache_test.npy", mmap_mode="r")[:512],
                             dtype=np.float32)
    po = probs(out, X)
    with torch.no_grad():
        pt = m(torch.from_numpy(X)).numpy()
    d = float(np.abs(po - pt).max())
    print(f"  onnx vs torch  max|delta| {d:.2e}   ({out.stat().st_size/1e6:.0f} MB)")
    assert d < 1e-4, "exported graph disagrees with the model it came from"
    assert 0.0 <= po.min() and po.max() <= 1.0, "output is not a probability"

    if a.against:
        pr = probs(MODELS / a.against, X)
        print(f"  vs {a.against}  max|delta| {float(np.abs(po - pr).max()):.2e}  "
              f"agree@0.5 {100*((po > .5) == (pr > .5)).mean():.2f}%")

    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
