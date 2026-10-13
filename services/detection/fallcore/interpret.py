"""Interpretability probes: gradient saliency and attention capture.

Both of these are fiddly enough to be worth having tested code for rather than
a cell that looks right.

On attention in particular: nn.TransformerEncoderLayer calls its
MultiheadAttention with need_weights=False and throws the weights away, so a
plain forward hook observes nothing useful. The workaround is to hook the
attention submodule and recompute it with need_weights=True -- which naively
re-enters the same hook and recurses forever. AttentionRecorder guards against
that.
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as cfg


class AttentionRecorder:
    """Capture per-layer self-attention weights during a forward pass.

    Use as a context manager:

        with AttentionRecorder(model) as rec:
            model(x)
        maps = rec.maps      # one (B, heads, T, T) array per encoder layer
    """

    def __init__(self, model):
        self.model = model
        self.maps: list[np.ndarray] = []
        self._busy = False
        self._handles: list = []

    def _hook(self, module, args, kwargs, output):
        # The recompute below calls `module` again, which fires this same hook.
        # Without the guard that recurses until the stack runs out.
        if self._busy:
            return
        self._busy = True
        try:
            q = args[0] if args else kwargs.get("query")
            if q is None:
                return
            with torch.no_grad():
                _, weights = module(q, q, q, need_weights=True,
                                    average_attn_weights=False)
            self.maps.append(weights.detach().cpu().numpy())
        finally:
            self._busy = False

    def __enter__(self) -> "AttentionRecorder":
        self.maps.clear()
        self._handles = [
            layer.self_attn.register_forward_hook(self._hook, with_kwargs=True)
            for layer in self.model.encoder.layers
        ]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> np.ndarray:
        """All layers concatenated along the head axis: (B, layers*heads, T, T)."""
        if not self.maps:
            raise RuntimeError("no attention captured -- was a forward pass run?")
        return np.concatenate(self.maps, axis=1)

    def attention_received(self) -> np.ndarray:
        """Mean attention each frame position receives, over heads and layers.

        Averaging down the query axis of the attention matrix gives, for each
        key position, how much the rest of the sequence attends to it.
        """
        return self.stacked()[0].mean(axis=0).mean(axis=0)


def keypoint_saliency(model, loader, device: str = "cuda") -> np.ndarray:
    """Mean |d(logit)/d(input)| per keypoint and channel, over a whole loader.

    Returns (17, 3) covering the (y, x, confidence) channels.

    Only valid for the 51-dimensional "full" feature layout -- the other
    variants do not have a one-to-one mapping back onto keypoints.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    total = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float64)
    n_seen = 0

    for x, _ in loader:
        x = x.to(device)
        if x.shape[-1] != cfg.N_FEATURES:
            raise ValueError(
                f"keypoint_saliency needs the 51-dim 'full' features, got "
                f"{x.shape[-1]}. Retrain with features='full' to use this probe."
            )
        x = x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()

        g = x.grad.detach().abs().cpu().numpy()
        g = g.reshape(g.shape[0], g.shape[1], cfg.N_KEYPOINTS, 3)
        total += g.sum(axis=(0, 1))
        n_seen += g.shape[0]

    return total / max(n_seen, 1)


BODY_REGIONS = {
    "head (nose, eyes, ears)": [0, 1, 2, 3, 4],
    "shoulders": [5, 6],
    "arms (elbows, wrists)": [7, 8, 9, 10],
    "hips": [11, 12],
    "legs (knees, ankles)": [13, 14, 15, 16],
}


def region_importance(per_keypoint: np.ndarray) -> dict[str, float]:
    """Average a per-keypoint score within each body region."""
    return {name: float(np.asarray(per_keypoint)[idx].mean())
            for name, idx in BODY_REGIONS.items()}
