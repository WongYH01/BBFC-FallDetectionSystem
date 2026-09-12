"""The paper's appendix ablation tables, transcribed from Tables 14-20.

The PDF's table bodies do not survive text extraction, so these are typed in
from the rendered tables. Kept as data rather than as prose in a notebook cell
so the comparison against our own sweep is computed rather than narrated, and
so a transcription error is fixable in one place.

F1 values are percentages, exactly as printed. The baseline row of every table
is d=128, H=4, L=3, ff=256, dropout=0.1, sinusoidal, mean pooling, ReLU, full
features, lr=1e-4, seq=45, batch=64, BCE -- reported at F1 = 94.49 everywhere
except Table 19, which prints 94.59 for the same configuration. That 0.10
inconsistency is the paper's, not a typo here.
"""

#: dimension -> {value: F1 percent}
PAPER: dict[str, dict] = {
    "model dimension": {32: 91.08, 64: 93.81, 128: 94.49, 256: 96.62,
                        320: 95.83, 400: 96.63},
    "encoder layers":  {1: 92.87, 2: 95.09, 3: 94.49, 4: 94.94, 6: 94.32},
    "attention heads": {1: 94.25, 2: 94.42, 4: 94.49, 8: 94.60},
    "feedforward dim": {128: 95.02, 256: 94.49, 512: 94.77, 720: 95.79,
                        1024: 95.42},
    "dropout":         {0.0: 94.60, 0.1: 94.49, 0.2: 94.80, 0.3: 94.57,
                        0.4: 93.99},
    "positional enc.": {"sinusoidal": 94.49, "learned": 94.62, "none": 94.17},
    "pooling":         {"mean": 94.49, "max": 94.64, "last": 94.52},
    "activation":      {"relu": 94.49, "leaky_relu": 94.47, "gelu": 94.34,
                        "silu": 93.58, "elu": 93.42},
    # `centered` collapsed to all-No-Fall in the paper (F1 0.00). Recorded as
    # None rather than 0.0: a range including a collapse measures whether the
    # run diverged, not how much the feature choice is worth.
    "input features":  {"full": 94.59, "coords": 94.62, "velocity": 95.24,
                        "centered": None},
    "learning rate":   {5e-05: 93.10, 0.0001: 94.49, 0.0005: 95.47,
                        0.001: 94.87},
    "sequence length": {45: 94.49, 60: 95.89},
    "batch size":      {32: 94.75, 64: 94.49, 128: 93.83},
    "loss function":   {"bce": 94.49, "focal": 95.00},
}

#: The configuration every table varies one parameter away from.
BASELINE_F1 = 94.49

#: The three changes the paper adopted for its final model, as
#: (dimension, field, adopted value). Gains are computed, never typed.
ADOPTED = [("model dimension", "d_model", 256),
           ("learning rate", "lr", 0.0005),
           ("sequence length", "seq_len", 60)]


def ranges() -> dict[str, float]:
    """F1 range per dimension, skipping runs that failed to train."""
    out = {}
    for dim, vals in PAPER.items():
        ok = [v for v in vals.values() if v is not None]
        out[dim] = max(ok) - min(ok)
    return out


def gain(dimension: str, value) -> float:
    """Baseline -> value gain in F1 points, on the paper's own numbers."""
    return PAPER[dimension][value] - BASELINE_F1
