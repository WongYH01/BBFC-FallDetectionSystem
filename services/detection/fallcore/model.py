"""The Fall Detector Transformer, plus the parameter-matched baselines.

Transcribed from Section III-B-2 and equations 1-9. The target to hit is
1,208,897 parameters at the paper's final configuration -- `assert_paper_params`
checks it, and that single number validates the whole architecture reading.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import config as cfg
from .data import feature_dim

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani-style fixed encoding, equation 3.

    Registered as a buffer, not a parameter: it moves with .to(device) and gets
    saved in the state dict, but no gradient ever reaches it.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        # An odd d_model leaves the cosine block one column short; slicing div
        # keeps d_model=400-style ablation values from crashing here.
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FallDetectorTransformer(nn.Module):
    """(B, T, D_in) keypoint features -> a single fall logit.

    Pipeline, equations 1-9:
        Linear(D_in -> d) * sqrt(d)  ->  + positional encoding
        ->  L post-LN encoder layers (H heads, d_ff, dropout)
        ->  pool over time  ->  Linear(d -> 32) -> ReLU -> Dropout -> Linear(32 -> 1)

    Returns raw logits; the sigmoid lives inside BCEWithLogitsLoss for numerical
    stability, exactly as the paper describes.
    """

    def __init__(self, config: cfg.ModelConfig | None = None, input_dim: int | None = None):
        super().__init__()
        self.config = config or cfg.ModelConfig()
        c = self.config
        self.input_dim = input_dim if input_dim is not None else feature_dim(c.features)
        self.d_model = c.d_model

        self.embedding = nn.Linear(self.input_dim, c.d_model)

        if c.positional_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(c.d_model)
        elif c.positional_encoding == "learned":
            self.pos_encoder = None
            self.pos_embedding = nn.Parameter(torch.zeros(1, 512, c.d_model))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        elif c.positional_encoding == "none":
            self.pos_encoder = None
        else:
            raise ValueError(f"unknown positional_encoding {c.positional_encoding!r}")

        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation=c.activation if c.activation in ("relu", "gelu") else _act_fn(c.activation),
            batch_first=True,
            # Post-LN: equations 6-7 apply LayerNorm *after* the residual add.
            norm_first=False,
        )
        # The nested-tensor fast path only engages when a key-padding mask is
        # passed, which this model never does (windows are always full length).
        # Disabling it explicitly costs nothing and silences a warning that
        # would otherwise fire on every odd-head or non-ReLU ablation config.
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.n_layers,
                                             enable_nested_tensor=False)

        act = ACTIVATIONS[c.activation]
        self.classifier = nn.Sequential(
            nn.Linear(c.d_model, 32),
            act(),
            nn.Dropout(c.dropout),
            # n_classes is 1 for the paper's single fall logit; widening it is
            # the only architectural difference in the multi-class variant, and
            # it costs 33 parameters per extra class.
            nn.Linear(32, c.n_classes),
        )

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "mean":
            return h.mean(dim=1)          # equation 8
        if self.config.pooling == "max":
            return h.max(dim=1).values
        if self.config.pooling == "last":
            return h[:, -1]
        raise ValueError(f"unknown pooling {self.config.pooling!r}")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding + positional encoding, split out so hooks can reach it."""
        e = self.embedding(x) * math.sqrt(self.d_model)   # equation 1
        if self.pos_encoder is not None:
            e = self.pos_encoder(e)
        elif getattr(self, "pos_embedding", None) is not None:
            e = e + self.pos_embedding[:, : e.size(1)]
        return e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(x))
        return self.classifier(self._pool(h)).squeeze(-1)


def _act_fn(name: str):
    """nn.TransformerEncoderLayer takes a callable for anything but relu/gelu."""
    return ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Parameter-matched baselines (Section IV-A-2, Tables 9 and 21)
# ---------------------------------------------------------------------------
class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM, hidden 192, 1 layer, with the same classifier head."""

    def __init__(self, input_dim: int = 51, hidden: int = 192, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.classifier(h.mean(dim=1)).squeeze(-1)


class CNN1DBaseline(nn.Module):
    """Four conv blocks (128, 256, 512, 256), kernel 3, BN, adaptive avg pool."""

    def __init__(self, input_dim: int = 51, channels=(128, 256, 512, 256),
                 dropout: float = 0.1):
        super().__init__()
        blocks, in_c = [], input_dim
        for out_c in channels:
            blocks += [nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
                       nn.BatchNorm1d(out_c), nn.ReLU()]
            in_c = out_c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_c, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (B, C, T); the datasets emit (B, T, C).
        h = self.features(x.transpose(1, 2))
        return self.classifier(self.pool(h).squeeze(-1)).squeeze(-1)


class MLPBaseline(nn.Module):
    """Flatten the whole window, two hidden layers (168, 64). No temporal model."""

    def __init__(self, input_dim: int = 51, seq_len: int = 60,
                 hidden=(168, 64), dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


PAPER_PARAM_COUNT = 1_208_897


def assert_paper_params(model: nn.Module) -> int:
    """Check the final config against the paper's stated parameter count.

    Worth doing once, loudly: if this matches, the layer sizes, the 32-unit
    classifier bottleneck and the encoder configuration are all right. If it
    does not, something in the architecture transcription is wrong and every
    number downstream is measuring a different model.
    """
    n = count_parameters(model)
    if n != PAPER_PARAM_COUNT:
        raise AssertionError(
            f"parameter count {n:,} != paper's {PAPER_PARAM_COUNT:,} "
            f"(difference {n - PAPER_PARAM_COUNT:+,}). Check ModelConfig."
        )
    return n


def build_model(config: cfg.ModelConfig | None = None,
                input_dim: int | None = None) -> FallDetectorTransformer:
    return FallDetectorTransformer(config, input_dim=input_dim)


def build_baseline(name: str, input_dim: int = 51, seq_len: int = 60) -> nn.Module:
    if name == "lstm":
        return BiLSTMBaseline(input_dim)
    if name == "cnn":
        return CNN1DBaseline(input_dim)
    if name == "mlp":
        return MLPBaseline(input_dim, seq_len)
    raise ValueError(f"unknown baseline {name!r}")
