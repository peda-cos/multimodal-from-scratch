"""Projection layer that maps frozen image tokens into a frozen text space.

The module keeps the expensive-looking pieces fixed: a deterministic mock
vision encoder produces 197 image tokens from a synthetic paired caption, and a
deterministic mock text table turns caption ids into 512-dimensional vectors.
Only the two-layer MLP projector trains, using a per-pair cosine alignment
loss. Everything is NumPy with a hand-written backward pass so the gradient
path is visible without an autograd dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

__all__ = [
    "CAPTION_LEN",
    "MLPProjector",
    "NUM_TOKENS",
    "PROJECTION_HIDDEN",
    "PROJECTION_SEED",
    "TEXT_HIDDEN",
    "TEXT_SEED",
    "Parameter",
    "MockTextEmbedding",
    "MockVisionEncoder",
    "VISION_HIDDEN",
    "VISION_SEED",
    "VOCAB_SIZE",
    "gelu",
    "gelu_grad",
    "sgd_step",
]

VISION_HIDDEN = 768
TEXT_HIDDEN = 512
PROJECTION_HIDDEN = 1024
NUM_TOKENS = 197
CAPTION_LEN = 6
VOCAB_SIZE = 512
TEXT_SEED = 0
VISION_SEED = 1
PROJECTION_SEED = 0


@dataclass
class Parameter:
    """A named-free parameter tensor with an explicit gradient and freeze flag."""

    data: np.ndarray
    grad: np.ndarray
    requires_grad: bool = True

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class MockTextEmbedding:
    """Frozen embedding table with deterministic initialization.

    The table is the stand-in for a language model's token embedding matrix.
    It has no tokenizer and no corpus; caption ids are sampled directly by
    ``make_pair``. A caption embedding is the mean of its rows.
    """

    requires_grad = False

    def __init__(self, vocab_size: int, dim: int, seed: int = 0) -> None:
        if vocab_size <= 0 or dim <= 0:
            raise ValueError("vocab_size and dim must be positive")
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.dim = dim
        self.seed = seed
        self.table = rng.normal(
            0.0, 1.0 / sqrt(dim), size=(vocab_size, dim)
        ).astype(np.float32)

    def parameters(self) -> list[Parameter]:
        return [
            Parameter(
                self.table,
                np.zeros_like(self.table),
                requires_grad=False,
            )
        ]

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Return token rows for ``(T,)`` or ``(B, T)`` integer ids."""
        indices = np.asarray(ids, dtype=np.int64)
        if indices.ndim not in (1, 2):
            raise ValueError("caption ids must be 1-D or 2-D")
        if indices.shape[-1] == 0:
            raise ValueError("a caption must contain at least one token")
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= self.vocab_size
        ):
            raise ValueError("caption id outside the embedding vocabulary")
        return self.table[indices]

    def caption_embedding(self, ids: np.ndarray) -> np.ndarray:
        """Mean-pool a caption over its token axis."""
        vectors = self.embed(ids)
        return vectors.mean(axis=-2)


class MockVisionEncoder:
    """Frozen deterministic stand-in for a vision encoder's token output.

    It maps a 512-dimensional caption-derived concept into 768-dimensional
    tokens through a fixed random bridge. The first token is the CLS token and
    carries the concept; the remaining tokens add independent noise. This is a
    synthetic fixture, not a pretrained ViT.
    """

    requires_grad = False

    def __init__(
        self,
        text_dim: int = TEXT_HIDDEN,
        vision_hidden: int = VISION_HIDDEN,
        num_tokens: int = NUM_TOKENS,
        seed: int = VISION_SEED,
    ) -> None:
        if text_dim <= 0 or vision_hidden <= 0 or num_tokens <= 0:
            raise ValueError("encoder dimensions and token count must be positive")
        rng = np.random.default_rng(seed)
        self.text_dim = text_dim
        self.vision_hidden = vision_hidden
        self.num_tokens = num_tokens
        self.seed = seed
        self.projection = Parameter(
            rng.normal(
                0.0, 1.0 / sqrt(text_dim), size=(vision_hidden, text_dim)
            ).astype(np.float32),
            np.zeros((vision_hidden, text_dim), dtype=np.float32),
            requires_grad=False,
        )

    def parameters(self) -> list[Parameter]:
        return [self.projection]

    def encode(self, text_embedding: np.ndarray, noise_seed: int = 0) -> np.ndarray:
        """Return ``(num_tokens, vision_hidden)`` tokens for one concept."""
        vector = np.asarray(text_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.text_dim:
            raise ValueError(
                f"expected a ({self.text_dim},) text embedding, got {vector.shape}"
            )
        concept = vector @ self.projection.data.T
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(
            0.0, 1.0, size=(self.num_tokens, self.vision_hidden)
        ).astype(np.float32)
        tokens = concept[None, :] + 0.05 * noise
        tokens[0] = concept + 0.01 * noise[0]
        return tokens.astype(np.float32, copy=False)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU used between the two linear layers."""
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU."""
    scale = np.sqrt(2.0 / np.pi)
    inner = scale * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    return (
        0.5 * (1.0 + tanh_inner)
        + 0.5 * x * (1.0 - tanh_inner**2) * scale * (1.0 + 3.0 * 0.044715 * x**2)
    )


class MLPProjector:
    """Two-layer linear-GELU-linear projector with a manual backward pass."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        seed: int = PROJECTION_SEED,
    ) -> None:
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("projector dimensions must be positive")
        rng = np.random.default_rng(seed)
        bound_1 = 1.0 / sqrt(in_dim)
        bound_2 = 1.0 / sqrt(hidden_dim)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.seed = seed
        self.fc1 = Parameter(
            rng.uniform(-bound_1, bound_1, (hidden_dim, in_dim)).astype(np.float32),
            np.zeros((hidden_dim, in_dim), dtype=np.float32),
        )
        self.b1 = Parameter(
            rng.uniform(-bound_1, bound_1, hidden_dim).astype(np.float32),
            np.zeros(hidden_dim, dtype=np.float32),
        )
        self.fc2 = Parameter(
            rng.uniform(-bound_2, bound_2, (out_dim, hidden_dim)).astype(np.float32),
            np.zeros((out_dim, hidden_dim), dtype=np.float32),
        )
        self.b2 = Parameter(
            rng.uniform(-bound_2, bound_2, out_dim).astype(np.float32),
            np.zeros(out_dim, dtype=np.float32),
        )
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def parameters(self) -> list[Parameter]:
        return [self.fc1, self.b1, self.fc2, self.b2]

    def num_parameters(self) -> int:
        return sum(int(parameter.data.size) for parameter in self.parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project ``(B, in_dim)`` features to ``(B, out_dim)``."""
        features = np.asarray(x, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"expected a 2-D input, got shape {features.shape}")
        if features.shape[0] == 0:
            raise ValueError("projector input batch must not be empty")
        if features.shape[1] != self.in_dim:
            raise ValueError(
                f"expected {self.in_dim} input features, got {features.shape[1]}"
            )
        hidden = features @ self.fc1.data.T + self.b1.data
        activated = gelu(hidden)
        output = activated @ self.fc2.data.T + self.b2.data
        self._cache = (features, hidden, activated)
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagate ``dL/dout`` and return ``dL/dinput``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        features, hidden, activated = self._cache
        gradient = np.asarray(dout, dtype=np.float32)
        expected = (features.shape[0], self.out_dim)
        if gradient.shape != expected:
            raise ValueError(
                f"expected an upstream gradient of shape {expected}, "
                f"got {gradient.shape}"
            )

        self.fc2.grad += gradient.T @ activated
        self.b2.grad += gradient.sum(axis=0)
        d_activated = gradient @ self.fc2.data
        d_hidden = d_activated * gelu_grad(hidden)
        self.fc1.grad += d_hidden.T @ features
        self.b1.grad += d_hidden.sum(axis=0)
        return d_hidden @ self.fc1.data


def sgd_step(projector: MLPProjector, lr: float) -> None:
    """Apply one plain SGD update to every trainable projector parameter."""
    if lr <= 0.0:
        raise ValueError("learning rate must be positive")
    for parameter in projector.parameters():
        if parameter.requires_grad:
            parameter.data -= lr * parameter.grad
