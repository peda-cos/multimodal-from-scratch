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
    "NUM_TOKENS",
    "PROJECTION_HIDDEN",
    "TEXT_HIDDEN",
    "TEXT_SEED",
    "Parameter",
    "MockTextEmbedding",
    "MockVisionEncoder",
    "VISION_HIDDEN",
    "VISION_SEED",
    "VOCAB_SIZE",
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
