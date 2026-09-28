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
    "TrainingReport",
    "VISION_HIDDEN",
    "VISION_SEED",
    "VOCAB_SIZE",
    "cls_pool",
    "cosine_alignment_grad",
    "cosine_alignment_loss",
    "cosine_alignment_loss_and_grad",
    "cosine_similarity",
    "gelu",
    "gelu_grad",
    "demo",
    "make_pair",
    "sgd_step",
    "train",
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
_SQRT_2_OVER_PI = sqrt(2.0 / 3.141592653589793)


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
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    return (
        0.5 * (1.0 + tanh_inner)
        + 0.5
        * x
        * (1.0 - tanh_inner**2)
        * _SQRT_2_OVER_PI
        * (1.0 + 3.0 * 0.044715 * x**2)
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


def cls_pool(tokens: np.ndarray) -> np.ndarray:
    """Take the first (CLS) token from ``(B, N, D)`` image tokens."""
    token_batch = np.asarray(tokens, dtype=np.float32)
    if token_batch.ndim != 3:
        raise ValueError(f"expected a 3-D token batch, got {token_batch.shape}")
    if token_batch.shape[0] == 0 or token_batch.shape[1] == 0:
        raise ValueError("token batch must have a batch and a sequence dimension")
    return token_batch[:, 0, :]


def make_pair(
    seed: int,
    vocab_size: int,
    *,
    text_table: MockTextEmbedding | None = None,
    vision_encoder: MockVisionEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize one paired ``(image_tokens, caption_ids)`` sample.

    The image CLS token is a frozen random projection of the caption embedding
    plus small noise. This is a deterministic fixture for the alignment
    mechanism, not a real image-encoder output.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    table = (
        text_table
        if text_table is not None
        else MockTextEmbedding(vocab_size, TEXT_HIDDEN, seed=TEXT_SEED)
    )
    encoder = (
        vision_encoder
        if vision_encoder is not None
        else MockVisionEncoder(
            TEXT_HIDDEN, VISION_HIDDEN, NUM_TOKENS, seed=VISION_SEED
        )
    )
    if table.dim != encoder.text_dim:
        raise ValueError("text table and vision encoder dimensions differ")
    if vocab_size > table.vocab_size:
        raise ValueError("vocab_size exceeds the shared text table")

    rng = np.random.default_rng(seed)
    caption_ids = rng.choice(vocab_size, CAPTION_LEN, replace=False)
    caption = table.caption_embedding(caption_ids)
    image_tokens = encoder.encode(caption, noise_seed=seed)
    return image_tokens, caption_ids


def _as_batch_pair(
    image_emb: np.ndarray, text_emb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_emb, dtype=np.float32)
    text = np.asarray(text_emb, dtype=np.float32)
    if image.ndim == 1:
        image = image[None, :]
    if text.ndim == 1:
        text = text[None, :]
    if image.ndim != 2 or text.ndim != 2:
        raise ValueError("embeddings must be 1-D or 2-D")
    if image.shape != text.shape:
        raise ValueError(
            f"image and text embeddings must share a shape, got "
            f"{image.shape} and {text.shape}"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("embeddings must have a batch and a feature dimension")
    return image, text


def cosine_similarity(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Return the per-pair cosine similarity for matching rows."""
    image, text = _as_batch_pair(image_emb, text_emb)
    image_norm = np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    text_norm = np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    return np.sum((image / image_norm) * (text / text_norm), axis=1)


def cosine_alignment_loss_and_grad(
    image_emb: np.ndarray, text_emb: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the mean ``1 - cos`` loss and its gradient w.r.t. image rows."""
    image, text = _as_batch_pair(image_emb, text_emb)
    image_norm = np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    text_norm = np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    image_hat = image / image_norm
    text_hat = text / text_norm
    cosine = np.sum(image_hat * text_hat, axis=1)
    loss = float(np.mean(1.0 - cosine))
    gradient = (
        -(text_hat - cosine[:, None] * image_hat)
        / image_norm
        / image.shape[0]
    )
    return loss, gradient


def cosine_alignment_loss(image_emb: np.ndarray, text_emb: np.ndarray) -> float:
    """Mean per-pair cosine alignment objective."""
    loss, _ = cosine_alignment_loss_and_grad(image_emb, text_emb)
    return loss


def cosine_alignment_grad(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Gradient of ``cosine_alignment_loss`` with respect to image rows."""
    _, gradient = cosine_alignment_loss_and_grad(image_emb, text_emb)
    return gradient


@dataclass(frozen=True)
class TrainingReport:
    """Full-dataset metrics captured at a training step."""

    step: int
    loss: float
    mean_cosine: float


def train(
    projector: MLPProjector,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    text_table: MockTextEmbedding,
    *,
    steps: int = 200,
    batch_size: int = 8,
    lr: float = 0.1,
    report_every: int = 25,
    verbose: bool = False,
) -> list[TrainingReport]:
    """Train only the projector on cycled synthetic pairs."""
    if not pairs:
        raise ValueError("at least one pair is required")
    if steps <= 0 or batch_size <= 0 or report_every <= 0 or lr <= 0.0:
        raise ValueError("steps, batch_size, report_every, and lr must be positive")

    images = cls_pool(np.stack([image_tokens for image_tokens, _ in pairs]))
    texts = np.stack(
        [text_table.caption_embedding(caption_ids) for _, caption_ids in pairs]
    )
    if images.shape[1] != projector.in_dim:
        raise ValueError(
            f"image token width {images.shape[1]} does not match "
            f"projector input {projector.in_dim}"
        )
    if texts.shape[1] != projector.out_dim:
        raise ValueError(
            f"text embedding width {texts.shape[1]} does not match "
            f"projector output {projector.out_dim}"
        )

    def make_report(step: int) -> TrainingReport:
        projected = projector(images)
        similarities = cosine_similarity(projected, texts)
        loss = float(np.mean(1.0 - similarities))
        mean_cosine = float(np.mean(similarities))
        if verbose:
            print(
                f"step {step:3d} | loss {loss:.4f} | "
                f"mean cosine {mean_cosine:+.4f}"
            )
        return TrainingReport(step=step, loss=loss, mean_cosine=mean_cosine)

    history = [make_report(0)]
    for step in range(1, steps + 1):
        indices = (
            (step - 1) * batch_size + np.arange(batch_size)
        ) % len(pairs)
        projector.zero_grad()
        projected = projector(images[indices])
        _, gradient = cosine_alignment_loss_and_grad(projected, texts[indices])
        projector.backward(gradient)
        sgd_step(projector, lr)
        if step % report_every == 0 or step == steps:
            history.append(make_report(step))
    return history


def demo() -> None:
    """Run the 32-pair, 200-step projection alignment demonstration."""
    table = MockTextEmbedding(VOCAB_SIZE, TEXT_HIDDEN, seed=TEXT_SEED)
    encoder = MockVisionEncoder(
        TEXT_HIDDEN, VISION_HIDDEN, NUM_TOKENS, seed=VISION_SEED
    )
    pairs = [
        make_pair(seed, VOCAB_SIZE, text_table=table, vision_encoder=encoder)
        for seed in range(32)
    ]
    projector = MLPProjector(
        VISION_HIDDEN, PROJECTION_HIDDEN, TEXT_HIDDEN, seed=PROJECTION_SEED
    )
    trainable = sum(
        int(parameter.data.size)
        for parameter in projector.parameters()
        if parameter.requires_grad
    )
    frozen_table = sum(
        int(parameter.data.size) for parameter in table.parameters()
    )
    frozen_encoder = sum(
        int(parameter.data.size) for parameter in encoder.parameters()
    )

    print("projection layer: 768 -> 1024 -> 512")
    print(f"trainable projector parameters: {trainable:,}")
    print(f"frozen text table parameters: {frozen_table:,} (requires_grad=False)")
    print(
        f"frozen vision bridge parameters: {frozen_encoder:,} "
        "(requires_grad=False)"
    )
    print("pairs: 32, steps: 200, batch: 8, lr: 0.1, optimizer: SGD")

    table_before = table.table.copy()
    encoder_before = encoder.projection.data.copy()
    tokens_before = [image_tokens.copy() for image_tokens, _ in pairs]
    projector_before = [
        parameter.data.copy() for parameter in projector.parameters()
    ]

    reports = train(
        projector,
        pairs,
        table,
        steps=200,
        batch_size=8,
        lr=0.1,
        report_every=25,
        verbose=True,
    )

    images = cls_pool(np.stack([image_tokens for image_tokens, _ in pairs]))
    texts = np.stack(
        [table.caption_embedding(caption_ids) for _, caption_ids in pairs]
    )
    final_cosine = cosine_similarity(projector(images), texts)
    print(f"loss: {reports[0].loss:.4f} -> {reports[-1].loss:.4f}")
    print(
        f"mean cosine: {reports[0].mean_cosine:+.4f} -> "
        f"{reports[-1].mean_cosine:+.4f}"
    )
    print(
        "final cosine per pair:",
        np.array2string(final_cosine, precision=2, separator=", "),
    )
    print(f"final mean cosine: {float(np.mean(final_cosine)):+.4f}")
    print(
        "frozen text table unchanged:",
        bool(np.array_equal(table.table, table_before)),
    )
    print(
        "frozen vision bridge unchanged:",
        bool(np.array_equal(encoder.projection.data, encoder_before)),
    )
    print(
        "frozen image tokens unchanged:",
        all(
            np.array_equal(before, after)
            for before, (after, _) in zip(tokens_before, pairs, strict=True)
        ),
    )
    print(
        "projector parameters changed:",
        any(
            not np.array_equal(before, parameter.data)
            for before, parameter in zip(projector_before, projector.parameters(), strict=True)
        ),
    )


if __name__ == "__main__":
    demo()
