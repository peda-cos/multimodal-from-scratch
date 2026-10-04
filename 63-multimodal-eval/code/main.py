"""Evaluate a small vision-language model on retrieval, VQA, and captioning.

The three evaluation surfaces are primitives over model outputs: R@1/R@5/R@10
from a cosine similarity matrix, exact-match accuracy for VQA, and multi-
reference BLEU-4 for captions. A deterministic synthetic suite is held out by
seed from the training corpus.

The model is the lesson-62 NumPy stack (ViT, projector, text tower, causal
cross-attention decoder) copied into this module so it verifies independently,
plus two readout heads. The caption head reads the projected patch tokens in
parallel; the VQA head fuses a question-token embedding with the queried patch
token. Both heads are trained end-to-end with the contrastive and LM losses.

The corpus is a deterministic synthetic visual codebook, not real image data.
Each caption content token is rendered into a dedicated 2x2 patch, so the
surfaces are learnable in a short demo while remaining a fixture.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import cos, exp, log, pi, sin, sqrt
from typing import NamedTuple

import numpy as np

__all__ = [
    "Adam",
    "Attention",
    "BOS_ID",
    "CAPTION_LEN",
    "CONTENT_LEN",
    "CONTENT_OFFSET",
    "CONTENT_VOCAB",
    "CaptionHead",
    "CaptionSample",
    "CrossAttentionDecoder",
    "DECODER_HEADS",
    "DECODER_HIDDEN",
    "DECODER_LAYERS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CAPTION_WEIGHT",
    "DEFAULT_EVAL_SAMPLES",
    "DEFAULT_EVAL_SEED",
    "DEFAULT_HEAD_LR",
    "DEFAULT_LM_WEIGHT",
    "DEFAULT_LR",
    "DEFAULT_SEED",
    "DEFAULT_STEPS",
    "DEFAULT_VQA_WEIGHT",
    "EOS_ID",
    "Embedding",
    "EvalSuite",
    "CorpusEvaluation",
    "FeedForward",
    "IMAGE_CHANNELS",
    "IMAGE_SIZE",
    "INIT_TEMPERATURE",
    "ImageProjector",
    "JOINT_DIM",
    "LayerNorm",
    "Linear",
    "LossBreakdown",
    "MAX_SEQ_LEN",
    "MAX_TEMPERATURE",
    "MIN_TEMPERATURE",
    "MockCorpus",
    "ModelOutput",
    "MultimodalModel",
    "NUM_PATCHES",
    "PAD_ID",
    "PATCH_SIZE",
    "PROJECTOR_HIDDEN",
    "Parameter",
    "RetrievalPair",
    "SmallViT",
    "StepRecord",
    "TEXT_HIDDEN",
    "TextEncoder",
    "TransformerBlock",
    "VISION_FF",
    "VISION_HEADS",
    "VISION_HIDDEN",
    "VISION_LAYERS",
    "VOCAB_SIZE",
    "VQAHead",
    "VQA_HIDDEN",
    "VQASample",
    "bleu4",
    "build_default_model",
    "build_eval_suite",
    "clip_grad_norm",
    "content_patch_indices",
    "content_token_indices",
    "cosine_similarity_matrix",
    "demo",
    "evaluate",
    "evaluate_corpus",
    "gelu",
    "gelu_grad",
    "info_nce_loss",
    "info_nce_loss_and_grad",
    "lm_loss",
    "lm_loss_and_grad",
    "log_softmax",
    "make_caption",
    "make_mock_corpus",
    "patchify",
    "recall_at_k",
    "render_image",
    "softmax",
    "train",
    "unpatchify",
    "vqa_exact_match",
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 512
CONTENT_VOCAB = 64
CONTENT_OFFSET = 3
CAPTION_LEN = 10
CONTENT_LEN = 6
IMAGE_SIZE = 8
PATCH_SIZE = 2
IMAGE_CHANNELS = 1
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VISION_HIDDEN = 32
VISION_HEADS = 2
VISION_LAYERS = 2
VISION_FF = 64
JOINT_DIM = 32
PROJECTOR_HIDDEN = 64
TEXT_HIDDEN = 32
DECODER_HIDDEN = 64
DECODER_HEADS = 4
DECODER_LAYERS = 2
DECODER_FF = 128
MAX_SEQ_LEN = 10
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 1.0
INIT_TEMPERATURE = 0.07
DEFAULT_STEPS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 3e-3
DEFAULT_LM_WEIGHT = 1.0
DEFAULT_CAPTION_WEIGHT = 1.0
DEFAULT_VQA_WEIGHT = 1.0
DEFAULT_SEED = 62
DEFAULT_EVAL_SEED = 6300
DEFAULT_EVAL_SAMPLES = 50
DEFAULT_HEAD_LR = 3e-2
VQA_HIDDEN = 64


def _float32(value: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite float32 array."""
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def content_patch_indices(content_len: int) -> tuple[int, ...]:
    """Patch indices, in the 4x4 patch grid, used by the content slots."""
    length = _positive_int(content_len, "content_len")
    grid = IMAGE_SIZE // PATCH_SIZE
    if length > 2 * grid:
        raise ValueError(
            f"content_len must be at most {2 * grid}, got {length}"
        )
    return tuple((p // 2) * grid + (p % 2) for p in range(length))


def content_token_indices(content_len: int) -> tuple[int, ...]:
    """Content patch positions in the CLS-prefixed vision token sequence."""
    return tuple(index + 1 for index in content_patch_indices(content_len))


def render_image(
    tokens: Sequence[int] | np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Render caption tokens into their dedicated 2x2 image patches."""
    if len(tokens) == 0:
        raise ValueError("tokens must be non-empty")
    if len(tokens) > 2 * (IMAGE_SIZE // PATCH_SIZE):
        raise ValueError("too many content slots for the image")
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for position, token in enumerate(tokens):
        token_id = int(token)
        if not CONTENT_OFFSET <= token_id < CONTENT_OFFSET + CONTENT_VOCAB:
            raise ValueError(f"token id {token_id} is outside the content range")
        row = (position // 2) * PATCH_SIZE
        column = (position % 2) * PATCH_SIZE
        image[row : row + PATCH_SIZE, column : column + PATCH_SIZE] = (
            _content_code(token_id).reshape(PATCH_SIZE, PATCH_SIZE)
        )
    image += rng.normal(0.0, 0.02, (IMAGE_SIZE, IMAGE_SIZE)).astype(
        np.float32
    )
    return image[None, :, :]


def make_caption(
    tokens: Sequence[int] | np.ndarray, max_len: int
) -> np.ndarray:
    """Create BOS + content + EOS + PAD ids of length ``max_len``."""
    length = _positive_int(max_len, "max_len")
    if len(tokens) == 0:
        raise ValueError("tokens must be non-empty")
    if len(tokens) + 2 > length:
        raise ValueError(
            f"tokens of length {len(tokens)} do not fit in max_len {length}"
        )
    caption = np.full(length, PAD_ID, dtype=np.int64)
    caption[0] = BOS_ID
    caption[1 : 1 + len(tokens)] = np.asarray(tokens, dtype=np.int64)
    caption[1 + len(tokens)] = EOS_ID
    return caption


def _strip_special_tokens(ids: Sequence[int]) -> list[int]:
    return [
        int(token)
        for token in ids
        if int(token) not in (PAD_ID, BOS_ID, EOS_ID)
    ]


def recall_at_k(
    sim_matrix: np.ndarray,
    k: int,
    *,
    direction: str = "text_to_image",
) -> float:
    """Fraction of queries whose correct match lands in the top ``k``.

    ``text_to_image`` ranks columns for each caption row; ``image_to_text``
    ranks rows for each image column. One float is returned per call.
    """
    matrix = np.asarray(sim_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sim_matrix must be a square 2-D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("sim_matrix must be non-empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sim_matrix must contain only finite values")
    if not isinstance(k, (int, np.integer)) or int(k) <= 0:
        raise ValueError("k must be a positive integer")
    top_k = int(k)
    size = matrix.shape[0]
    if top_k > size:
        raise ValueError(f"k must be <= {size}, got {top_k}")
    if direction == "text_to_image":
        ranked = matrix
    elif direction == "image_to_text":
        ranked = matrix.T
    else:
        raise ValueError(
            "direction must be 'text_to_image' or 'image_to_text'"
        )
    top = np.argsort(-ranked, axis=1, kind="stable")[:, :top_k]
    labels = np.arange(size)[:, None]
    hits = np.any(top == labels, axis=1)
    return float(np.mean(hits))


def vqa_exact_match(
    predictions: Sequence[int], references: Sequence[int]
) -> float:
    """Mean one-bit equality over paired answer token ids."""
    predicted = np.asarray(predictions)
    reference = np.asarray(references)
    if predicted.ndim != 1 or reference.ndim != 1:
        raise ValueError("predictions and references must be 1-D")
    if predicted.size == 0 or reference.size == 0:
        raise ValueError("predictions and references must be non-empty")
    if predicted.shape != reference.shape:
        raise ValueError(
            "predictions and references must have equal length"
        )
    if not np.issubdtype(predicted.dtype, np.integer) or not np.issubdtype(
        reference.dtype, np.integer
    ):
        raise ValueError("answer ids must be integers")
    return float(np.mean(predicted == reference))


def _ngram_counts(
    tokens: Sequence[int], n: int
) -> Counter[tuple[int, ...]]:
    return Counter(
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    )


def bleu4(
    generated: Sequence[int],
    references: Sequence[Sequence[int]],
    smoothing: bool = True,
) -> float:
    """Sentence BLEU-4 with multi-reference clipping and add-one smoothing.

    The capstone calls the add-one rule "method 1"; in Chen and Cherry's
    numbering add-one is smoothing 2, while smoothing 1 replaces zero counts
    with epsilon. This implementation follows the add-one rule the spec states.
    A completely disjoint hypothesis returns 0.0, even with smoothing.
    """
    gen = tuple(int(token) for token in generated)
    if not references:
        raise ValueError("references must be non-empty")
    refs: list[tuple[int, ...]] = []
    for reference in references:
        ref = tuple(int(token) for token in reference)
        if not ref:
            raise ValueError("references must not contain empty sequences")
        refs.append(ref)
    if not gen:
        return 0.0
    if gen in refs:
        return 1.0

    reference_length = min(
        (abs(len(ref) - len(gen)), len(ref)) for ref in refs
    )[1]
    precisions: list[float] = []
    total_matches = 0
    for order in range(1, 5):
        generated_counts = _ngram_counts(gen, order)
        max_reference: Counter[tuple[int, ...]] = Counter()
        for ref in refs:
            for ngram, count in _ngram_counts(ref, order).items():
                if count > max_reference[ngram]:
                    max_reference[ngram] = count
        matches = sum(
            min(count, max_reference[ngram])
            for ngram, count in generated_counts.items()
        )
        total = max(0, len(gen) - order + 1)
        total_matches += matches
        if total == 0:
            precision = 0.0
        elif matches == 0:
            precision = 1.0 / (total + 1.0) if smoothing else 0.0
        else:
            precision = matches / total
        precisions.append(precision)

    if total_matches == 0 or any(p <= 0.0 for p in precisions):
        return 0.0
    brevity = (
        1.0
        if len(gen) >= reference_length
        else exp(1.0 - reference_length / len(gen))
    )
    score = brevity * exp(
        sum(log(precision) for precision in precisions) / 4.0
    )
    return float(min(1.0, max(0.0, score)))


def cosine_similarity_matrix(
    row_emb: np.ndarray, column_emb: np.ndarray
) -> np.ndarray:
    """Row-normalized cosine similarities, rows scored against columns."""
    rows = np.asarray(row_emb, dtype=np.float64)
    columns = np.asarray(column_emb, dtype=np.float64)
    if rows.ndim != 2 or columns.ndim != 2 or rows.shape != columns.shape:
        raise ValueError(
            "row_emb and column_emb must be equal-shape 2-D arrays"
        )
    if rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("embeddings must be non-empty")
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(columns)):
        raise ValueError("embeddings must contain only finite values")
    row_norm = np.linalg.norm(rows, axis=1, keepdims=True)
    column_norm = np.linalg.norm(columns, axis=1, keepdims=True)
    if np.any(row_norm <= 0.0) or np.any(column_norm <= 0.0):
        raise ValueError("embeddings must not contain all-zero rows")
    return (rows / row_norm) @ (columns / column_norm).T


class RetrievalPair(NamedTuple):
    """One image-caption pair in the retrieval pool."""

    image: np.ndarray
    caption_ids: np.ndarray


class VQASample(NamedTuple):
    """One synthetic visual question and its answer token."""

    image: np.ndarray
    question_ids: np.ndarray
    answer_id: int


class CaptionSample(NamedTuple):
    """One image with multiple reference captions."""

    image: np.ndarray
    reference_ids: list[np.ndarray]


@dataclass(frozen=True)
class EvalSuite:
    """Deterministic held-out retrieval, VQA, and captioning lists."""

    pairs: list[RetrievalPair]
    vqa: list[VQASample]
    caps: list[CaptionSample]
    seed: int
    vocab_size: int
    max_len: int
    content_len: int

    @property
    def question_offset(self) -> int:
        """First token id in the synthetic VQA question vocabulary."""
        return self.vocab_size - self.content_len

    def __iter__(
        self,
    ) -> Iterator[
        list[RetrievalPair] | list[VQASample] | list[CaptionSample]
    ]:
        yield self.pairs
        yield self.vqa
        yield self.caps


def build_eval_suite(
    seed: int, n_samples: int, vocab_size: int, max_len: int
) -> EvalSuite:
    """Build deterministic held-out retrieval, VQA, and caption lists."""
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    samples = _positive_int(n_samples, "n_samples")
    vocab = _positive_int(vocab_size, "vocab_size")
    length = _positive_int(max_len, "max_len")
    if not 3 <= length <= CAPTION_LEN:
        raise ValueError(
            f"max_len must be in [3, {CAPTION_LEN}], got {length}"
        )
    minimum_vocab = CONTENT_OFFSET + CONTENT_VOCAB + CONTENT_LEN
    if vocab < minimum_vocab:
        raise ValueError(
            f"vocab_size must be at least {minimum_vocab}, got {vocab}"
        )
    content_len = min(CONTENT_LEN, length - 2)
    question_offset = vocab - content_len
    rng = np.random.default_rng(int(seed))
    pairs: list[RetrievalPair] = []
    vqa: list[VQASample] = []
    caps: list[CaptionSample] = []
    for index in range(samples):
        tokens = (
            rng.choice(CONTENT_VOCAB, size=content_len, replace=False)
            .astype(np.int64)
            + CONTENT_OFFSET
        )
        image = render_image(tokens, rng)
        canonical = make_caption(tokens, length)
        pairs.append(RetrievalPair(image=image, caption_ids=canonical))

        position = index % content_len
        question_ids = np.array(
            [BOS_ID, question_offset + position], dtype=np.int64
        )
        vqa.append(
            VQASample(
                image=image,
                question_ids=question_ids,
                answer_id=int(tokens[position]),
            )
        )

        references = [
            make_caption(tokens, length),
            make_caption(tokens[::-1], length),
            make_caption(np.roll(tokens, 1), length),
        ]
        caps.append(CaptionSample(image=image, reference_ids=references))
    return EvalSuite(
        pairs=pairs,
        vqa=vqa,
        caps=caps,
        seed=int(seed),
        vocab_size=vocab,
        max_len=length,
        content_len=content_len,
    )


class Parameter:
    """A float32 tensor with an accumulated gradient."""

    def __init__(self, name: str, data: np.ndarray) -> None:
        self.name = name
        self.data = np.array(data, dtype=np.float32, copy=True)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def numel(self) -> int:
        return int(self.data.size)

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, shape={self.data.shape})"


class Linear:
    """Affine projection ``y = x @ weight.T + bias``."""

    def __init__(
        self, in_features: int, out_features: int, *, seed: int = 0
    ) -> None:
        self.in_features = _positive_int(in_features, "in_features")
        self.out_features = _positive_int(out_features, "out_features")
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        bound = 1.0 / sqrt(self.in_features)
        self.weight = Parameter(
            f"linear.{self.seed}.weight",
            rng.uniform(
                -bound, bound, (self.out_features, self.in_features)
            ),
        )
        self.bias = Parameter(
            f"linear.{self.seed}.bias", np.zeros(self.out_features)
        )
        self._input: np.ndarray | None = None
        self._flat_input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        array = _float32(x, "x")
        if array.ndim < 1 or array.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected last dimension {self.in_features}, "
                f"got shape {array.shape}"
            )
        self._input = array
        self._flat_input = array.reshape(-1, self.in_features)
        output = array @ self.weight.data.T + self.bias.data
        return output.astype(np.float32, copy=False)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if self._input is None or self._flat_input is None:
            raise RuntimeError("Linear.backward called before forward")
        grad = _float32(doutput, "doutput")
        expected = self._input.shape[:-1] + (self.out_features,)
        if grad.shape != expected:
            raise ValueError(
                f"Linear backward expected shape {expected}, got {grad.shape}"
            )
        flat_grad = grad.reshape(-1, self.out_features)
        self.weight.grad += flat_grad.T @ self._flat_input
        self.bias.grad += flat_grad.sum(axis=0)
        input_grad = grad @ self.weight.data
        return input_grad.reshape(self._input.shape).astype(
            np.float32, copy=False
        )

    def parameters(self) -> tuple[Parameter, ...]:
        return (self.weight, self.bias)

    def num_parameters(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class LayerNorm:
    """Layer normalization over the last axis with affine parameters."""

    def __init__(self, hidden: int, *, eps: float = 1e-5) -> None:
        self.hidden = _positive_int(hidden, "hidden")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.weight = Parameter("layer_norm.weight", np.ones(hidden))
        self.bias = Parameter("layer_norm.bias", np.zeros(hidden))
        self._input: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None
        self._normalized: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        array = _float32(x, "x")
        if array.ndim < 1 or array.shape[-1] != self.hidden:
            raise ValueError(
                f"LayerNorm expected last dimension {self.hidden}, "
                f"got shape {array.shape}"
            )
        mean = array.mean(axis=-1, keepdims=True)
        centered = array - mean
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(variance + np.float32(self.eps))
        normalized = centered * inv_std
        self._input = array
        self._mean = mean
        self._var = variance
        self._inv_std = inv_std
        self._normalized = normalized
        return (
            normalized * self.weight.data + self.bias.data
        ).astype(np.float32, copy=False)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if (
            self._input is None
            or self._mean is None
            or self._var is None
            or self._inv_std is None
            or self._normalized is None
        ):
            raise RuntimeError("LayerNorm.backward called before forward")
        grad = _float32(doutput, "doutput")
        if grad.shape != self._input.shape:
            raise ValueError(
                f"LayerNorm backward expected shape {self._input.shape}, "
                f"got {grad.shape}"
            )
        axes = tuple(range(grad.ndim - 1))
        self.weight.grad += (grad * self._normalized).sum(axis=axes)
        self.bias.grad += grad.sum(axis=axes)
        dnormalized = grad * self.weight.data
        n = self._input.shape[-1]
        dvar = (
            dnormalized
            * (self._input - self._mean)
            * np.float32(-0.5)
            * (self._var + np.float32(self.eps)) ** np.float32(-1.5)
        ).sum(axis=-1, keepdims=True)
        dmean = (
            dnormalized * -self._inv_std
        ).sum(axis=-1, keepdims=True) + dvar * (
            np.float32(-2.0 / n)
            * (self._input - self._mean).sum(axis=-1, keepdims=True)
        )
        dinput = (
            dnormalized * self._inv_std
            + dvar * np.float32(2.0 / n) * (self._input - self._mean)
            + dmean * np.float32(1.0 / n)
        )
        return dinput.astype(np.float32, copy=False)

    def parameters(self) -> tuple[Parameter, ...]:
        return (self.weight, self.bias)

    def num_parameters(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Embedding:
    """Lookup table over integer token ids."""

    def __init__(
        self, vocab_size: int, dim: int, *, seed: int = 0
    ) -> None:
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        self.dim = _positive_int(dim, "dim")
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.table = Parameter(
            f"embedding.{self.seed}.table",
            rng.normal(0.0, 0.02, (self.vocab_size, self.dim)),
        )
        self._ids: np.ndarray | None = None

    def _validate_ids(self, ids: np.ndarray) -> np.ndarray:
        array = np.asarray(ids)
        if array.ndim < 1 or array.size == 0:
            raise ValueError("ids must be a non-empty integer array")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"ids must be integers, got dtype {array.dtype}")
        indices = array.astype(np.int64, copy=False)
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocab_size:
            raise ValueError(
                f"token id outside [0, {self.vocab_size - 1}]"
            )
        return indices

    def forward(self, ids: np.ndarray) -> np.ndarray:
        indices = self._validate_ids(ids)
        self._ids = indices
        return self.table.data[indices].astype(np.float32, copy=False)

    def backward(self, doutput: np.ndarray) -> None:
        if self._ids is None:
            raise RuntimeError("Embedding.backward called before forward")
        grad = _float32(doutput, "doutput")
        expected = self._ids.shape + (self.dim,)
        if grad.shape != expected:
            raise ValueError(
                f"Embedding backward expected shape {expected}, got {grad.shape}"
            )
        np.add.at(self.table.grad, self._ids, grad)

    def parameters(self) -> tuple[Parameter, ...]:
        return (self.table,)

    def num_parameters(self) -> int:
        return self.table.numel()

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        return self.forward(ids)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU."""
    values = _float32(x, "x")
    coefficient = np.float32(sqrt(2.0 / pi))
    inner = coefficient * (
        values + np.float32(0.044715) * values**3
    )
    return (
        np.float32(0.5)
        * values
        * (np.float32(1.0) + np.tanh(inner))
    ).astype(np.float32, copy=False)


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU."""
    values = _float32(x, "x")
    coefficient = np.float32(sqrt(2.0 / pi))
    inner = coefficient * (
        values + np.float32(0.044715) * values**3
    )
    tanh_inner = np.tanh(inner)
    derivative = (
        np.float32(0.5) * (np.float32(1.0) + tanh_inner)
        + np.float32(0.5)
        * values
        * (np.float32(1.0) - tanh_inner * tanh_inner)
        * coefficient
        * (
            np.float32(1.0)
            + np.float32(3.0 * 0.044715) * values * values
        )
    )
    return derivative.astype(np.float32, copy=False)


def _log_softmax64(values: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("log_softmax input must have at least one axis")
    if np.any(np.isnan(array)) or np.any(array == np.inf):
        raise ValueError("log_softmax input contains NaN or +inf")
    maximum = np.max(array, axis=axis, keepdims=True)
    if not np.all(np.isfinite(maximum)):
        raise ValueError("log_softmax input has a row with no finite entry")
    shifted = array - maximum
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )


def log_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    return _log_softmax64(values, axis).astype(np.float64, copy=False)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    return np.exp(_log_softmax64(values, axis))


def _normalize_rows(
    values: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty rows and columns")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} must not contain an all-zero row")
    return array / norms, norms


def _info_nce_inputs(
    image_emb: np.ndarray, text_emb: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image_emb, dtype=np.float64)
    text = np.asarray(text_emb, dtype=np.float64)
    if image.ndim != 2 or text.ndim != 2:
        raise ValueError("image_emb and text_emb must be 2-D arrays")
    if image.shape != text.shape:
        raise ValueError(
            f"image_emb and text_emb shapes differ: {image.shape} vs {text.shape}"
        )
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    image_hat, image_norm = _normalize_rows(image, "image_emb")
    text_hat, text_norm = _normalize_rows(text, "text_emb")
    return image_hat, text_hat, image_norm, text_norm, np.asarray(
        temperature, dtype=np.float64
    )


def info_nce_loss(
    image_emb: np.ndarray, text_emb: np.ndarray, temperature: float
) -> float:
    """Bidirectional CLIP-style InfoNCE loss."""
    loss, _, _, _ = info_nce_loss_and_grad(
        image_emb, text_emb, temperature
    )
    return loss


def info_nce_loss_and_grad(
    image_emb: np.ndarray, text_emb: np.ndarray, temperature: float
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """InfoNCE loss plus gradients for both embeddings and temperature."""
    image_hat, text_hat, image_norm, text_norm, tau = _info_nce_inputs(
        image_emb, text_emb, temperature
    )
    n = image_hat.shape[0]
    logits = (image_hat @ text_hat.T) / tau
    row_log_probs = _log_softmax64(logits, axis=1)
    col_log_probs = _log_softmax64(logits, axis=0)
    labels = np.arange(n)
    loss = -0.5 * float(
        row_log_probs[labels, labels].mean()
        + col_log_probs[labels, labels].mean()
    )

    row_probs = np.exp(row_log_probs)
    col_probs = np.exp(col_log_probs)
    identity = np.eye(n, dtype=np.float64)
    dlogits = 0.5 * (
        (row_probs - identity) / n + (col_probs - identity) / n
    )
    d_image_hat = dlogits @ text_hat / tau
    d_text_hat = dlogits.T @ image_hat / tau
    d_temperature = float(-np.sum(dlogits * logits) / tau)

    d_image = (
        d_image_hat
        - image_hat
        * np.sum(d_image_hat * image_hat, axis=1, keepdims=True)
    ) / image_norm
    d_text = (
        d_text_hat
        - text_hat * np.sum(d_text_hat * text_hat, axis=1, keepdims=True)
    ) / text_norm

    input_dtype = np.asarray(image_emb).dtype
    return (
        loss,
        d_image.astype(input_dtype),
        d_text.astype(input_dtype),
        d_temperature,
    )


def _validate_lm_inputs(
    logits: np.ndarray,
    target_ids: np.ndarray,
    padding_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    logits_array = np.asarray(logits)
    if logits_array.ndim != 3:
        raise ValueError(
            f"logits must be a 3-D (batch, time, vocab) array, "
            f"got {logits_array.shape}"
        )
    if logits_array.shape[0] == 0 or logits_array.shape[1] == 0:
        raise ValueError("logits must have non-empty batch and time axes")
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("logits must contain only finite values")
    targets = np.asarray(target_ids)
    if targets.ndim != 2:
        raise ValueError(
            f"target_ids must be 2-D, got shape {targets.shape}"
        )
    if targets.shape != logits_array.shape[:2]:
        raise ValueError(
            f"target_ids shape {targets.shape} does not match logits "
            f"batch/time {logits_array.shape[:2]}"
        )
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(
            f"target_ids must be integers, got dtype {targets.dtype}"
        )
    vocab_size = logits_array.shape[2]
    if not isinstance(padding_id, (int, np.integer)):
        raise ValueError("padding_id must be an integer")
    pad = int(padding_id)
    if pad < 0 or pad >= vocab_size:
        raise ValueError(
            f"padding_id {pad} outside vocabulary [0, {vocab_size - 1}]"
        )
    active = targets != pad
    if not bool(active.any()):
        raise ValueError("no non-padding targets")
    active_targets = targets[active]
    if int(active_targets.min()) < 0 or int(active_targets.max()) >= vocab_size:
        raise ValueError(
            f"target id outside [0, {vocab_size - 1}]"
        )
    return logits_array, targets.astype(np.int64), active, pad


def lm_loss(
    logits: np.ndarray, target_ids: np.ndarray, padding_id: int
) -> float:
    """Masked next-token cross-entropy."""
    loss, _ = lm_loss_and_grad(logits, target_ids, padding_id)
    return loss


def lm_loss_and_grad(
    logits: np.ndarray, target_ids: np.ndarray, padding_id: int
) -> tuple[float, np.ndarray]:
    """Masked language-modeling loss and its gradient w.r.t. logits."""
    logits_array, targets, active, _ = _validate_lm_inputs(
        logits, target_ids, padding_id
    )
    batch, time, _ = logits_array.shape
    log_probs = _log_softmax64(logits_array, axis=-1)
    batch_index = np.arange(batch)[:, None]
    time_index = np.arange(time)[None, :]
    gathered = log_probs[batch_index, time_index, targets]
    loss = -float(gathered[active].mean())

    dlogits = np.exp(log_probs)
    dlogits[batch_index, time_index, targets] -= 1.0
    dlogits /= float(np.count_nonzero(active))
    dlogits[~active] = 0.0
    return loss, dlogits.astype(logits_array.dtype, copy=False)


def patchify(images: np.ndarray, patch_size: int) -> np.ndarray:
    """Split ``(B, C, H, W)`` into row-major flattened patches."""
    array = _float32(images, "images")
    patch = _positive_int(patch_size, "patch_size")
    if array.ndim != 4:
        raise ValueError(
            f"images must be a 4-D (B, C, H, W) array, got {array.shape}"
        )
    batch, channels, height, width = array.shape
    if height % patch != 0 or width % patch != 0:
        raise ValueError(
            f"image size {(height, width)} is not divisible by "
            f"patch size {patch}"
        )
    rows = height // patch
    cols = width // patch
    reshaped = array.reshape(
        batch, channels, rows, patch, cols, patch
    )
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)
    return transposed.reshape(
        batch, rows * cols, channels * patch * patch
    )


def unpatchify(
    patches: np.ndarray,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> np.ndarray:
    """Inverse of :func:`patchify`."""
    array = _float32(patches, "patches")
    channel_count = _positive_int(channels, "channels")
    image_height = _positive_int(height, "height")
    image_width = _positive_int(width, "width")
    patch = _positive_int(patch_size, "patch_size")
    if array.ndim != 3:
        raise ValueError(
            f"patches must be a 3-D array, got {array.shape}"
        )
    if image_height % patch != 0 or image_width % patch != 0:
        raise ValueError("image size is not divisible by patch size")
    rows = image_height // patch
    cols = image_width // patch
    expected = (array.shape[0], rows * cols, channel_count * patch * patch)
    if array.shape != expected:
        raise ValueError(
            f"patches shape {array.shape} does not match {expected}"
        )
    reshaped = array.reshape(
        array.shape[0], rows, cols, channel_count, patch, patch
    )
    transposed = reshaped.transpose(0, 3, 1, 4, 2, 5)
    return transposed.reshape(
        array.shape[0], channel_count, image_height, image_width
    )


def _split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    array = _float32(x, "x")
    if array.ndim != 3:
        raise ValueError(f"expected 3-D tensor, got {array.shape}")
    batch, sequence, hidden = array.shape
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) not divisible by heads ({heads})")
    return array.reshape(
        batch, sequence, heads, hidden // heads
    ).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    array = _float32(x, "x")
    if array.ndim != 4:
        raise ValueError(f"expected 4-D tensor, got {array.shape}")
    batch, heads, sequence, head_dim = array.shape
    return array.transpose(0, 2, 1, 3).reshape(
        batch, sequence, heads * head_dim
    )


class Attention:
    """Multi-head self- or cross-attention."""

    def __init__(self, hidden: int, heads: int, *, seed: int = 0) -> None:
        self.hidden = _positive_int(hidden, "hidden")
        self.heads = _positive_int(heads, "heads")
        if self.hidden % self.heads != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by heads "
                f"({self.heads})"
            )
        self.head_dim = self.hidden // self.heads
        self.scale = np.float32(1.0 / sqrt(self.head_dim))
        self.q_proj = Linear(self.hidden, self.hidden, seed=seed)
        self.k_proj = Linear(self.hidden, self.hidden, seed=seed + 1)
        self.v_proj = Linear(self.hidden, self.hidden, seed=seed + 2)
        self.out_proj = Linear(self.hidden, self.hidden, seed=seed + 3)
        self._query_input: np.ndarray | None = None
        self._kv_input: np.ndarray | None = None
        self._query_heads: np.ndarray | None = None
        self._key_heads: np.ndarray | None = None
        self._value_heads: np.ndarray | None = None
        self._probabilities: np.ndarray | None = None
        self._causal = False

    def forward(
        self,
        query_input: np.ndarray,
        kv_input: np.ndarray | None = None,
        *,
        causal: bool = False,
    ) -> np.ndarray:
        query = _float32(query_input, "query_input")
        if query.ndim != 3:
            raise ValueError(
                f"query_input must be 3-D, got {query.shape}"
            )
        if query.shape[2] != self.hidden:
            raise ValueError(
                f"query_input last dimension must be {self.hidden}, "
                f"got {query.shape[2]}"
            )
        if kv_input is None:
            key_value = query
        else:
            key_value = _float32(kv_input, "kv_input")
            if key_value.ndim != 3:
                raise ValueError(
                    f"kv_input must be 3-D, got {key_value.shape}"
                )
        if key_value.shape[0] != query.shape[0]:
            raise ValueError(
                "query_input and kv_input must share a batch dimension"
            )
        if key_value.shape[2] != self.hidden:
            raise ValueError(
                f"kv_input last dimension must be {self.hidden}, "
                f"got {key_value.shape[2]}"
            )
        if causal and query.shape[1] != key_value.shape[1]:
            raise ValueError("causal attention requires equal sequence lengths")

        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key_value)
        v = self.v_proj.forward(key_value)
        q_heads = _split_heads(q, self.heads)
        k_heads = _split_heads(k, self.heads)
        v_heads = _split_heads(v, self.heads)
        scores = (q_heads @ k_heads.transpose(0, 1, 3, 2)) * self.scale
        if causal:
            length = query.shape[1]
            mask = np.triu(
                np.ones((length, length), dtype=bool), k=1
            )
            scores = np.where(mask[None, None, :, :], -np.inf, scores)
        probabilities = softmax(scores, axis=-1).astype(np.float32)
        context = probabilities @ v_heads
        merged = _merge_heads(context)
        output = self.out_proj.forward(merged)

        self._query_input = query
        self._kv_input = key_value
        self._query_heads = q_heads
        self._key_heads = k_heads
        self._value_heads = v_heads
        self._probabilities = probabilities
        self._causal = causal
        return output

    def backward(
        self, doutput: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._query_input is None
            or self._kv_input is None
            or self._query_heads is None
            or self._key_heads is None
            or self._value_heads is None
            or self._probabilities is None
        ):
            raise RuntimeError("Attention.backward called before forward")
        grad = _float32(doutput, "doutput")
        merged_grad = self.out_proj.backward(grad)
        grad_heads = _split_heads(merged_grad, self.heads)
        dprobabilities = grad_heads @ self._value_heads.transpose(
            0, 1, 3, 2
        )
        dvalues = (
            self._probabilities.transpose(0, 1, 3, 2) @ grad_heads
        )
        dprobabilities = dprobabilities.astype(np.float32)
        probabilities = self._probabilities
        dscores = probabilities * (
            dprobabilities
            - np.sum(
                dprobabilities * probabilities, axis=-1, keepdims=True
            )
        )
        if self._causal:
            length = self._query_input.shape[1]
            mask = np.triu(
                np.ones((length, length), dtype=bool), k=1
            )
            dscores = np.where(mask[None, None, :, :], 0.0, dscores)
        dquery_heads = (dscores @ self._key_heads) * self.scale
        dkey_heads = (
            dscores.transpose(0, 1, 3, 2) @ self._query_heads
        ) * self.scale
        dquery = _merge_heads(dquery_heads)
        dkey = _merge_heads(dkey_heads)
        dvalue = _merge_heads(dvalues)
        dquery_input = self.q_proj.backward(dquery)
        dkv_input = self.k_proj.backward(dkey) + self.v_proj.backward(
            dvalue
        )
        return dquery_input, dkv_input

    def parameters(self) -> tuple[Parameter, ...]:
        return (
            *self.q_proj.parameters(),
            *self.k_proj.parameters(),
            *self.v_proj.parameters(),
            *self.out_proj.parameters(),
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(
        self,
        query_input: np.ndarray,
        kv_input: np.ndarray | None = None,
        *,
        causal: bool = False,
    ) -> np.ndarray:
        return self.forward(query_input, kv_input, causal=causal)


class FeedForward:
    """Two-layer GELU feed-forward network."""

    def __init__(
        self, hidden: int, ff_hidden: int, *, seed: int = 0
    ) -> None:
        self.hidden = _positive_int(hidden, "hidden")
        self.ff_hidden = _positive_int(ff_hidden, "ff_hidden")
        self.fc1 = Linear(self.hidden, self.ff_hidden, seed=seed)
        self.fc2 = Linear(self.ff_hidden, self.hidden, seed=seed + 1)
        self._pre_activation: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_activation = self.fc1.forward(x)
        return self.fc2.forward(gelu(self._pre_activation))

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if self._pre_activation is None:
            raise RuntimeError("FeedForward.backward called before forward")
        dactivation = self.fc2.backward(doutput)
        dpre = gelu_grad(self._pre_activation) * dactivation
        return self.fc1.backward(dpre)

    def parameters(self) -> tuple[Parameter, ...]:
        return (*self.fc1.parameters(), *self.fc2.parameters())

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TransformerBlock:
    """Pre-LN block with optional cross-attention."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        ff_hidden: int,
        *,
        cross_attention: bool,
        causal: bool,
        seed: int = 0,
    ) -> None:
        self.hidden = _positive_int(hidden, "hidden")
        self.heads = _positive_int(heads, "heads")
        self.ff_hidden = _positive_int(ff_hidden, "ff_hidden")
        self.cross_attention = bool(cross_attention)
        self.causal = bool(causal)
        self.ln1 = LayerNorm(self.hidden)
        self.self_attn = Attention(self.hidden, self.heads, seed=seed)
        if self.cross_attention:
            self.ln2: LayerNorm | None = LayerNorm(self.hidden)
            self.cross_attn: Attention | None = Attention(
                self.hidden, self.heads, seed=seed + 4
            )
            self.ln3: LayerNorm | None = LayerNorm(self.hidden)
            self.ffn = FeedForward(
                self.hidden, self.ff_hidden, seed=seed + 8
            )
        else:
            self.ln2 = LayerNorm(self.hidden)
            self.cross_attn = None
            self.ln3 = None
            self.ffn = FeedForward(
                self.hidden, self.ff_hidden, seed=seed + 4
            )

    def forward(
        self,
        x: np.ndarray,
        memory: np.ndarray | None = None,
        *,
        causal: bool | None = None,
    ) -> np.ndarray:
        values = _float32(x, "x")
        if values.ndim != 3 or values.shape[2] != self.hidden:
            raise ValueError(
                f"block input must be (batch, time, {self.hidden}), "
                f"got {values.shape}"
            )
        use_causal = self.causal if causal is None else bool(causal)
        h1 = values + self.self_attn.forward(
            self.ln1.forward(values), causal=use_causal
        )
        if self.cross_attn is not None:
            if self.ln2 is None or self.ln3 is None:
                raise RuntimeError("cross-attention block is misconfigured")
            if memory is None:
                raise ValueError(
                    "memory is required for a cross-attention block"
                )
            memory_values = _float32(memory, "memory")
            if memory_values.ndim != 3:
                raise ValueError(
                    f"memory must be 3-D, got {memory_values.shape}"
                )
            if memory_values.shape[0] != values.shape[0]:
                raise ValueError(
                    "memory and text batch dimensions must match"
                )
            if memory_values.shape[2] != self.hidden:
                raise ValueError(
                    f"memory last dimension must be {self.hidden}, "
                    f"got {memory_values.shape[2]}"
                )
            h2 = h1 + self.cross_attn.forward(
                self.ln2.forward(h1), memory_values, causal=False
            )
            h3 = h2 + self.ffn.forward(self.ln3.forward(h2))
            return h3
        if memory is not None:
            raise ValueError(
                "memory was provided to a block without cross-attention"
            )
        if self.ln2 is None:
            raise RuntimeError("non-cross block is missing its second norm")
        return h1 + self.ffn.forward(self.ln2.forward(h1))

    def backward(
        self, doutput: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        grad = _float32(doutput, "doutput")
        if self.cross_attn is not None:
            if self.ln2 is None or self.ln3 is None:
                raise RuntimeError("cross-attention block is misconfigured")
            d_ffn_input = self.ffn.backward(grad)
            d_ln3 = self.ln3.backward(d_ffn_input)
            d_h2 = grad + d_ln3
            d_cross_input, d_memory = self.cross_attn.backward(d_h2)
            d_ln2 = self.ln2.backward(d_cross_input)
            d_h1 = d_h2 + d_ln2
            d_query, d_key_value = self.self_attn.backward(d_h1)
            d_ln1 = self.ln1.backward(d_query + d_key_value)
            return d_h1 + d_ln1, d_memory
        if self.ln2 is None:
            raise RuntimeError("non-cross block is missing its second norm")
        d_ffn_input = self.ffn.backward(grad)
        d_ln2 = self.ln2.backward(d_ffn_input)
        d_h1 = grad + d_ln2
        d_query, d_key_value = self.self_attn.backward(d_h1)
        d_ln1 = self.ln1.backward(d_query + d_key_value)
        return d_h1 + d_ln1, None

    def parameters(self) -> tuple[Parameter, ...]:
        parameters: list[Parameter] = list(self.ln1.parameters())
        parameters.extend(self.self_attn.parameters())
        if self.cross_attn is not None:
            if self.ln2 is not None:
                parameters.extend(self.ln2.parameters())
            parameters.extend(self.cross_attn.parameters())
            if self.ln3 is not None:
                parameters.extend(self.ln3.parameters())
        elif self.ln2 is not None:
            parameters.extend(self.ln2.parameters())
        parameters.extend(self.ffn.parameters())
        return tuple(parameters)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SmallViT:
    """Small ViT encoder producing CLS plus patch tokens."""

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        channels: int = IMAGE_CHANNELS,
        hidden: int = VISION_HIDDEN,
        heads: int = VISION_HEADS,
        layers: int = VISION_LAYERS,
        ff_hidden: int = VISION_FF,
        *,
        seed: int = 0,
    ) -> None:
        self.image_size = _positive_int(image_size, "image_size")
        self.patch_size = _positive_int(patch_size, "patch_size")
        self.channels = _positive_int(channels, "channels")
        self.hidden = _positive_int(hidden, "hidden")
        self.heads = _positive_int(heads, "heads")
        self.layers = _positive_int(layers, "layers")
        self.ff_hidden = _positive_int(ff_hidden, "ff_hidden")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )
        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2
        self.num_tokens = self.num_patches + 1
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.patch_embed = Linear(
            self.channels * self.patch_size * self.patch_size,
            self.hidden,
            seed=self.seed + 1,
        )
        self.cls_token = Parameter(
            "vit.cls_token",
            rng.normal(0.0, 0.02, (1, 1, self.hidden)),
        )
        self.position = Parameter(
            "vit.position",
            rng.normal(0.0, 0.02, (1, self.num_tokens, self.hidden)),
        )
        self.blocks = tuple(
            TransformerBlock(
                self.hidden,
                self.heads,
                self.ff_hidden,
                cross_attention=False,
                causal=False,
                seed=self.seed + 10 + 4 * index,
            )
            for index in range(self.layers)
        )
        self.norm = LayerNorm(self.hidden)
        self._batch_size: int | None = None

    def forward(self, images: np.ndarray) -> np.ndarray:
        values = _float32(images, "images")
        expected = (
            values.shape[0],
            self.channels,
            self.image_size,
            self.image_size,
        )
        if values.ndim != 4 or values.shape != expected:
            raise ValueError(
                f"images must have shape {expected}, got {values.shape}"
            )
        batch = values.shape[0]
        patches = patchify(values, self.patch_size)
        embedded = self.patch_embed.forward(patches)
        cls = np.broadcast_to(
            self.cls_token.data, (batch, 1, self.hidden)
        )
        tokens = (
            np.concatenate([cls, embedded], axis=1)
            + self.position.data
        )
        for block in self.blocks:
            tokens = block.forward(tokens)
        tokens = self.norm.forward(tokens)
        self._batch_size = batch
        return tokens

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if self._batch_size is None:
            raise RuntimeError("SmallViT.backward called before forward")
        grad = _float32(doutput, "doutput")
        expected = (self._batch_size, self.num_tokens, self.hidden)
        if grad.shape != expected:
            raise ValueError(
                f"SmallViT backward expected shape {expected}, "
                f"got {grad.shape}"
            )
        d_tokens = self.norm.backward(grad)
        for block in reversed(self.blocks):
            d_tokens, d_memory = block.backward(d_tokens)
            if d_memory is not None:
                raise RuntimeError(
                    "vision block unexpectedly returned a memory gradient"
                )
        self.position.grad += d_tokens.sum(axis=0, keepdims=True)
        self.cls_token.grad += d_tokens[:, :1, :].sum(
            axis=0, keepdims=True
        )
        d_patches = self.patch_embed.backward(d_tokens[:, 1:, :])
        return unpatchify(
            d_patches,
            self.channels,
            self.image_size,
            self.image_size,
            self.patch_size,
        )

    def parameters(self) -> tuple[Parameter, ...]:
        parameters: list[Parameter] = list(self.patch_embed.parameters())
        parameters.append(self.cls_token)
        parameters.append(self.position)
        for block in self.blocks:
            parameters.extend(block.parameters())
        parameters.extend(self.norm.parameters())
        return tuple(parameters)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.forward(images)


class ImageProjector:
    """Token-wise MLP projecting vision features into the joint space."""

    def __init__(
        self,
        input_dim: int = VISION_HIDDEN,
        hidden: int = PROJECTOR_HIDDEN,
        output_dim: int = JOINT_DIM,
        *,
        seed: int = 0,
    ) -> None:
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.hidden = _positive_int(hidden, "hidden")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.fc1 = Linear(self.input_dim, self.hidden, seed=seed)
        self.fc2 = Linear(self.hidden, self.output_dim, seed=seed + 1)
        self._pre_activation: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_activation = self.fc1.forward(x)
        return self.fc2.forward(gelu(self._pre_activation))

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if self._pre_activation is None:
            raise RuntimeError(
                "ImageProjector.backward called before forward"
            )
        dactivation = self.fc2.backward(doutput)
        dpre = gelu_grad(self._pre_activation) * dactivation
        return self.fc1.backward(dpre)

    def parameters(self) -> tuple[Parameter, ...]:
        return (*self.fc1.parameters(), *self.fc2.parameters())

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TextEncoder:
    """Mean-pooled token embedding tower for contrastive alignment."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden: int = TEXT_HIDDEN,
        output_dim: int = JOINT_DIM,
        *,
        seed: int = 0,
    ) -> None:
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        self.hidden = _positive_int(hidden, "hidden")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.embedding = Embedding(
            self.vocab_size, self.hidden, seed=seed
        )
        self.project = Linear(self.hidden, self.output_dim, seed=seed + 1)
        self._mask: np.ndarray | None = None
        self._counts: np.ndarray | None = None

    def forward(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask)
        if mask_array.dtype != np.dtype(bool):
            raise ValueError(
                f"mask must be boolean, got dtype {mask_array.dtype}"
            )
        if mask_array.ndim != 2:
            raise ValueError(
                f"mask must be 2-D, got shape {mask_array.shape}"
            )
        if mask_array.shape != np.asarray(ids).shape:
            raise ValueError(
                "mask shape must match ids shape"
            )
        if not bool(mask_array.any(axis=1).all()):
            raise ValueError("each caption must contain a non-padding token")
        embedded = self.embedding.forward(ids)
        counts = mask_array.sum(axis=1, keepdims=True).astype(np.float32)
        pooled = (
            embedded * mask_array[..., None]
        ).sum(axis=1) / counts
        self._mask = mask_array
        self._counts = counts
        return self.project.forward(pooled)

    def backward(self, doutput: np.ndarray) -> None:
        if self._mask is None or self._counts is None:
            raise RuntimeError("TextEncoder.backward called before forward")
        d_pooled = self.project.backward(doutput)
        d_embedded = (
            d_pooled[:, None, :] / self._counts[..., None]
        ) * self._mask[..., None]
        self.embedding.backward(d_embedded)

    def parameters(self) -> tuple[Parameter, ...]:
        return (*self.embedding.parameters(), *self.project.parameters())

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.forward(ids, mask)


class CrossAttentionDecoder:
    """Causal decoder that cross-attends to projected image tokens."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden: int = DECODER_HIDDEN,
        heads: int = DECODER_HEADS,
        layers: int = DECODER_LAYERS,
        ff_hidden: int = DECODER_FF,
        max_seq_len: int = MAX_SEQ_LEN,
        memory_dim: int = JOINT_DIM,
        *,
        seed: int = 0,
    ) -> None:
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        self.hidden = _positive_int(hidden, "hidden")
        self.heads = _positive_int(heads, "heads")
        self.layers = _positive_int(layers, "layers")
        self.ff_hidden = _positive_int(ff_hidden, "ff_hidden")
        self.max_seq_len = _positive_int(max_seq_len, "max_seq_len")
        self.memory_dim = _positive_int(memory_dim, "memory_dim")
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.token_embedding = Embedding(
            self.vocab_size, self.hidden, seed=self.seed
        )
        self.position = Parameter(
            "decoder.position",
            rng.normal(0.0, 0.02, (1, self.max_seq_len, self.hidden)),
        )
        self.memory_proj = Linear(
            self.memory_dim, self.hidden, seed=self.seed + 2
        )
        self.blocks = tuple(
            TransformerBlock(
                self.hidden,
                self.heads,
                self.ff_hidden,
                cross_attention=True,
                causal=True,
                seed=self.seed + 10 + 4 * index,
            )
            for index in range(self.layers)
        )
        self.norm = LayerNorm(self.hidden)
        self.lm_head = Linear(
            self.hidden, self.vocab_size, seed=self.seed + 3
        )
        self._memory: np.ndarray | None = None
        self._time = 0

    def forward(
        self, input_ids: np.ndarray, memory: np.ndarray
    ) -> np.ndarray:
        ids = np.asarray(input_ids)
        if ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2-D, got shape {ids.shape}"
            )
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                f"input_ids must be integers, got dtype {ids.dtype}"
            )
        time = ids.shape[1]
        if time < 1 or time > self.max_seq_len:
            raise ValueError(
                f"input length must be in [1, {self.max_seq_len}], "
                f"got {time}"
            )
        memory_values = _float32(memory, "memory")
        if memory_values.ndim != 3:
            raise ValueError(
                f"memory must be 3-D, got {memory_values.shape}"
            )
        if memory_values.shape[0] != ids.shape[0]:
            raise ValueError(
                "input_ids and memory batch dimensions must match"
            )
        if memory_values.shape[2] != self.memory_dim:
            raise ValueError(
                f"memory last dimension must be {self.memory_dim}, "
                f"got {memory_values.shape[2]}"
            )
        embedded = self.token_embedding.forward(ids)
        tokens = embedded + self.position.data[:, :time, :]
        memory_tokens = self.memory_proj.forward(memory_values)
        for block in self.blocks:
            tokens = block.forward(
                tokens, memory_tokens, causal=True
            )
        tokens = self.norm.forward(tokens)
        logits = self.lm_head.forward(tokens)
        self._memory = memory_tokens
        self._time = time
        return logits

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        if self._memory is None:
            raise RuntimeError(
                "CrossAttentionDecoder.backward called before forward"
            )
        grad = _float32(dlogits, "dlogits")
        expected = (self._memory.shape[0], self._time, self.vocab_size)
        if grad.shape != expected:
            raise ValueError(
                f"decoder backward expected shape {expected}, "
                f"got {grad.shape}"
            )
        d_tokens = self.lm_head.backward(grad)
        d_tokens = self.norm.backward(d_tokens)
        d_memory = np.zeros_like(self._memory)
        for block in reversed(self.blocks):
            d_tokens, d_block_memory = block.backward(d_tokens)
            if d_block_memory is None:
                raise RuntimeError(
                    "decoder block unexpectedly returned no memory gradient"
                )
            d_memory += d_block_memory
        self.position.grad[:, : self._time, :] += d_tokens.sum(
            axis=0, keepdims=True
        )
        self.token_embedding.backward(d_tokens)
        return self.memory_proj.backward(d_memory)

    def parameters(self) -> tuple[Parameter, ...]:
        parameters: list[Parameter] = list(
            self.token_embedding.parameters()
        )
        parameters.append(self.position)
        parameters.extend(self.memory_proj.parameters())
        for block in self.blocks:
            parameters.extend(block.parameters())
        parameters.extend(self.norm.parameters())
        parameters.extend(self.lm_head.parameters())
        return tuple(parameters)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def __call__(
        self, input_ids: np.ndarray, memory: np.ndarray
    ) -> np.ndarray:
        return self.forward(input_ids, memory)


@dataclass(frozen=True)
class MockCorpus:
    """Deterministic synthetic image-caption pairs."""

    images: np.ndarray
    captions: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        if self.images.ndim != 4:
            raise ValueError("images must be (N, C, H, W)")
        if self.images.shape[1:] != (
            IMAGE_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                "images must have shape (N, 1, 8, 8), got "
                f"{self.images.shape}"
            )
        if self.captions.ndim != 2 or self.captions.shape[1] != CAPTION_LEN:
            raise ValueError(
                f"captions must be (N, {CAPTION_LEN}), got "
                f"{self.captions.shape}"
            )
        if self.images.shape[0] != self.captions.shape[0]:
            raise ValueError("images and captions must have equal length")
        if not np.issubdtype(self.captions.dtype, np.integer):
            raise ValueError("captions must contain integers")
        if not np.all(np.isfinite(self.images)):
            raise ValueError("images must be finite")
        if not np.all(self.captions[:, 0] == BOS_ID):
            raise ValueError("every caption must start with BOS")
        eos_position = 1 + CONTENT_LEN
        if not np.all(self.captions[:, eos_position] == EOS_ID):
            raise ValueError(
                f"every caption must have EOS at position {eos_position}"
            )
        if not np.all(self.captions[:, eos_position + 1 :] == PAD_ID):
            raise ValueError("tokens after EOS must be padding")
        content = self.captions[:, 1:eos_position]
        if np.any(content < CONTENT_OFFSET) or np.any(
            content >= CONTENT_OFFSET + CONTENT_VOCAB
        ):
            raise ValueError("content ids outside the active vocabulary")

    def __len__(self) -> int:
        return int(self.images.shape[0])


def _content_code(token_id: int) -> np.ndarray:
    """Fixed 4-pixel visual code for one active caption token."""
    index = int(token_id) - CONTENT_OFFSET
    angle = 2.0 * pi * index / CONTENT_VOCAB
    return np.array(
        [
            cos(angle),
            sin(angle),
            cos(2.0 * angle),
            sin(2.0 * angle),
        ],
        dtype=np.float32,
    )


def make_mock_corpus(seed: int, n_pairs: int = 200) -> MockCorpus:
    """Create deterministic paired images and caption ids."""
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(n_pairs, (int, np.integer)) or int(n_pairs) < 1:
        raise ValueError("n_pairs must be a positive integer")
    rng = np.random.default_rng(int(seed))
    images = np.empty(
        (int(n_pairs), IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )
    captions = np.empty(
        (int(n_pairs), CAPTION_LEN), dtype=np.int64
    )
    for index in range(int(n_pairs)):
        tokens = (
            rng.choice(CONTENT_VOCAB, size=CONTENT_LEN, replace=False)
            .astype(np.int64)
            + CONTENT_OFFSET
        )
        captions[index] = make_caption(tokens, CAPTION_LEN)
        images[index, 0] = render_image(tokens, rng)[0]
    return MockCorpus(images=images, captions=captions, seed=int(seed))


class CaptionHead:
    """Parallel MLP readout from projected patch tokens to caption tokens."""

    def __init__(
        self,
        input_dim: int = JOINT_DIM,
        vocab_size: int = CONTENT_VOCAB,
        hidden: int = VQA_HIDDEN,
        *,
        seed: int = 0,
    ) -> None:
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        self.hidden = _positive_int(hidden, "hidden")
        self.fc1 = Linear(self.input_dim, self.hidden, seed=seed)
        self.fc2 = Linear(self.hidden, self.vocab_size, seed=seed + 1)
        self._pre_activation: np.ndarray | None = None

    def forward(self, patch_tokens: np.ndarray) -> np.ndarray:
        self._pre_activation = self.fc1.forward(patch_tokens)
        return self.fc2.forward(gelu(self._pre_activation))

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        if self._pre_activation is None:
            raise RuntimeError("CaptionHead.backward called before forward")
        dactivation = self.fc2.backward(dlogits)
        dpre = gelu_grad(self._pre_activation) * dactivation
        return self.fc1.backward(dpre)

    def parameters(self) -> tuple[Parameter, ...]:
        return (*self.fc1.parameters(), *self.fc2.parameters())

    def num_parameters(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters()
        )

    def __call__(self, patch_tokens: np.ndarray) -> np.ndarray:
        return self.forward(patch_tokens)


class VQAHead:
    """Fuse one image patch token with one learned question-slot embedding."""

    def __init__(
        self,
        joint_dim: int = JOINT_DIM,
        hidden: int = VQA_HIDDEN,
        answer_vocab: int = CONTENT_VOCAB,
        question_vocab: int = CONTENT_LEN,
        *,
        seed: int = 0,
    ) -> None:
        self.joint_dim = _positive_int(joint_dim, "joint_dim")
        self.hidden = _positive_int(hidden, "hidden")
        self.answer_vocab = _positive_int(answer_vocab, "answer_vocab")
        self.question_vocab = _positive_int(
            question_vocab, "question_vocab"
        )
        self.question_embedding = Embedding(
            self.question_vocab, self.joint_dim, seed=seed + 2
        )
        self.fusion = Linear(2 * self.joint_dim, self.hidden, seed=seed)
        self.classifier = Linear(
            self.hidden, self.answer_vocab, seed=seed + 1
        )
        self._pre_activation: np.ndarray | None = None

    def forward(
        self, patch: np.ndarray, question_slots: np.ndarray
    ) -> np.ndarray:
        patch_values = _float32(patch, "patch")
        slots = np.asarray(question_slots)
        if patch_values.ndim != 2 or patch_values.shape[1] != self.joint_dim:
            raise ValueError(
                f"patch must be (B, {self.joint_dim}), "
                f"got {patch_values.shape}"
            )
        if slots.ndim != 1 or slots.shape[0] != patch_values.shape[0]:
            raise ValueError(
                "question_slots must be 1-D with one id per patch"
            )
        question_values = self.question_embedding.forward(slots)
        self._pre_activation = self.fusion.forward(
            np.concatenate([patch_values, question_values], axis=1)
        )
        return self.classifier.forward(gelu(self._pre_activation))

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        if self._pre_activation is None:
            raise RuntimeError("VQAHead.backward called before forward")
        dactivation = self.classifier.backward(dlogits)
        dpre = gelu_grad(self._pre_activation) * dactivation
        dconcat = self.fusion.backward(dpre)
        self.question_embedding.backward(
            np.ascontiguousarray(dconcat[:, self.joint_dim :])
        )
        return np.ascontiguousarray(dconcat[:, : self.joint_dim])

    def parameters(self) -> tuple[Parameter, ...]:
        return (
            *self.question_embedding.parameters(),
            *self.fusion.parameters(),
            *self.classifier.parameters(),
        )

    def num_parameters(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters()
        )

    def __call__(
        self, patch: np.ndarray, question_slots: np.ndarray
    ) -> np.ndarray:
        return self.forward(patch, question_slots)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    loss, _ = _cross_entropy_and_grad(logits, targets)
    return loss


def _cross_entropy_and_grad(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)
    if values.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    vocab = values.shape[-1]
    if vocab <= 0:
        raise ValueError("logits must have a non-empty last dimension")
    if labels.shape != values.shape[:-1]:
        raise ValueError(
            "targets shape must match logits shape without the last axis"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("targets must be integers")
    flat_logits = values.reshape(-1, vocab)
    flat_targets = labels.reshape(-1).astype(np.int64)
    if flat_targets.size == 0:
        raise ValueError("targets must be non-empty")
    if int(flat_targets.min()) < 0 or int(flat_targets.max()) >= vocab:
        raise ValueError("target id outside the logits vocabulary")
    log_probs = log_softmax(flat_logits, axis=-1)
    positions = np.arange(flat_targets.size)
    loss = -float(np.mean(log_probs[positions, flat_targets]))
    gradient = softmax(flat_logits, axis=-1)
    gradient[positions, flat_targets] -= 1.0
    gradient /= float(flat_targets.size)
    return loss, gradient.reshape(values.shape).astype(np.float32)


@dataclass(frozen=True)
class ModelOutput:
    """Forward-pass tensors needed by the losses and readout heads."""

    image_emb: np.ndarray
    text_emb: np.ndarray
    logits: np.ndarray
    targets: np.ndarray
    temperature: float
    projected: np.ndarray
    vision_tokens: np.ndarray


@dataclass(frozen=True)
class LossBreakdown:
    """Weighted training-loss components for one batch."""

    contrastive: float
    lm: float
    caption: float
    vqa: float
    total: float


@dataclass(frozen=True)
class _HeadTerms:
    """Caption/VQA loss values and gradients for one forward pass."""

    caption_loss: float
    vqa_loss: float
    caption_grad: np.ndarray
    vqa_grad: np.ndarray
    token_indices: np.ndarray
    content_len: int


def _weight(value: float, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return weight


class MultimodalModel:
    """Vision encoder, projector, text tower, decoder, and eval heads."""

    def __init__(
        self,
        vision_encoder: SmallViT,
        image_projector: ImageProjector,
        text_encoder: TextEncoder,
        decoder: CrossAttentionDecoder,
        caption_head: CaptionHead,
        vqa_head: VQAHead,
        *,
        init_temperature: float = INIT_TEMPERATURE,
    ) -> None:
        if image_projector.input_dim != vision_encoder.hidden:
            raise ValueError(
                "projector input_dim must equal vision hidden size"
            )
        if image_projector.output_dim != text_encoder.output_dim:
            raise ValueError(
                "projector and text encoder output dimensions must match"
            )
        if image_projector.output_dim != decoder.memory_dim:
            raise ValueError(
                "projector output_dim must equal decoder memory_dim"
            )
        if caption_head.input_dim != vision_encoder.hidden:
            raise ValueError(
                "caption head input_dim must equal vision hidden size"
            )
        if vqa_head.joint_dim != vision_encoder.hidden:
            raise ValueError(
                "vqa head joint_dim must equal vision hidden size"
            )
        if caption_head.vocab_size != CONTENT_VOCAB:
            raise ValueError(
                f"caption head must have {CONTENT_VOCAB} answer tokens"
            )
        if vqa_head.answer_vocab != CONTENT_VOCAB:
            raise ValueError(
                f"vqa head must have {CONTENT_VOCAB} answer tokens"
            )
        if vqa_head.question_vocab < CONTENT_LEN:
            raise ValueError(
                f"vqa head needs at least {CONTENT_LEN} question slots"
            )
        if not (
            MIN_TEMPERATURE <= init_temperature <= MAX_TEMPERATURE
        ):
            raise ValueError(
                f"init_temperature must be in "
                f"[{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]"
            )
        self.vision_encoder = vision_encoder
        self.image_projector = image_projector
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.caption_head = caption_head
        self.vqa_head = vqa_head
        self.log_temperature = Parameter(
            "temperature.log",
            np.array(log(float(init_temperature)), dtype=np.float32),
        )
        self._projected: np.ndarray | None = None

    def temperature(self) -> tuple[float, float]:
        """Return the projected temperature and d tau / d log_tau."""
        low = log(MIN_TEMPERATURE)
        high = log(MAX_TEMPERATURE)
        raw = float(self.log_temperature.data.item())
        clipped = min(max(raw, low), high)
        if clipped != raw:
            self.log_temperature.data[...] = np.float32(clipped)
        tau = float(np.exp(np.float32(clipped)))
        return tau, tau

    def encode_images(
        self, images: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (vision tokens, projected tokens) for a batch of images."""
        vision_tokens = self.vision_encoder.forward(images)
        projected = self.image_projector.forward(vision_tokens)
        return vision_tokens, projected

    def encode_image_tokens(self, images: np.ndarray) -> np.ndarray:
        """Return projected vision tokens without running the decoder."""
        return self.encode_images(images)[1]

    def forward(
        self, images: np.ndarray, captions: np.ndarray
    ) -> ModelOutput:
        image_values = _float32(images, "images")
        caption_values = np.asarray(captions)
        if image_values.ndim != 4:
            raise ValueError(
                f"images must be 4-D, got shape {image_values.shape}"
            )
        if caption_values.ndim != 2:
            raise ValueError(
                f"captions must be 2-D, got shape {caption_values.shape}"
            )
        if not np.issubdtype(caption_values.dtype, np.integer):
            raise ValueError(
                f"captions must be integers, got dtype "
                f"{caption_values.dtype}"
            )
        if image_values.shape[0] != caption_values.shape[0]:
            raise ValueError(
                "images and captions batch dimensions must match"
            )
        if caption_values.shape[1] < 2:
            raise ValueError("captions must contain at least two tokens")
        if caption_values.shape[1] > self.decoder.max_seq_len + 1:
            raise ValueError(
                "caption length exceeds the decoder position table"
            )
        vision_tokens, projected = self.encode_images(image_values)
        image_emb = projected[:, 0, :]
        text_mask = caption_values != PAD_ID
        text_emb = self.text_encoder.forward(
            caption_values, text_mask
        )
        logits = self.decoder.forward(
            caption_values[:, :-1], projected
        )
        tau, _ = self.temperature()
        self._projected = projected
        return ModelOutput(
            image_emb=image_emb,
            text_emb=text_emb,
            logits=logits,
            targets=caption_values[:, 1:],
            temperature=tau,
            projected=projected,
            vision_tokens=vision_tokens,
        )

    def _head_terms(self, output: ModelOutput) -> _HeadTerms:
        """Compute caption and VQA losses plus their logit gradients."""
        targets = np.asarray(output.targets)
        if targets.ndim != 2 or targets.shape[0] != output.vision_tokens.shape[0]:
            raise ValueError(
                "targets must be a 2-D batch array matching vision tokens"
            )
        eos_positions = np.argmax(targets == EOS_ID, axis=1)
        if not bool(np.all(np.any(targets == EOS_ID, axis=1))):
            raise ValueError("every caption target needs an EOS token")
        if not bool(np.all(eos_positions == eos_positions[0])):
            raise ValueError(
                "all captions in a batch must have equal content length"
            )
        content_len = min(CONTENT_LEN, int(eos_positions[0]))
        if content_len < 1:
            raise ValueError("captions must contain at least one content token")
        if content_len > self.vqa_head.question_vocab:
            raise ValueError(
                "caption content length exceeds the VQA question vocabulary"
            )
        token_indices = np.asarray(
            content_token_indices(content_len), dtype=np.int64
        )
        patch_tokens = output.vision_tokens[:, token_indices, :]
        content_targets = targets[:, :content_len] - CONTENT_OFFSET
        caption_logits = self.caption_head.forward(patch_tokens)
        caption_loss, caption_grad = _cross_entropy_and_grad(
            caption_logits, content_targets
        )

        batch = patch_tokens.shape[0]
        flat_patches = patch_tokens.reshape(
            batch * content_len, patch_tokens.shape[2]
        )
        question_slots = np.tile(
            np.arange(content_len, dtype=np.int64), batch
        )
        vqa_logits = self.vqa_head.forward(
            flat_patches, question_slots
        )
        vqa_loss, vqa_grad = _cross_entropy_and_grad(
            vqa_logits, content_targets.reshape(-1)
        )
        return _HeadTerms(
            caption_loss=caption_loss,
            vqa_loss=vqa_loss,
            caption_grad=caption_grad,
            vqa_grad=vqa_grad,
            token_indices=token_indices,
            content_len=content_len,
        )

    def losses(
        self,
        output: ModelOutput,
        *,
        lm_weight: float = DEFAULT_LM_WEIGHT,
        caption_weight: float = DEFAULT_CAPTION_WEIGHT,
        vqa_weight: float = DEFAULT_VQA_WEIGHT,
    ) -> LossBreakdown:
        lm_scale = _weight(lm_weight, "lm_weight")
        caption_scale = _weight(caption_weight, "caption_weight")
        vqa_scale = _weight(vqa_weight, "vqa_weight")
        contrastive = info_nce_loss(
            output.image_emb, output.text_emb, output.temperature
        )
        language_model = lm_loss(
            output.logits, output.targets, PAD_ID
        )
        terms = self._head_terms(output)
        total = (
            contrastive
            + lm_scale * language_model
            + caption_scale * terms.caption_loss
            + vqa_scale * terms.vqa_loss
        )
        return LossBreakdown(
            contrastive=contrastive,
            lm=language_model,
            caption=terms.caption_loss,
            vqa=terms.vqa_loss,
            total=total,
        )

    def backward(
        self,
        output: ModelOutput,
        *,
        lm_weight: float = DEFAULT_LM_WEIGHT,
        caption_weight: float = DEFAULT_CAPTION_WEIGHT,
        vqa_weight: float = DEFAULT_VQA_WEIGHT,
    ) -> LossBreakdown:
        if self._projected is None:
            raise RuntimeError(
                "MultimodalModel.backward called before forward"
            )
        lm_scale = _weight(lm_weight, "lm_weight")
        caption_scale = _weight(caption_weight, "caption_weight")
        vqa_scale = _weight(vqa_weight, "vqa_weight")
        contrastive, d_image, d_text, d_temperature = (
            info_nce_loss_and_grad(
                output.image_emb, output.text_emb, output.temperature
            )
        )
        language_model, d_logits = lm_loss_and_grad(
            output.logits, output.targets, PAD_ID
        )
        terms = self._head_terms(output)
        total = (
            contrastive
            + lm_scale * language_model
            + caption_scale * terms.caption_loss
            + vqa_scale * terms.vqa_loss
        )

        self.zero_grad()
        d_projected = np.zeros_like(self._projected)
        d_vision_tokens = np.zeros_like(output.vision_tokens)
        if caption_scale != 0.0:
            d_caption = self.caption_head.backward(
                terms.caption_grad * np.float32(caption_scale)
            )
            d_vision_tokens[:, terms.token_indices, :] = (
                d_vision_tokens[:, terms.token_indices, :] + d_caption
            )
        if vqa_scale != 0.0:
            d_vqa = self.vqa_head.backward(
                terms.vqa_grad * np.float32(vqa_scale)
            )
            d_vqa = d_vqa.reshape(
                output.vision_tokens.shape[0],
                terms.content_len,
                output.vision_tokens.shape[2],
            )
            d_vision_tokens[:, terms.token_indices, :] = (
                d_vision_tokens[:, terms.token_indices, :] + d_vqa
            )
        if lm_scale != 0.0:
            d_projected += self.decoder.backward(
                d_logits * np.float32(lm_scale)
            )
        d_projected[:, 0, :] += d_image
        d_vision = (
            self.image_projector.backward(d_projected) + d_vision_tokens
        )
        self.vision_encoder.backward(d_vision)
        self.text_encoder.backward(d_text)
        _, dtau = self.temperature()
        self.log_temperature.grad += np.float32(
            d_temperature * dtau
        )
        return LossBreakdown(
            contrastive=contrastive,
            lm=language_model,
            caption=terms.caption_loss,
            vqa=terms.vqa_loss,
            total=total,
        )

    def loss_and_backward(
        self,
        images: np.ndarray,
        captions: np.ndarray,
        *,
        lm_weight: float = DEFAULT_LM_WEIGHT,
        caption_weight: float = DEFAULT_CAPTION_WEIGHT,
        vqa_weight: float = DEFAULT_VQA_WEIGHT,
    ) -> LossBreakdown:
        output = self.forward(images, captions)
        return self.backward(
            output,
            lm_weight=lm_weight,
            caption_weight=caption_weight,
            vqa_weight=vqa_weight,
        )

    def base_parameters(self) -> tuple[Parameter, ...]:
        """Parameters trained by the original contrastive and LM objectives."""
        parameters: list[Parameter] = list(
            self.vision_encoder.parameters()
        )
        parameters.extend(self.image_projector.parameters())
        parameters.extend(self.text_encoder.parameters())
        parameters.extend(self.decoder.parameters())
        parameters.append(self.log_temperature)
        return tuple(parameters)

    def head_parameters(self) -> tuple[Parameter, ...]:
        """Freshly initialized retrieval-independent evaluation heads."""
        parameters: list[Parameter] = list(
            self.caption_head.parameters()
        )
        parameters.extend(self.vqa_head.parameters())
        return tuple(parameters)

    def parameters(self) -> tuple[Parameter, ...]:
        return (*self.base_parameters(), *self.head_parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_default_model(seed: int = DEFAULT_SEED) -> MultimodalModel:
    """Build the demo-sized model with deterministic seeds."""
    return MultimodalModel(
        vision_encoder=SmallViT(seed=seed),
        image_projector=ImageProjector(seed=seed + 1),
        text_encoder=TextEncoder(seed=seed + 2),
        decoder=CrossAttentionDecoder(seed=seed + 3),
        caption_head=CaptionHead(
            input_dim=VISION_HIDDEN,
            vocab_size=CONTENT_VOCAB,
            seed=seed + 4,
        ),
        vqa_head=VQAHead(
            joint_dim=VISION_HIDDEN,
            hidden=VQA_HIDDEN,
            answer_vocab=CONTENT_VOCAB,
            question_vocab=CONTENT_LEN,
            seed=seed + 5,
        ),
        init_temperature=INIT_TEMPERATURE,
    )


class Adam:
    """Adam optimizer over a list of parameters."""

    def __init__(
        self,
        parameters: Sequence[Parameter],
        *,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("Adam requires at least one parameter")
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.first_moment = [
            np.zeros_like(parameter.data) for parameter in self.parameters
        ]
        self.second_moment = [
            np.zeros_like(parameter.data) for parameter in self.parameters
        ]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        for parameter, first, second in zip(
            self.parameters, self.first_moment, self.second_moment
        ):
            gradient = parameter.grad
            first[...] = (
                self.beta1 * first + (1.0 - self.beta1) * gradient
            )
            second[...] = (
                self.beta2 * second
                + (1.0 - self.beta2) * gradient * gradient
            )
            first_hat = first / correction1
            second_hat = second / correction2
            parameter.data -= (
                self.lr
                * first_hat
                / (np.sqrt(second_hat) + self.eps)
            )


def clip_grad_norm(
    parameters: Sequence[Parameter], max_norm: float = 1.0
) -> float:
    """Clip gradients by global L2 norm and return the pre-clip norm."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    total_squared = 0.0
    for parameter in parameters:
        norm = float(np.linalg.norm(parameter.grad))
        total_squared += norm * norm
    total_norm = sqrt(total_squared)
    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm
        for parameter in parameters:
            parameter.grad *= np.float32(scale)
    return total_norm


def similarity_accuracy(
    image_emb: np.ndarray, text_emb: np.ndarray
) -> tuple[float, float]:
    """Top-1 image-to-text and text-to-image accuracy on cosine similarity."""
    scores = cosine_similarity_matrix(text_emb, image_emb)
    image_to_text = recall_at_k(
        scores, 1, direction="image_to_text"
    )
    text_to_image = recall_at_k(scores, 1, direction="text_to_image")
    return image_to_text, text_to_image


@dataclass(frozen=True)
class CorpusEvaluation:
    """Full-corpus loss and retrieval metrics."""

    contrastive: float
    lm: float
    caption: float
    vqa: float
    total: float
    image_to_text_top1: float
    text_to_image_top1: float
    temperature: float


def evaluate_corpus(
    model: MultimodalModel,
    corpus: MockCorpus,
    *,
    batch_size: int = 64,
    lm_weight: float = DEFAULT_LM_WEIGHT,
    caption_weight: float = DEFAULT_CAPTION_WEIGHT,
    vqa_weight: float = DEFAULT_VQA_WEIGHT,
) -> CorpusEvaluation:
    """Evaluate weighted losses and retrieval on the full training corpus."""
    size = _positive_int(batch_size, "batch_size")
    lm_scale = _weight(lm_weight, "lm_weight")
    caption_scale = _weight(caption_weight, "caption_weight")
    vqa_scale = _weight(vqa_weight, "vqa_weight")
    total_pairs = len(corpus)
    contrastive_sum = 0.0
    lm_sum = 0.0
    caption_sum = 0.0
    vqa_sum = 0.0
    image_parts: list[np.ndarray] = []
    text_parts: list[np.ndarray] = []
    for start in range(0, total_pairs, size):
        stop = min(start + size, total_pairs)
        output = model.forward(
            corpus.images[start:stop], corpus.captions[start:stop]
        )
        losses = model.losses(
            output,
            lm_weight=lm_scale,
            caption_weight=caption_scale,
            vqa_weight=vqa_scale,
        )
        count = stop - start
        contrastive_sum += losses.contrastive * count
        lm_sum += losses.lm * count
        caption_sum += losses.caption * count
        vqa_sum += losses.vqa * count
        image_parts.append(output.image_emb)
        text_parts.append(output.text_emb)
    contrastive_mean = contrastive_sum / total_pairs
    lm_mean = lm_sum / total_pairs
    caption_mean = caption_sum / total_pairs
    vqa_mean = vqa_sum / total_pairs
    image_emb = np.concatenate(image_parts, axis=0)
    text_emb = np.concatenate(text_parts, axis=0)
    image_to_text, text_to_image = similarity_accuracy(
        image_emb, text_emb
    )
    tau, _ = model.temperature()
    return CorpusEvaluation(
        contrastive=contrastive_mean,
        lm=lm_mean,
        caption=caption_mean,
        vqa=vqa_mean,
        total=(
            contrastive_mean
            + lm_scale * lm_mean
            + caption_scale * caption_mean
            + vqa_scale * vqa_mean
        ),
        image_to_text_top1=image_to_text,
        text_to_image_top1=text_to_image,
        temperature=tau,
    )


def _question_slot(
    question_ids: Sequence[int], offset: int, content_len: int
) -> int:
    """Read the queried content slot from the question token sequence."""
    ids = np.asarray(question_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("question_ids must be a non-empty 1-D sequence")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("question_ids must contain integers")
    slot = int(ids[-1]) - offset
    if slot < 0 or slot >= content_len:
        raise ValueError(
            f"question token is outside the {content_len} question slots"
        )
    return slot


def evaluate(
    model: MultimodalModel,
    suite: EvalSuite,
    *,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run retrieval, VQA, and captioning metrics on a held-out suite."""
    if not isinstance(suite, EvalSuite):
        raise TypeError("suite must be an EvalSuite")
    if not suite.pairs:
        raise ValueError("suite must contain at least one sample")
    size = _positive_int(batch_size, "batch_size")
    count = len(suite.pairs)
    if len(suite.vqa) != count or len(suite.caps) != count:
        raise ValueError(
            "suite pairs, vqa, and caps lists must have equal length"
        )
    content_len = suite.content_len
    if content_len < 1 or content_len > CONTENT_LEN:
        raise ValueError("suite content_len is outside the supported range")
    token_indices = np.asarray(
        content_token_indices(content_len), dtype=np.int64
    )

    image_parts: list[np.ndarray] = []
    text_parts: list[np.ndarray] = []
    vision_parts: list[np.ndarray] = []
    for start in range(0, count, size):
        stop = min(start + size, count)
        images = np.stack(
            [suite.pairs[index].image for index in range(start, stop)]
        )
        captions = np.stack(
            [
                suite.pairs[index].caption_ids
                for index in range(start, stop)
            ]
        )
        vision_tokens, projected = model.encode_images(images)
        vision_parts.append(vision_tokens)
        image_parts.append(projected[:, 0, :])
        text_parts.append(
            model.text_encoder.forward(captions, captions != PAD_ID)
        )
    image_emb = np.concatenate(image_parts, axis=0)
    text_emb = np.concatenate(text_parts, axis=0)
    vision_all = np.concatenate(vision_parts, axis=0)

    # Rows are captions and columns are images, matching the direction names.
    similarity = cosine_similarity_matrix(text_emb, image_emb)
    metrics: dict[str, float] = {
        "retrieval_r1": float(
            recall_at_k(similarity, 1)
        ),
        "retrieval_r5": float(
            recall_at_k(similarity, min(5, count))
        ),
        "retrieval_r10": float(
            recall_at_k(similarity, min(10, count))
        ),
        "retrieval_r1_i2t": float(
            recall_at_k(
                similarity,
                1,
                direction="image_to_text",
            )
        ),
        "retrieval_r5_i2t": float(
            recall_at_k(
                similarity,
                min(5, count),
                direction="image_to_text",
            )
        ),
        "retrieval_r10_i2t": float(
            recall_at_k(
                similarity,
                min(10, count),
                direction="image_to_text",
            )
        ),
    }

    question_slots = np.asarray(
        [
            _question_slot(
                sample.question_ids.tolist(),
                suite.question_offset,
                content_len,
            )
            for sample in suite.vqa
        ],
        dtype=np.int64,
    )
    vqa_patches = vision_all[
        np.arange(count), token_indices[question_slots], :
    ]
    vqa_logits = model.vqa_head.forward(vqa_patches, question_slots)
    vqa_predictions = np.argmax(vqa_logits, axis=-1) + CONTENT_OFFSET
    vqa_references = np.asarray(
        [sample.answer_id for sample in suite.vqa], dtype=np.int64
    )
    metrics["vqa_em"] = vqa_exact_match(
        [int(value) for value in vqa_predictions],
        [int(value) for value in vqa_references],
    )

    caption_logits = model.caption_head.forward(
        vision_all[:, token_indices, :]
    )
    caption_predictions = (
        np.argmax(caption_logits, axis=-1) + CONTENT_OFFSET
    )
    bleu_scores: list[float] = []
    for index, sample in enumerate(suite.caps):
        generated = [int(token) for token in caption_predictions[index]]
        references = [
            _strip_special_tokens(reference.tolist())
            for reference in sample.reference_ids
        ]
        bleu_scores.append(bleu4(generated, references))
    metrics["bleu4"] = float(np.mean(bleu_scores))
    metrics["n_samples"] = float(count)
    return metrics


@dataclass(frozen=True)
class StepRecord:
    """One training-step observation."""

    step: int
    contrastive: float
    lm: float
    caption: float
    vqa: float
    total: float
    temperature: float


@dataclass(frozen=True)
class TrainingReport:
    """Training history plus initial and final full-corpus metrics."""

    records: tuple[StepRecord, ...]
    initial: CorpusEvaluation
    final: CorpusEvaluation


def train(
    model: MultimodalModel,
    corpus: MockCorpus,
    *,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    head_lr: float = DEFAULT_HEAD_LR,
    lm_weight: float = DEFAULT_LM_WEIGHT,
    caption_weight: float = DEFAULT_CAPTION_WEIGHT,
    vqa_weight: float = DEFAULT_VQA_WEIGHT,
    seed: int = DEFAULT_SEED,
    log_every: int = 5,
    print_fn: Callable[[str], None] = print,
) -> TrainingReport:
    """Run the deterministic demo training loop."""
    step_count = _positive_int(steps, "steps")
    batch = _positive_int(batch_size, "batch_size")
    if batch > len(corpus):
        raise ValueError("batch_size cannot exceed corpus size")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if head_lr <= 0.0:
        raise ValueError("head_lr must be positive")
    lm_scale = _weight(lm_weight, "lm_weight")
    caption_scale = _weight(caption_weight, "caption_weight")
    vqa_scale = _weight(vqa_weight, "vqa_weight")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")
    rng = np.random.default_rng(int(seed))
    optimizer = Adam(model.base_parameters(), lr=lr)
    head_optimizer = Adam(model.head_parameters(), lr=head_lr)
    initial = evaluate_corpus(
        model,
        corpus,
        lm_weight=lm_scale,
        caption_weight=caption_scale,
        vqa_weight=vqa_scale,
    )
    records: list[StepRecord] = []
    for step in range(1, step_count + 1):
        indices = rng.choice(
            len(corpus), size=batch, replace=False
        )
        losses = model.loss_and_backward(
            corpus.images[indices],
            corpus.captions[indices],
            lm_weight=lm_scale,
            caption_weight=caption_scale,
            vqa_weight=vqa_scale,
        )
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        head_optimizer.step()
        tau, _ = model.temperature()
        records.append(
            StepRecord(
                step=step,
                contrastive=losses.contrastive,
                lm=losses.lm,
                caption=losses.caption,
                vqa=losses.vqa,
                total=losses.total,
                temperature=tau,
            )
        )
        if step == 1 or step % log_every == 0:
            print_fn(
                f"step {step:3d} | contrastive {losses.contrastive:.4f} | "
                f"lm {losses.lm:.4f} | caption {losses.caption:.4f} | "
                f"vqa {losses.vqa:.4f} | total {losses.total:.4f} | "
                f"tau {tau:.4f}"
            )
    final = evaluate_corpus(
        model,
        corpus,
        lm_weight=lm_scale,
        caption_weight=caption_scale,
        vqa_weight=vqa_scale,
    )
    return TrainingReport(
        records=tuple(records), initial=initial, final=final
    )


def _print_metric_table(
    before: dict[str, float], after: dict[str, float], n_samples: int
) -> None:
    rows = (
        ("R@1 text->image", "retrieval_r1", 1.0 / n_samples),
        ("R@5 text->image", "retrieval_r5", 5.0 / n_samples),
        ("R@10 text->image", "retrieval_r10", 10.0 / n_samples),
        ("R@1 image->text", "retrieval_r1_i2t", 1.0 / n_samples),
        ("R@5 image->text", "retrieval_r5_i2t", 5.0 / n_samples),
        ("R@10 image->text", "retrieval_r10_i2t", 10.0 / n_samples),
        ("VQA exact match", "vqa_em", 1.0 / CONTENT_VOCAB),
        ("BLEU-4", "bleu4", 0.0),
    )
    print(f"{'metric':<20} {'random':>8} {'before':>8} {'after':>8}")
    print("-" * 48)
    for label, key, baseline in rows:
        print(
            f"{label:<20} {baseline:8.4f} "
            f"{before[key]:8.4f} {after[key]:8.4f}"
        )


def demo() -> None:
    """Run the required 50-step, batch-16 training demo."""
    corpus = make_mock_corpus(seed=DEFAULT_SEED, n_pairs=200)
    suite = build_eval_suite(
        seed=DEFAULT_EVAL_SEED,
        n_samples=DEFAULT_EVAL_SAMPLES,
        vocab_size=VOCAB_SIZE,
        max_len=CAPTION_LEN,
    )
    model = build_default_model(seed=DEFAULT_SEED)
    print("multimodal evaluation demo")
    print(
        f"corpus: {len(corpus)} pairs; image batch shape "
        f"{corpus.images.shape}; captions {corpus.captions.shape}"
    )
    print(
        f"held-out suite: {len(suite.pairs)} pairs, "
        f"{len(suite.vqa)} VQA items, {len(suite.caps)} caption items"
    )
    print(
        f"vocabulary: {VOCAB_SIZE}; active content ids "
        f"{CONTENT_OFFSET}..{CONTENT_OFFSET + CONTENT_VOCAB - 1}; "
        f"pad/bos/eos: {PAD_ID}/{BOS_ID}/{EOS_ID}"
    )
    print(
        f"vision tokens: {model.vision_encoder.num_tokens}; "
        f"joint dim: {model.image_projector.output_dim}; "
        f"decoder hidden: {model.decoder.hidden}"
    )
    print(f"model parameters: {model.num_parameters():,}")
    train_captions = {
        tuple(int(token) for token in row[1 : 1 + CONTENT_LEN])
        for row in corpus.captions
    }
    suite_captions = {
        tuple(
            int(token)
            for token in pair.caption_ids[1 : 1 + CONTENT_LEN]
        )
        for pair in suite.pairs
    }
    overlap = train_captions & suite_captions
    print(f"held-out caption overlap: {len(overlap)} of {len(suite.pairs)}")
    if overlap:
        raise SystemExit(
            "demo failed: the evaluation suite overlaps the training corpus"
        )

    before = evaluate(model, suite)
    report = train(
        model,
        corpus,
        steps=DEFAULT_STEPS,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        lm_weight=DEFAULT_LM_WEIGHT,
        caption_weight=DEFAULT_CAPTION_WEIGHT,
        vqa_weight=DEFAULT_VQA_WEIGHT,
        seed=DEFAULT_SEED,
    )
    after = evaluate(model, suite)

    print(
        f"initial full-corpus: contrastive "
        f"{report.initial.contrastive:.4f}, lm "
        f"{report.initial.lm:.4f}, caption "
        f"{report.initial.caption:.4f}, vqa "
        f"{report.initial.vqa:.4f}, total "
        f"{report.initial.total:.4f}, tau "
        f"{report.initial.temperature:.4f}"
    )
    print(
        f"final full-corpus:   contrastive "
        f"{report.final.contrastive:.4f}, lm "
        f"{report.final.lm:.4f}, caption "
        f"{report.final.caption:.4f}, vqa "
        f"{report.final.vqa:.4f}, total "
        f"{report.final.total:.4f}, tau "
        f"{report.final.temperature:.4f}"
    )
    print("\nheld-out evaluation, before vs after 50 steps")
    _print_metric_table(before, after, len(suite.pairs))

    n_samples = len(suite.pairs)
    checks = {
        "R@1 above random": after["retrieval_r1"] > 1.0 / n_samples,
        "R@5 above random": after["retrieval_r5"] > 5.0 / n_samples,
        "R@10 above random": after["retrieval_r10"] > 10.0 / n_samples,
        "VQA above random": (
            after["vqa_em"] > 1.0 / CONTENT_VOCAB
        ),
        "R@1 improved": (
            after["retrieval_r1"] > before["retrieval_r1"]
        ),
        "R@5 improved": (
            after["retrieval_r5"] > before["retrieval_r5"]
        ),
        "R@10 improved": (
            after["retrieval_r10"] > before["retrieval_r10"]
        ),
        "VQA improved": after["vqa_em"] > before["vqa_em"],
        "BLEU-4 improved": after["bleu4"] > before["bleu4"],
        "contrastive decreased": (
            report.final.contrastive < report.initial.contrastive
        ),
        "LM decreased": report.final.lm < report.initial.lm,
        "caption loss decreased": (
            report.final.caption < report.initial.caption
        ),
        "VQA loss decreased": (
            report.final.vqa < report.initial.vqa
        ),
    }
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {label}")
    if not all(checks.values()):
        failed = ", ".join(
            label for label, passed in checks.items() if not passed
        )
        raise SystemExit(f"demo failed: {failed}")


if __name__ == "__main__":
    demo()
