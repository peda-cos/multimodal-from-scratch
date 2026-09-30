"""Cross-attention fusion decoder implemented from scratch with NumPy.

The decoder keeps text and vision in separate streams. Text tokens are embedded
with a small learned table and processed by causal self-attention; every decoder
block then reaches into a fixed pool of image memory tokens through
multi-head cross-attention. Image keys and values are projected once and held
in a per-layer cache so later decode steps reuse them.

Everything here is forward-only, float32, and deterministic under a seed. The
vision encoder is a synthetic fixture: it preserves the memory-token shape and
the frozen-stream contract, not the semantics of a pretrained ViT.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

__all__ = [
    "DEFAULT_HEADS",
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_MEMORY_TOKENS",
    "DEFAULT_NUM_IMAGES",
    "DEFAULT_SEED",
    "DEFAULT_TEXT_VOCAB",
    "CausalSelfAttention",
    "CrossAttention",
    "CrossAttentionCache",
    "DecoderBlock",
    "DecoderCache",
    "FeedForward",
    "LayerNorm",
    "Linear",
    "MockVisionEncoder",
    "VisionLanguageDecoder",
    "causal_mask",
    "demo",
    "gelu",
    "softmax",
    "validate_attention_masks",
]

DEFAULT_HIDDEN = 128
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 4
DEFAULT_TEXT_VOCAB = 256
DEFAULT_MAX_SEQ_LEN = 64
DEFAULT_MEMORY_TOKENS = 197
DEFAULT_NUM_IMAGES = 8
DEFAULT_SEED = 61


def _sequence(value: np.ndarray, hidden: int, name: str) -> np.ndarray:
    """Validate and convert a ``(batch, sequence, hidden)`` tensor."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D tensor, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty batch and sequence axes")
    if array.shape[2] != hidden:
        raise ValueError(
            f"{name} must have last dimension {hidden}, got {array.shape[2]}"
        )
    return array


def _attention_dims(hidden: int, heads: int) -> int:
    """Validate attention dimensions and return the per-head width."""
    if hidden <= 0 or heads <= 0:
        raise ValueError("hidden and heads must be positive")
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
    return hidden // heads


def _bool_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Validate that a mask is boolean and return it as an array."""
    array = np.asarray(mask)
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a boolean array, got dtype {array.dtype}")
    return array


def causal_mask(length: int) -> np.ndarray:
    """Return a ``(length, length)`` lower-triangular boolean mask."""
    if length <= 0:
        raise ValueError("mask length must be positive")
    return np.tril(np.ones((length, length), dtype=bool))


def validate_attention_masks(
    text_mask: np.ndarray | None,
    cross_mask: np.ndarray | None,
    text_len: int,
    memory_len: int,
) -> None:
    """Validate the two masks used by one decoder block.

    The self-attention mask must be lower-triangular with shape
    ``(text_len, text_len)``. Cross-attention has no mask by default; an
    optional contextual mask has shape ``(text_len, memory_len)`` and no
    causality requirement. Every row must keep at least one key visible.
    """
    if text_len <= 0 or memory_len <= 0:
        raise ValueError("text_len and memory_len must be positive")
    if text_mask is not None:
        mask = _bool_mask(text_mask, "text_mask")
        expected = (text_len, text_len)
        if mask.shape != expected:
            raise ValueError(
                f"text_mask must have shape {expected}, got {mask.shape}"
            )
        if np.any(np.triu(mask, k=1)):
            raise ValueError("text_mask must be causal (lower-triangular)")
        empty_rows = np.flatnonzero(~mask.any(axis=1))
        if empty_rows.size:
            raise ValueError(
                f"text_mask row {int(empty_rows[0])} attends to no keys"
            )
    if cross_mask is not None:
        mask = _bool_mask(cross_mask, "cross_mask")
        expected = (text_len, memory_len)
        if mask.shape != expected:
            raise ValueError(
                f"cross_mask must have shape {expected}, got {mask.shape}"
            )
        empty_rows = np.flatnonzero(~mask.any(axis=1))
        if empty_rows.size:
            raise ValueError(
                f"cross_mask row {int(empty_rows[0])} attends to no keys"
            )


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along one axis."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("softmax input must have at least one axis")
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    total = exponentiated.sum(axis=axis, keepdims=True)
    if not np.all(np.isfinite(total)) or np.any(total == 0.0):
        raise ValueError("softmax received a row with no finite entries")
    return (exponentiated / total).astype(np.float32, copy=False)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU used in the feed-forward network."""
    values = np.asarray(x, dtype=np.float32)
    coefficient = np.float32(sqrt(2.0 / np.pi))
    inner = coefficient * (values + np.float32(0.044715) * values**3)
    return (
        np.float32(0.5) * values * (np.float32(1.0) + np.tanh(inner))
    ).astype(np.float32, copy=False)


def _split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    """Map ``(B, N, D)`` to ``(B, heads, N, D // heads)``."""
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D tensor, got shape {array.shape}")
    batch, sequence, hidden = array.shape
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
    head_dim = hidden // heads
    return array.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    """Map ``(B, heads, N, d)`` back to ``(B, N, heads * d)``."""
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"expected a 4-D tensor, got shape {array.shape}")
    batch, heads, sequence, head_dim = array.shape
    return array.transpose(0, 2, 1, 3).reshape(
        batch, sequence, heads * head_dim
    )


class Linear:
    """Affine projection ``y = x @ weight.T + bias``."""

    def __init__(
        self, in_features: int, out_features: int, *, seed: int = 0
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        rng = np.random.default_rng(seed)
        bound = 1.0 / sqrt(in_features)
        self.weight = rng.uniform(
            -bound, bound, (out_features, in_features)
        ).astype(np.float32)
        self.bias = rng.uniform(-bound, bound, out_features).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 0 or array.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected last dimension {self.in_features}, "
                f"got shape {array.shape}"
            )
        return (array @ self.weight.T + self.bias).astype(
            np.float32, copy=False
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def num_parameters(self) -> int:
        return int(self.weight.size + self.bias.size)


class LayerNorm:
    """Layer normalization over the last axis with affine parameters."""

    def __init__(self, hidden: int, *, eps: float = 1e-5) -> None:
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.hidden = hidden
        self.eps = eps
        self.weight = np.ones(hidden, dtype=np.float32)
        self.bias = np.zeros(hidden, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 0 or array.shape[-1] != self.hidden:
            raise ValueError(
                f"LayerNorm expected last dimension {self.hidden}, "
                f"got shape {array.shape}"
            )
        mean = array.mean(axis=-1, keepdims=True)
        variance = np.mean((array - mean) ** 2, axis=-1, keepdims=True)
        normalized = (array - mean) / np.sqrt(variance + np.float32(self.eps))
        return (normalized * self.weight + self.bias).astype(
            np.float32, copy=False
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def num_parameters(self) -> int:
        return int(self.weight.size + self.bias.size)


class CausalSelfAttention:
    """Masked multi-head self-attention with a fused QKV projection."""

    def __init__(self, hidden: int, heads: int, *, seed: int = 0) -> None:
        self.head_dim = _attention_dims(hidden, heads)
        self.hidden = hidden
        self.heads = heads
        self.scale = np.float32(1.0 / sqrt(self.head_dim))
        self.qkv_proj = Linear(hidden, 3 * hidden, seed=seed)
        self.out_proj = Linear(hidden, hidden, seed=seed + 1)

    def forward(
        self, text_states: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        text = _sequence(text_states, self.hidden, "text_states")
        batch, text_len, _ = text.shape
        if mask is None:
            causal = causal_mask(text_len)
        else:
            causal = _bool_mask(mask, "mask")
            validate_attention_masks(causal, None, text_len, text_len)

        qkv = self.qkv_proj(text)
        qkv = qkv.reshape(batch, text_len, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        scores = np.where(
            causal[None, None, :, :], scores, np.float32(-np.inf)
        )
        weights = softmax(scores, axis=-1)
        context = (weights @ value).transpose(0, 2, 1, 3).reshape(
            batch, text_len, self.hidden
        )
        return self.out_proj(context)

    def __call__(
        self, text_states: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        return self.forward(text_states, mask)

    def num_parameters(self) -> int:
        return self.qkv_proj.num_parameters() + self.out_proj.num_parameters()


@dataclass(frozen=True)
class CrossAttentionCache:
    """Image key/value tensors reused by every cross-attention decode step."""

    key: np.ndarray
    value: np.ndarray


class CrossAttention:
    """Multi-head attention with text queries and image key/value memory."""

    def __init__(self, hidden: int, heads: int, *, seed: int = 0) -> None:
        self.head_dim = _attention_dims(hidden, heads)
        self.hidden = hidden
        self.heads = heads
        self.scale = np.float32(1.0 / sqrt(self.head_dim))
        self.q_proj = Linear(hidden, hidden, seed=seed)
        self.kv_proj = Linear(hidden, 2 * hidden, seed=seed + 1)
        self.out_proj = Linear(hidden, hidden, seed=seed + 2)

    def build_cache(self, memory_tokens: np.ndarray) -> CrossAttentionCache:
        """Project memory to K/V once and freeze the cache arrays."""
        memory = _sequence(memory_tokens, self.hidden, "memory_tokens")
        key_value = self.kv_proj(memory)
        key, value = np.split(key_value, 2, axis=-1)
        key_heads = _split_heads(key, self.heads).copy()
        value_heads = _split_heads(value, self.heads).copy()
        key_heads.setflags(write=False)
        value_heads.setflags(write=False)
        return CrossAttentionCache(key=key_heads, value=value_heads)

    def forward(
        self,
        text_states: np.ndarray,
        memory_tokens: np.ndarray | None = None,
        *,
        mask: np.ndarray | None = None,
        cache: CrossAttentionCache | None = None,
    ) -> np.ndarray:
        text = _sequence(text_states, self.hidden, "text_states")
        if memory_tokens is not None and cache is not None:
            raise ValueError("provide exactly one of memory_tokens or cache")
        if memory_tokens is None and cache is None:
            raise ValueError("provide exactly one of memory_tokens or cache")
        if memory_tokens is not None:
            cache = self.build_cache(memory_tokens)
        assert cache is not None

        key = cache.key
        value = cache.value
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                "cross-attention cache must hold matching 4-D key/value tensors"
            )
        cache_batch, cache_heads, memory_len, cache_head_dim = key.shape
        batch, text_len, _ = text.shape
        if cache_batch != batch:
            raise ValueError(
                f"text batch {batch} does not match image batch {cache_batch}"
            )
        if (
            cache_heads != self.heads
            or cache_head_dim != self.head_dim
            or memory_len <= 0
        ):
            raise ValueError(
                f"cache shape {key.shape} does not match hidden={self.hidden}, "
                f"heads={self.heads}"
            )

        mask_array: np.ndarray | None = None
        if mask is not None:
            mask_array = _bool_mask(mask, "mask")
            validate_attention_masks(None, mask_array, text_len, memory_len)

        query = _split_heads(self.q_proj(text), self.heads)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        if mask_array is not None:
            scores = np.where(
                mask_array[None, None, :, :], scores, np.float32(-np.inf)
            )
        weights = softmax(scores, axis=-1)
        context = _merge_heads(weights @ value)
        return self.out_proj(context)

    def __call__(
        self,
        text_states: np.ndarray,
        memory_tokens: np.ndarray | None = None,
        *,
        mask: np.ndarray | None = None,
        cache: CrossAttentionCache | None = None,
    ) -> np.ndarray:
        return self.forward(
            text_states, memory_tokens, mask=mask, cache=cache
        )

    def num_parameters(self) -> int:
        return (
            self.q_proj.num_parameters()
            + self.kv_proj.num_parameters()
            + self.out_proj.num_parameters()
        )


class FeedForward:
    """Two-layer GELU feed-forward network with a configurable width."""

    def __init__(
        self, hidden: int, ff_hidden: int | None = None, *, seed: int = 0
    ) -> None:
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = 4 * hidden if ff_hidden is None else ff_hidden
        if width <= 0:
            raise ValueError("ff_hidden must be positive")
        self.hidden = hidden
        self.ff_hidden = width
        self.fc1 = Linear(hidden, width, seed=seed)
        self.fc2 = Linear(width, hidden, seed=seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = _sequence(x, self.hidden, "x")
        return self.fc2(gelu(self.fc1(hidden)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def num_parameters(self) -> int:
        return self.fc1.num_parameters() + self.fc2.num_parameters()


class DecoderBlock:
    """Pre-LN decoder block: self-attention, cross-attention, feed-forward."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        ff_hidden: int | None = None,
        *,
        seed: int = 0,
    ) -> None:
        _attention_dims(hidden, heads)
        self.hidden = hidden
        self.heads = heads
        self.ln1 = LayerNorm(hidden)
        self.self_attn = CausalSelfAttention(hidden, heads, seed=seed)
        self.ln2 = LayerNorm(hidden)
        self.cross_attn = CrossAttention(hidden, heads, seed=seed + 1)
        self.ln3 = LayerNorm(hidden)
        self.ffn = FeedForward(hidden, ff_hidden, seed=seed + 2)

    def build_cross_cache(
        self, image_tokens: np.ndarray
    ) -> CrossAttentionCache:
        return self.cross_attn.build_cache(image_tokens)

    def forward(
        self,
        text_tokens: np.ndarray,
        image_tokens: np.ndarray | None = None,
        *,
        text_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
        cache: CrossAttentionCache | None = None,
        cross_norm_sink: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        text = _sequence(text_tokens, self.hidden, "text_tokens")
        if image_tokens is not None and cache is not None:
            raise ValueError("provide exactly one of image_tokens or cache")
        if image_tokens is None and cache is None:
            raise ValueError("provide exactly one of image_tokens or cache")

        memory: np.ndarray | None = None
        if image_tokens is not None:
            memory = _sequence(image_tokens, self.hidden, "image_tokens")
            memory_len = memory.shape[1]
        else:
            assert cache is not None
            if cache.key.ndim != 4 or cache.value.ndim != 4:
                raise ValueError("cross-attention cache tensors must be 4-D")
            memory_len = cache.key.shape[-2]
            if memory_len <= 0:
                raise ValueError("cross-attention cache has no memory tokens")
        validate_attention_masks(
            text_mask, cross_mask, text.shape[1], memory_len
        )

        after_self = text + self.self_attn(self.ln1(text), mask=text_mask)
        cross_input = self.ln2(after_self)
        if memory is not None:
            cross_out = self.cross_attn(
                cross_input, memory, mask=cross_mask
            )
        else:
            assert cache is not None
            cross_out = self.cross_attn(
                cross_input, mask=cross_mask, cache=cache
            )
        if cross_norm_sink is not None:
            cross_norm_sink.append(
                np.linalg.norm(cross_out, axis=-1).astype(
                    np.float32, copy=False
                )
            )
        after_cross = after_self + cross_out
        return after_cross + self.ffn(self.ln3(after_cross))

    def __call__(
        self,
        text_tokens: np.ndarray,
        image_tokens: np.ndarray | None = None,
        *,
        text_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
        cache: CrossAttentionCache | None = None,
        cross_norm_sink: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        return self.forward(
            text_tokens,
            image_tokens,
            text_mask=text_mask,
            cross_mask=cross_mask,
            cache=cache,
            cross_norm_sink=cross_norm_sink,
        )

    def num_parameters(self) -> int:
        return (
            self.ln1.num_parameters()
            + self.self_attn.num_parameters()
            + self.ln2.num_parameters()
            + self.cross_attn.num_parameters()
            + self.ln3.num_parameters()
            + self.ffn.num_parameters()
        )


class MockVisionEncoder:
    """Deterministic shape-preserving stand-in for a frozen vision encoder.

    Image ids index fixed learned-looking tables. The output already has the
    decoder memory layout ``(batch, num_tokens, hidden)``; replacing this class
    with a real frozen ViT forward pass is the documented adapter boundary.
    """

    def __init__(
        self,
        hidden: int,
        num_tokens: int = DEFAULT_MEMORY_TOKENS,
        num_images: int = DEFAULT_NUM_IMAGES,
        *,
        seed: int = 0,
    ) -> None:
        if hidden <= 0 or num_tokens <= 0 or num_images <= 0:
            raise ValueError(
                "hidden, num_tokens, and num_images must be positive"
            )
        self.hidden = hidden
        self.num_tokens = num_tokens
        self.num_images = num_images
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.image_embedding = rng.normal(
            0.0, 0.02, size=(num_images, hidden)
        ).astype(np.float32)
        self.position_embedding = rng.normal(
            0.0, 0.02, size=(num_tokens, hidden)
        ).astype(np.float32)
        self.mix = rng.normal(
            0.0, 1.0 / sqrt(hidden), size=(hidden, hidden)
        ).astype(np.float32)

    def encode(self, image_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(image_ids)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("image_ids must be a non-empty 1-D array")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("image_ids must contain integers")
        indices = ids.astype(np.int64, copy=False)
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_images:
            raise ValueError(
                f"image id outside [0, {self.num_images - 1}]"
            )
        base = self.image_embedding[indices]
        combined = base[:, None, :] + self.position_embedding[None, :, :]
        return np.tanh(combined @ self.mix).astype(np.float32, copy=False)

    def __call__(self, image_ids: np.ndarray) -> np.ndarray:
        return self.encode(image_ids)

    def num_parameters(self) -> int:
        return int(
            self.image_embedding.size
            + self.position_embedding.size
            + self.mix.size
        )


@dataclass(frozen=True)
class DecoderCache:
    """Per-layer image K/V caches for one image batch."""

    per_layer: tuple[CrossAttentionCache, ...]
    batch_size: int
    memory_len: int

    def __post_init__(self) -> None:
        if not self.per_layer:
            raise ValueError("decoder cache must contain at least one layer")
        if self.batch_size <= 0 or self.memory_len <= 0:
            raise ValueError(
                "decoder cache batch_size and memory_len must be positive"
            )


class VisionLanguageDecoder:
    """Pre-LN vision-language decoder with cross-attention in every block."""

    def __init__(
        self,
        vocab_size: int = DEFAULT_TEXT_VOCAB,
        hidden: int = DEFAULT_HIDDEN,
        heads: int = DEFAULT_HEADS,
        layers: int = DEFAULT_LAYERS,
        ff_hidden: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        num_vision_tokens: int = DEFAULT_MEMORY_TOKENS,
        num_images: int = DEFAULT_NUM_IMAGES,
        *,
        seed: int = DEFAULT_SEED,
    ) -> None:
        _attention_dims(hidden, heads)
        width = 4 * hidden if ff_hidden is None else ff_hidden
        if (
            vocab_size <= 0
            or layers <= 0
            or width <= 0
            or max_seq_len <= 0
            or num_vision_tokens <= 0
            or num_images <= 0
        ):
            raise ValueError(
                "vocab_size, layers, ff_hidden, max_seq_len, "
                "num_vision_tokens, and num_images must be positive"
            )
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.heads = heads
        self.layers = layers
        self.ff_hidden = width
        self.max_seq_len = max_seq_len
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.token_embedding = rng.normal(
            0.0, 0.02, size=(vocab_size, hidden)
        ).astype(np.float32)
        self.position_embedding = rng.normal(
            0.0, 0.02, size=(max_seq_len, hidden)
        ).astype(np.float32)
        self.blocks: tuple[DecoderBlock, ...] = tuple(
            DecoderBlock(
                hidden,
                heads,
                width,
                seed=seed + 1 + 10 * layer_index,
            )
            for layer_index in range(layers)
        )
        self.final_norm = LayerNorm(hidden)
        self.lm_head = Linear(hidden, vocab_size, seed=seed + 1000)
        self.vision_encoder = MockVisionEncoder(
            hidden,
            num_tokens=num_vision_tokens,
            num_images=num_images,
            seed=seed + 2000,
        )

    def embed(self, text_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(text_ids)
        if ids.ndim != 2:
            raise ValueError(
                f"text_ids must be a 2-D integer array, got shape {ids.shape}"
            )
        if ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("text_ids must have non-empty batch and sequence axes")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("text_ids must contain integers")
        if ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"text length {ids.shape[1]} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        indices = ids.astype(np.int64, copy=False)
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocab_size:
            raise ValueError(
                f"text id outside [0, {self.vocab_size - 1}]"
            )
        tokens = self.token_embedding[indices]
        positions = self.position_embedding[: indices.shape[1]][None, :, :]
        return (tokens + positions).astype(np.float32, copy=False)

    def encode_image(self, image_tokens: np.ndarray) -> DecoderCache:
        memory = _sequence(image_tokens, self.hidden, "image_tokens")
        per_layer = tuple(
            block.build_cross_cache(memory) for block in self.blocks
        )
        return DecoderCache(
            per_layer=per_layer,
            batch_size=memory.shape[0],
            memory_len=memory.shape[1],
        )

    def _run(
        self,
        text_ids: np.ndarray,
        image_tokens: np.ndarray | None,
        *,
        cache: DecoderCache | None,
        text_mask: np.ndarray | None,
        cross_mask: np.ndarray | None,
        collect_cross_norms: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if image_tokens is not None and cache is not None:
            raise ValueError("provide exactly one of image_tokens or cache")
        if image_tokens is None and cache is None:
            raise ValueError("provide exactly one of image_tokens or cache")

        tokens = self.embed(text_ids)
        if image_tokens is not None:
            decoder_cache = self.encode_image(image_tokens)
        else:
            assert cache is not None
            decoder_cache = cache

        if len(decoder_cache.per_layer) != len(self.blocks):
            raise ValueError(
                f"cache has {len(decoder_cache.per_layer)} layers but decoder "
                f"has {len(self.blocks)}"
            )
        if decoder_cache.batch_size != tokens.shape[0]:
            raise ValueError(
                f"text batch {tokens.shape[0]} does not match image batch "
                f"{decoder_cache.batch_size}"
            )
        for layer_cache in decoder_cache.per_layer:
            if layer_cache.key.ndim != 4 or layer_cache.value.ndim != 4:
                raise ValueError("decoder cache contains a non-4-D layer cache")
            if layer_cache.key.shape[0] != decoder_cache.batch_size:
                raise ValueError(
                    "decoder cache batch metadata does not match its tensors"
                )
            if layer_cache.key.shape[-2] != decoder_cache.memory_len:
                raise ValueError(
                    "decoder cache memory metadata does not match its tensors"
                )

        validate_attention_masks(
            text_mask,
            cross_mask,
            tokens.shape[1],
            decoder_cache.memory_len,
        )

        states = tokens
        sink: list[np.ndarray] | None = [] if collect_cross_norms else None
        for block, layer_cache in zip(
            self.blocks, decoder_cache.per_layer, strict=True
        ):
            states = block.forward(
                states,
                text_mask=text_mask,
                cross_mask=cross_mask,
                cache=layer_cache,
                cross_norm_sink=sink,
            )
        logits = self.lm_head(self.final_norm(states))
        if sink is None:
            return logits, None
        return logits, np.stack(sink).astype(np.float32, copy=False)

    def forward(
        self,
        text_ids: np.ndarray,
        image_tokens: np.ndarray | None = None,
        *,
        cache: DecoderCache | None = None,
        text_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        logits, _ = self._run(
            text_ids,
            image_tokens,
            cache=cache,
            text_mask=text_mask,
            cross_mask=cross_mask,
            collect_cross_norms=False,
        )
        return logits

    def forward_with_cross_norms(
        self,
        text_ids: np.ndarray,
        image_tokens: np.ndarray | None = None,
        *,
        cache: DecoderCache | None = None,
        text_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits, norms = self._run(
            text_ids,
            image_tokens,
            cache=cache,
            text_mask=text_mask,
            cross_mask=cross_mask,
            collect_cross_norms=True,
        )
        if norms is None:
            raise RuntimeError("cross norms were not collected")
        return logits, norms

    def __call__(
        self,
        text_ids: np.ndarray,
        image_tokens: np.ndarray | None = None,
        *,
        cache: DecoderCache | None = None,
        text_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward(
            text_ids,
            image_tokens,
            cache=cache,
            text_mask=text_mask,
            cross_mask=cross_mask,
        )

    def num_parameters(self) -> int:
        return (
            int(self.token_embedding.size)
            + int(self.position_embedding.size)
            + sum(block.num_parameters() for block in self.blocks)
            + self.final_norm.num_parameters()
            + self.lm_head.num_parameters()
        )


def demo() -> None:
    """Run the deterministic two-sequence, 197-token cross-attention demo."""
    model = VisionLanguageDecoder()
    text_ids = np.array(
        [
            [1, 5, 9, 13, 17, 21, 25, 29, 33, 37],
            [2, 6, 10, 14, 18, 22, 26, 30, 34, 38],
        ],
        dtype=np.int64,
    )
    image_ids = np.array([3, 6], dtype=np.int64)
    memory = model.vision_encoder.encode(image_ids)
    mask = causal_mask(text_ids.shape[1])

    logits_uncached = model.forward(text_ids, memory, text_mask=mask)
    logits_with_norms, cross_norms = model.forward_with_cross_norms(
        text_ids, memory, text_mask=mask
    )
    cache = model.encode_image(memory)
    logits_cached = model.forward(text_ids, cache=cache, text_mask=mask)
    cache_delta = float(np.max(np.abs(logits_cached - logits_uncached)))

    print("cross-attention fusion decoder")
    print(
        f"dimensions: hidden={model.hidden}, heads={model.heads}, "
        f"layers={len(model.blocks)}, vocab={model.vocab_size}"
    )
    print(
        f"text: {text_ids.shape[0]} sequences x {text_ids.shape[1]} tokens; "
        f"image memory: {memory.shape[0]} x {memory.shape[1]} tokens"
    )
    print(f"decoder parameters: {model.num_parameters():,}")
    print(
        "mock vision encoder parameters: "
        f"{model.vision_encoder.num_parameters():,} (fixed fixture)"
    )
    print(f"logits shape: {logits_uncached.shape}")
    print(f"self-attention mask shape: {mask.shape}")
    print(f"causal mask is lower-triangular: {bool(np.array_equal(mask, np.tril(mask)))}")
    print(
        "cross-attention output norm (layer 1, sample 1):",
        np.array2string(
            cross_norms[0, 0], precision=6, separator=", "
        ),
    )
    print(f"cross-norm tensor shape: {cross_norms.shape}")
    print(f"KV-cache max |logit delta|: {cache_delta:.3e}")
    print(f"cached logits match uncached: {cache_delta < 1e-6}")
    print(
        "diagnostic logits match regular logits: "
        f"{bool(np.array_equal(logits_with_norms, logits_uncached))}"
    )

    prefix = text_ids[:, :5].copy()
    for _ in range(3):
        step_logits = model.forward(
            prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
        )
        next_ids = np.argmax(step_logits[:, -1, :], axis=-1).astype(
            np.int64, copy=False
        )
        prefix = np.concatenate([prefix, next_ids[:, None]], axis=1)
    cached_prefix_logits = model.forward(
        prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
    )
    uncached_prefix_logits = model.forward(
        prefix, memory, text_mask=causal_mask(prefix.shape[1])
    )
    decode_delta = float(
        np.max(np.abs(cached_prefix_logits - uncached_prefix_logits))
    )
    print(f"cached greedy decode: 5-token prompt -> {prefix.shape[1]} tokens")
    print(f"cached decode max |logit delta| vs uncached prefix: {decode_delta:.3e}")
    print(f"cached decode matches uncached prefix: {decode_delta < 1e-6}")


if __name__ == "__main__":
    demo()
