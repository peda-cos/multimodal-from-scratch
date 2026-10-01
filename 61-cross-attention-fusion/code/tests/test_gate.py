"""Top-level contract tests counted by the portfolio verification gate."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import numpy as np

from code.main import (
    CausalSelfAttention,
    CrossAttention,
    MockVisionEncoder,
    VisionLanguageDecoder,
    causal_mask,
    validate_attention_masks,
)


def _expect_value_error(call: Callable[[], object]) -> str:
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError("expected a ValueError")


def _default_inputs() -> tuple[VisionLanguageDecoder, np.ndarray, np.ndarray]:
    model = VisionLanguageDecoder()
    text_ids = np.array(
        [
            [1, 5, 9, 13, 17, 21, 25, 29, 33, 37],
            [2, 6, 10, 14, 18, 22, 26, 30, 34, 38],
        ],
        dtype=np.int64,
    )
    memory = model.vision_encoder.encode(
        np.array([3, 6], dtype=np.int64)
    )
    return model, text_ids, memory


def test_causal_mask_lower_triangular() -> None:
    """The causal mask is boolean, square, and exactly lower-triangular."""
    mask = causal_mask(10)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.dtype(bool)
    assert np.array_equal(mask, np.tril(np.ones((10, 10), dtype=bool)))
    assert np.array_equal(mask.sum(axis=1), np.arange(1, 11))


def test_mask_validation_rejects_mixed_shapes() -> None:
    """Swapping self/cross masks surfaces as a named shape ValueError."""
    validate_attention_masks(causal_mask(10), None, 10, 197)
    cross_error = _expect_value_error(
        lambda: validate_attention_masks(None, causal_mask(10), 10, 197)
    )
    assert "cross_mask" in cross_error
    assert "(10, 197)" in cross_error

    text_error = _expect_value_error(
        lambda: validate_attention_masks(
            np.ones((10, 197), dtype=bool), None, 10, 197
        )
    )
    assert "text_mask" in text_error
    assert "(10, 10)" in text_error


def test_cross_attention_shape_across_key_lengths() -> None:
    """Cross-attention returns (B, Nt, hidden) for 7 and 197 memory tokens."""
    attention = CrossAttention(32, 4, seed=0)
    rng = np.random.default_rng(1)
    text = rng.normal(size=(2, 10, 32)).astype(np.float32)

    for memory_len in (7, 197):
        memory = rng.normal(size=(2, memory_len, 32)).astype(np.float32)
        output = attention(text, memory)
        assert output.shape == (2, 10, 32)
        assert output.dtype == np.float32
        assert np.isfinite(output).all()


def test_cross_attention_cache_matches_uncached() -> None:
    """The cached image K/V path reproduces the uncached path."""
    attention = CrossAttention(32, 4, seed=2)
    rng = np.random.default_rng(3)
    text = rng.normal(size=(2, 10, 32)).astype(np.float32)
    memory = rng.normal(size=(2, 7, 32)).astype(np.float32)
    cache = attention.build_cache(memory)

    cached = attention(text, cache=cache)
    uncached = attention(text, memory)

    assert not cache.key.flags.writeable
    assert not cache.value.flags.writeable
    assert np.allclose(cached, uncached, rtol=0.0, atol=1e-6)


def test_cross_attention_sees_every_memory_token() -> None:
    """With no key mask, one changed memory token reaches every query."""
    attention = CrossAttention(16, 4, seed=4)
    rng = np.random.default_rng(5)
    text = rng.normal(size=(2, 10, 16)).astype(np.float32)
    memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
    changed = memory.copy()
    changed[:, 3, :] += 1.0

    difference = np.max(
        np.abs(attention(text, changed) - attention(text, memory)), axis=-1
    )

    assert np.all(difference > 1e-6)


def test_self_attention_future_tokens_do_not_leak() -> None:
    """Changing future text tokens leaves earlier causal outputs intact."""
    attention = CausalSelfAttention(32, 4, seed=6)
    rng = np.random.default_rng(7)
    text = rng.normal(size=(2, 10, 32)).astype(np.float32)
    changed = text.copy()
    changed[:, 5:, :] += 3.0

    baseline = attention(text)
    altered = attention(changed)

    assert np.allclose(
        baseline[:, :5, :], altered[:, :5, :], rtol=0.0, atol=1e-6
    )
    assert not np.allclose(baseline[:, 5:, :], altered[:, 5:, :])


def test_decoder_logits_and_mask_shapes() -> None:
    """The demo-shaped pass returns (2, 10, 256) logits and a (10, 10) mask."""
    model, text_ids, memory = _default_inputs()
    mask = causal_mask(text_ids.shape[1])

    logits = model.forward(text_ids, memory, text_mask=mask)

    assert logits.shape == (2, 10, 256)
    assert mask.shape == (10, 10)
    assert np.isfinite(logits).all()


def test_decoder_cache_matches_uncached_logits() -> None:
    """Full decoder logits are identical with and without a prebuilt cache."""
    model, text_ids, memory = _default_inputs()
    mask = causal_mask(text_ids.shape[1])
    cache = model.encode_image(memory)

    cached = model.forward(text_ids, cache=cache, text_mask=mask)
    uncached = model.forward(text_ids, memory, text_mask=mask)

    assert np.allclose(cached, uncached, rtol=0.0, atol=1e-6)


def test_decoder_rejects_text_image_batch_mismatch() -> None:
    """A text batch and image batch must agree before attention runs."""
    model, text_ids, _ = _default_inputs()
    memory = model.vision_encoder.encode(
        np.array([0, 1, 2], dtype=np.int64)
    )

    message = _expect_value_error(lambda: model.forward(text_ids, memory))

    assert "batch" in message


def test_decoder_has_four_blocks_and_cross_norms() -> None:
    """The default decoder is four blocks deep and exposes per-position norms."""
    model, text_ids, memory = _default_inputs()
    mask = causal_mask(text_ids.shape[1])

    logits, norms = model.forward_with_cross_norms(
        text_ids, memory, text_mask=mask
    )

    assert len(model.blocks) == 4
    assert logits.shape == (2, 10, 256)
    assert norms.shape == (4, 2, 10)
    assert np.all(norms >= 0.0)


def test_incremental_cached_decode_matches_uncached_prefix() -> None:
    """Three cached greedy steps match an uncached pass on the resulting prefix."""
    model, text_ids, memory = _default_inputs()
    cache = model.encode_image(memory)
    prefix = text_ids[:, :5].copy()

    for _ in range(3):
        step_logits = model.forward(
            prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
        )
        next_ids = np.argmax(step_logits[:, -1, :], axis=-1)
        prefix = np.concatenate([prefix, next_ids[:, None]], axis=1)

    cached = model.forward(
        prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
    )
    uncached = model.forward(
        prefix, memory, text_mask=causal_mask(prefix.shape[1])
    )

    assert np.allclose(cached, uncached, rtol=0.0, atol=1e-6)


def test_mock_vision_encoder_is_deterministic() -> None:
    """The fixture emits 197 deterministic tokens per image id."""
    encoder = MockVisionEncoder(
        hidden=16, num_tokens=197, num_images=4, seed=8
    )
    ids = np.array([1, 3], dtype=np.int64)

    first = encoder.encode(ids)
    second = encoder.encode(ids)
    other = encoder.encode(np.array([0, 2], dtype=np.int64))

    assert first.shape == (2, 197, 16)
    assert np.array_equal(first, second)
    assert not np.allclose(first, other)


def test_decoder_parameter_budget() -> None:
    """The default decoder and mock encoder have the documented parameter counts."""
    model = VisionLanguageDecoder()

    assert model.num_parameters() == 1_132_544
    assert model.vision_encoder.num_parameters() == 42_624


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Register the top-level functions with unittest discovery."""
    del loader, pattern
    for name, candidate in sorted(globals().items()):
        if name.startswith("test_") and callable(candidate):
            tests.addTest(unittest.FunctionTestCase(candidate, description=name))
    return tests


if __name__ == "__main__":
    unittest.main()
