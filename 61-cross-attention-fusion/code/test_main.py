"""Detailed tests for masks, attention, caching, and the decoder."""

from __future__ import annotations

import unittest

import numpy as np

from code.main import (
    CausalSelfAttention,
    CrossAttention,
    CrossAttentionCache,
    DecoderBlock,
    DecoderCache,
    FeedForward,
    LayerNorm,
    Linear,
    MockVisionEncoder,
    VisionLanguageDecoder,
    causal_mask,
    gelu,
    softmax,
    validate_attention_masks,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _small_text() -> np.ndarray:
    return np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int64
    )


def _small_model(seed: int = 3) -> VisionLanguageDecoder:
    return VisionLanguageDecoder(
        vocab_size=32,
        hidden=16,
        heads=4,
        layers=2,
        ff_hidden=32,
        max_seq_len=16,
        num_vision_tokens=7,
        num_images=4,
        seed=seed,
    )


def _small_memory(
    model: VisionLanguageDecoder, image_ids: np.ndarray | None = None
) -> np.ndarray:
    ids = (
        np.array([1, 2], dtype=np.int64)
        if image_ids is None
        else image_ids
    )
    return model.vision_encoder.encode(ids)


class PrimitiveTests(unittest.TestCase):
    def test_linear_shape_and_parameter_count(self) -> None:
        linear = Linear(4, 3, seed=0)
        values = _rng(1).normal(size=(2, 5, 4)).astype(np.float32)

        output = linear(values)

        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(linear.num_parameters(), 4 * 3 + 3)

    def test_layer_norm_normalizes_last_axis(self) -> None:
        norm = LayerNorm(8)
        values = _rng(2).normal(size=(2, 3, 8)).astype(np.float32)

        output = norm(values)

        self.assertEqual(output.shape, values.shape)
        np.testing.assert_allclose(
            output.mean(axis=-1), 0.0, rtol=0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            output.var(axis=-1), 1.0, rtol=0.0, atol=1e-4
        )
        self.assertEqual(norm.num_parameters(), 16)

    def test_gelu_and_softmax_basics(self) -> None:
        weights = softmax(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertAlmostEqual(
            float(gelu(np.asarray(0.0, dtype=np.float32))), 0.0, places=6
        )

    def test_constructor_validation(self) -> None:
        with self.assertRaises(ValueError):
            CausalSelfAttention(10, 3)
        with self.assertRaises(ValueError):
            CrossAttention(10, 4)
        with self.assertRaises(ValueError):
            Linear(0, 1)
        with self.assertRaises(ValueError):
            LayerNorm(0)
        with self.assertRaises(ValueError):
            FeedForward(0)


class MaskTests(unittest.TestCase):
    def test_causal_mask_is_lower_triangular_and_boolean(self) -> None:
        mask = causal_mask(10)

        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask.dtype, np.dtype(bool))
        np.testing.assert_array_equal(
            mask, np.tril(np.ones((10, 10), dtype=bool))
        )
        np.testing.assert_array_equal(mask.sum(axis=1), np.arange(1, 11))

    def test_causal_mask_rejects_nonpositive_length(self) -> None:
        with self.assertRaises(ValueError):
            causal_mask(0)
        with self.assertRaises(ValueError):
            causal_mask(-3)

    def test_validate_accepts_baseline_layout(self) -> None:
        validate_attention_masks(causal_mask(10), None, 10, 197)
        validate_attention_masks(
            None, np.ones((10, 197), dtype=bool), 10, 197
        )

    def test_validate_rejects_swapped_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "text_mask"):
            validate_attention_masks(
                np.ones((10, 197), dtype=bool), None, 10, 197
            )
        with self.assertRaisesRegex(ValueError, "cross_mask"):
            validate_attention_masks(None, causal_mask(10), 10, 197)

    def test_validate_rejects_non_causal_and_empty_rows(self) -> None:
        upper = np.triu(np.ones((4, 4), dtype=bool))
        with self.assertRaisesRegex(ValueError, "causal"):
            validate_attention_masks(upper, None, 4, 4)

        empty = np.zeros((4, 4), dtype=bool)
        empty[1, 1] = True
        empty[2, 2] = True
        empty[3, 3] = True
        with self.assertRaisesRegex(ValueError, "text_mask row"):
            validate_attention_masks(empty, None, 4, 4)

        cross = np.ones((4, 5), dtype=bool)
        cross[2, :] = False
        with self.assertRaisesRegex(ValueError, "cross_mask row"):
            validate_attention_masks(None, cross, 4, 5)

    def test_validate_rejects_non_boolean_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_attention_masks(
                np.ones((4, 4), dtype=np.int64), None, 4, 4
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_attention_masks(
                None, np.ones((4, 5), dtype=np.int64), 4, 5
            )

    def test_validate_rejects_nonpositive_lengths(self) -> None:
        with self.assertRaises(ValueError):
            validate_attention_masks(None, None, 0, 5)
        with self.assertRaises(ValueError):
            validate_attention_masks(None, None, 4, 0)


class CausalSelfAttentionTests(unittest.TestCase):
    def test_output_shape_and_dtype(self) -> None:
        attention = CausalSelfAttention(32, 4, seed=3)
        values = _rng(4).normal(size=(2, 10, 32)).astype(np.float32)

        output = attention(values)

        self.assertEqual(output.shape, (2, 10, 32))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_future_tokens_do_not_leak(self) -> None:
        attention = CausalSelfAttention(32, 4, seed=3)
        values = _rng(5).normal(size=(2, 10, 32)).astype(np.float32)
        changed = values.copy()
        changed[:, 5:, :] += 3.0

        baseline = attention(values)
        altered = attention(changed)

        np.testing.assert_allclose(
            baseline[:, :5, :], altered[:, :5, :], rtol=0.0, atol=1e-6
        )
        self.assertFalse(np.allclose(baseline[:, 5:, :], altered[:, 5:, :]))

    def test_explicit_causal_mask_matches_default(self) -> None:
        attention = CausalSelfAttention(32, 4, seed=6)
        values = _rng(7).normal(size=(2, 10, 32)).astype(np.float32)
        mask = causal_mask(10)

        np.testing.assert_allclose(
            attention(values), attention(values, mask), rtol=0.0, atol=1e-6
        )

    def test_rejects_bad_mask_and_input(self) -> None:
        attention = CausalSelfAttention(32, 4, seed=3)
        values = _rng(8).normal(size=(2, 10, 32)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "causal"):
            attention(values, np.triu(np.ones((10, 10), dtype=bool)))
        with self.assertRaises(ValueError):
            attention(values, np.ones((10, 7), dtype=bool))
        with self.assertRaises(ValueError):
            attention(np.zeros((2, 10, 16), dtype=np.float32))
        with self.assertRaises(ValueError):
            attention(np.zeros((2, 0, 32), dtype=np.float32))


class CrossAttentionTests(unittest.TestCase):
    def test_output_shape_for_varied_memory_lengths(self) -> None:
        attention = CrossAttention(32, 4, seed=4)
        rng = _rng(9)
        text = rng.normal(size=(2, 10, 32)).astype(np.float32)

        for memory_len in (7, 197):
            memory = rng.normal(size=(2, memory_len, 32)).astype(np.float32)
            output = attention(text, memory)
            self.assertEqual(output.shape, (2, 10, 32))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(output)))

    def test_manual_per_head_reference(self) -> None:
        attention = CrossAttention(4, 2, seed=7)
        rng = _rng(10)
        text = rng.normal(size=(2, 3, 4)).astype(np.float32)
        memory = rng.normal(size=(2, 5, 4)).astype(np.float32)

        actual = attention(text, memory)
        query = attention.q_proj(text)
        key_value = attention.kv_proj(memory)
        key = key_value[..., :4]
        value = key_value[..., 4:]
        heads = []
        for head in range(2):
            q = query[:, :, head * 2 : (head + 1) * 2]
            k = key[:, :, head * 2 : (head + 1) * 2]
            v = value[:, :, head * 2 : (head + 1) * 2]
            scores = np.einsum("bnd,bmd->bnm", q, k) / np.sqrt(2.0)
            scores -= scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            heads.append(np.einsum("bnm,bmd->bnd", weights, v))
        expected = attention.out_proj(np.concatenate(heads, axis=-1))

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_cache_matches_uncached_and_is_read_only(self) -> None:
        attention = CrossAttention(32, 4, seed=11)
        rng = _rng(12)
        text = rng.normal(size=(2, 10, 32)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 32)).astype(np.float32)
        cache = attention.build_cache(memory)
        self.assertFalse(cache.key.flags.writeable)
        self.assertFalse(cache.value.flags.writeable)
        key_before = cache.key.copy()
        value_before = cache.value.copy()

        cached = attention(text, cache=cache)
        uncached = attention(text, memory)
        for _ in range(3):
            attention(text, cache=cache)

        np.testing.assert_allclose(cached, uncached, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(cache.key, key_before)
        np.testing.assert_array_equal(cache.value, value_before)

    def test_no_key_mask_reaches_every_memory_token(self) -> None:
        attention = CrossAttention(16, 4, seed=13)
        rng = _rng(14)
        text = rng.normal(size=(2, 10, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
        changed = memory.copy()
        changed[:, 3, :] += 1.0

        difference = np.max(
            np.abs(attention(text, changed) - attention(text, memory)), axis=-1
        )

        self.assertTrue(np.all(difference > 1e-6))

    def test_cross_mask_ones_matches_no_mask(self) -> None:
        attention = CrossAttention(16, 4, seed=15)
        rng = _rng(16)
        text = rng.normal(size=(2, 10, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
        mask = np.ones((10, 7), dtype=bool)

        np.testing.assert_allclose(
            attention(text, memory, mask=mask),
            attention(text, memory),
            rtol=0.0,
            atol=1e-6,
        )

    def test_rejects_batch_mismatch_and_invalid_inputs(self) -> None:
        attention = CrossAttention(16, 4, seed=17)
        rng = _rng(18)
        text = rng.normal(size=(2, 10, 16)).astype(np.float32)
        memory = rng.normal(size=(3, 7, 16)).astype(np.float32)

        with self.assertRaisesRegex(ValueError, "batch"):
            attention(text, memory)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            attention(text)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            attention(text, memory, cache=attention.build_cache(memory))

    def test_rejects_bad_cache_shape(self) -> None:
        attention = CrossAttention(16, 4, seed=19)
        text = _rng(20).normal(size=(2, 10, 16)).astype(np.float32)
        bad = CrossAttentionCache(
            key=np.zeros((2, 4, 7, 2), dtype=np.float32),
            value=np.zeros((2, 4, 7, 2), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "cache shape"):
            attention(text, cache=bad)


class DecoderBlockTests(unittest.TestCase):
    def test_zero_sublayers_leave_input_unchanged(self) -> None:
        block = DecoderBlock(16, 4, 32, seed=21)
        for linear in (
            block.self_attn.out_proj,
            block.cross_attn.out_proj,
            block.ffn.fc2,
        ):
            linear.weight.fill(0.0)
            linear.bias.fill(0.0)
        rng = _rng(22)
        text = rng.normal(size=(2, 5, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)

        output = block(text, memory)

        np.testing.assert_array_equal(output, text)

    def test_shape_and_masks(self) -> None:
        block = DecoderBlock(16, 4, 32, seed=23)
        rng = _rng(24)
        text = rng.normal(size=(2, 5, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
        text_mask = causal_mask(5)
        cross_mask = np.ones((5, 7), dtype=bool)

        default = block(text, memory)
        explicit = block(
            text,
            memory,
            text_mask=text_mask,
            cross_mask=cross_mask,
        )

        self.assertEqual(default.shape, (2, 5, 16))
        self.assertTrue(np.all(np.isfinite(default)))
        np.testing.assert_allclose(default, explicit, rtol=0.0, atol=1e-6)

    def test_cache_path_matches_uncached(self) -> None:
        block = DecoderBlock(16, 4, 32, seed=25)
        rng = _rng(26)
        text = rng.normal(size=(2, 5, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
        cache = block.build_cross_cache(memory)

        np.testing.assert_allclose(
            block(text, cache=cache),
            block(text, memory),
            rtol=0.0,
            atol=1e-6,
        )

    def test_cross_norm_sink_and_parameter_count(self) -> None:
        block = DecoderBlock(16, 4, 32, seed=27)
        rng = _rng(28)
        text = rng.normal(size=(2, 5, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)
        sink: list[np.ndarray] = []

        block(text, memory, cross_norm_sink=sink)

        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0].shape, (2, 5))
        self.assertTrue(np.all(sink[0] >= 0.0))
        self.assertEqual(block.num_parameters(), 3_344)

    def test_rejects_invalid_sources_and_batches(self) -> None:
        block = DecoderBlock(16, 4, 32, seed=29)
        rng = _rng(30)
        text = rng.normal(size=(2, 5, 16)).astype(np.float32)
        memory = rng.normal(size=(2, 7, 16)).astype(np.float32)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            block(text)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            block(text, memory, cache=block.build_cross_cache(memory))
        with self.assertRaisesRegex(ValueError, "batch"):
            block(text, memory[:1])


class MockVisionEncoderTests(unittest.TestCase):
    def test_shape_determinism_and_distinct_images(self) -> None:
        encoder = MockVisionEncoder(
            hidden=16, num_tokens=197, num_images=4, seed=31
        )
        ids = np.array([1, 3], dtype=np.int64)

        first = encoder.encode(ids)
        second = encoder.encode(ids)
        other = encoder.encode(np.array([0, 2], dtype=np.int64))

        self.assertEqual(first.shape, (2, 197, 16))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.allclose(first, other))
        self.assertEqual(encoder.num_parameters(), 4 * 16 + 197 * 16 + 16 * 16)

    def test_rejects_bad_image_ids(self) -> None:
        encoder = MockVisionEncoder(
            hidden=16, num_tokens=7, num_images=4, seed=32
        )

        with self.assertRaises(ValueError):
            encoder.encode(np.array([], dtype=np.int64))
        with self.assertRaises(ValueError):
            encoder.encode(np.array([[0]], dtype=np.int64))
        with self.assertRaises(ValueError):
            encoder.encode(np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            encoder.encode(np.array([4], dtype=np.int64))


class VisionLanguageDecoderTests(unittest.TestCase):
    def test_forward_shape_and_parameter_budget(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)

        logits = model.forward(
            text_ids, memory, text_mask=causal_mask(text_ids.shape[1])
        )

        self.assertEqual(logits.shape, (2, 6, 32))
        self.assertEqual(logits.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertEqual(model.num_parameters(), 8_032)

    def test_default_model_has_four_blocks(self) -> None:
        model = VisionLanguageDecoder()

        self.assertEqual(len(model.blocks), 4)
        self.assertEqual(model.num_parameters(), 1_132_544)
        self.assertEqual(model.vision_encoder.num_parameters(), 42_624)

    def test_cached_and_uncached_logits_match(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)
        mask = causal_mask(text_ids.shape[1])
        cache = model.encode_image(memory)

        uncached = model.forward(text_ids, memory, text_mask=mask)
        cached = model.forward(text_ids, cache=cache, text_mask=mask)

        np.testing.assert_allclose(cached, uncached, rtol=0.0, atol=1e-6)

    def test_incremental_cached_decode_matches_uncached_prefix(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)
        cache = model.encode_image(memory)
        prefix = text_ids[:, :3].copy()

        for _ in range(2):
            step_logits = model.forward(
                prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
            )
            next_ids = np.argmax(step_logits[:, -1, :], axis=-1).astype(
                np.int64, copy=False
            )
            prefix = np.concatenate([prefix, next_ids[:, None]], axis=1)

        cached = model.forward(
            prefix, cache=cache, text_mask=causal_mask(prefix.shape[1])
        )
        uncached = model.forward(
            prefix, memory, text_mask=causal_mask(prefix.shape[1])
        )

        np.testing.assert_allclose(cached, uncached, rtol=0.0, atol=1e-6)

    def test_forward_with_cross_norms(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)

        logits, norms = model.forward_with_cross_norms(
            text_ids, memory, text_mask=causal_mask(text_ids.shape[1])
        )

        self.assertEqual(logits.shape, (2, 6, 32))
        self.assertEqual(norms.shape, (2, 2, 6))
        self.assertTrue(np.all(norms >= 0.0))
        np.testing.assert_allclose(
            logits,
            model.forward(
                text_ids, memory, text_mask=causal_mask(text_ids.shape[1])
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_embed_adds_position_rows(self) -> None:
        model = _small_model()
        text_ids = _small_text()

        embedded = model.embed(text_ids)
        expected = (
            model.token_embedding[text_ids]
            + model.position_embedding[: text_ids.shape[1]][None, :, :]
        )

        np.testing.assert_allclose(embedded, expected, rtol=0.0, atol=1e-7)

    def test_is_deterministic_for_seed(self) -> None:
        first = _small_model(seed=3)
        second = _small_model(seed=3)
        third = _small_model(seed=4)
        text_ids = _small_text()
        memory = _small_memory(first)
        mask = causal_mask(text_ids.shape[1])

        first_logits = first.forward(text_ids, memory, text_mask=mask)
        second_logits = second.forward(text_ids, memory, text_mask=mask)
        third_logits = third.forward(
            text_ids, third.vision_encoder.encode(np.array([1, 2])), text_mask=mask
        )

        np.testing.assert_array_equal(first_logits, second_logits)
        self.assertFalse(np.allclose(first_logits, third_logits))

    def test_rejects_batch_mismatch(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        mismatched = model.vision_encoder.encode(
            np.array([0, 1, 2], dtype=np.int64)
        )

        with self.assertRaisesRegex(ValueError, "batch"):
            model.forward(text_ids, mismatched)

    def test_rejects_bad_text_ids_and_length(self) -> None:
        model = _small_model()
        with self.assertRaises(ValueError):
            model.embed(np.array([1, 2], dtype=np.int64))
        with self.assertRaises(ValueError):
            model.embed(np.array([[1.0, 2.0]], dtype=np.float64))
        with self.assertRaises(ValueError):
            model.embed(np.array([[32]], dtype=np.int64))
        with self.assertRaises(ValueError):
            model.embed(np.zeros((1, 17), dtype=np.int64))
        with self.assertRaises(ValueError):
            model.embed(np.zeros((0, 3), dtype=np.int64))
        with self.assertRaises(ValueError):
            model.embed(np.zeros((1, 0), dtype=np.int64))

    def test_rejects_cache_layer_count_and_sources(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)
        cache = model.encode_image(memory)
        bad_cache = DecoderCache(
            per_layer=cache.per_layer[:-1],
            batch_size=cache.batch_size,
            memory_len=cache.memory_len,
        )

        with self.assertRaisesRegex(ValueError, "layers"):
            model.forward(text_ids, cache=bad_cache)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.forward(text_ids)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.forward(text_ids, memory, cache=cache)

    def test_rejects_cache_batch_metadata_mismatch(self) -> None:
        model = _small_model()
        text_ids = _small_text()
        memory = _small_memory(model)
        cache = model.encode_image(memory)
        bad_cache = DecoderCache(
            per_layer=cache.per_layer,
            batch_size=3,
            memory_len=cache.memory_len,
        )

        with self.assertRaisesRegex(ValueError, "batch"):
            model.forward(text_ids, cache=bad_cache)


if __name__ == "__main__":
    unittest.main()
