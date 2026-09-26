"""Tests for the NumPy vision transformer encoder."""

from __future__ import annotations

import unittest

import numpy as np

from code.main import (
    Block,
    FeedForward,
    MultiHeadSelfAttention,
    VisionEncoder,
    synthesize_image,
)


def make_encoder(
    patch_size: int = 8,
    hidden: int = 32,
    depth: int = 2,
    heads: int = 4,
    seed: int = 0,
) -> VisionEncoder:
    """Small deterministic encoder used by most tests."""
    return VisionEncoder(
        patch_size=patch_size,
        hidden=hidden,
        depth=depth,
        heads=heads,
        seed=seed,
    )


def manual_attention(
    attention: MultiHeadSelfAttention, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent attention reference built from the module's raw weights."""
    batch, seq, _ = x.shape
    qkv = x @ attention.qkv.weight.T + attention.qkv.bias
    qkv = qkv.reshape(
        batch, seq, 3, attention.heads, attention.head_dim
    ).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(0, 1, 3, 2)) * attention.scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    context = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq, attention.hidden)
    out = context @ attention.proj.weight.T + attention.proj.bias
    return out, probs


class AttentionTests(unittest.TestCase):
    def test_attention_weights_sum_to_one(self) -> None:
        attention = MultiHeadSelfAttention(32, 4, seed=1)
        rng = np.random.default_rng(10)
        tokens = rng.normal(size=(2, 7, 32)).astype(np.float32)

        out, probs = attention.forward_with_weights(tokens)

        self.assertEqual(out.shape, (2, 7, 32))
        self.assertEqual(probs.shape, (2, 4, 7, 7))
        self.assertTrue(np.all(probs >= 0.0))
        np.testing.assert_allclose(
            probs.sum(axis=-1), np.ones((2, 4, 7)), rtol=0.0, atol=1e-5
        )

    def test_attention_matches_manual_einsum(self) -> None:
        attention = MultiHeadSelfAttention(32, 4, seed=2)
        rng = np.random.default_rng(11)
        tokens = rng.normal(size=(1, 6, 32)).astype(np.float32)

        out, probs = attention.forward_with_weights(tokens)
        expected_out, expected_probs = manual_attention(attention, tokens)

        np.testing.assert_allclose(out, expected_out, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            probs, expected_probs, rtol=1e-5, atol=1e-6
        )

    def test_attention_rejects_bad_config(self) -> None:
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(30, 4)
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(32, 0)

        attention = MultiHeadSelfAttention(32, 4, seed=0)
        with self.assertRaises(ValueError):
            attention.forward(np.zeros((1, 5, 31), dtype=np.float32))


class BlockTests(unittest.TestCase):
    def test_block_preserves_shape_and_batch_invariance(self) -> None:
        block = Block(32, 4, seed=0)
        rng = np.random.default_rng(12)
        tokens = rng.normal(size=(1, 8, 32)).astype(np.float32)

        single = block(tokens)
        batched = block(np.repeat(tokens, 3, axis=0))

        self.assertEqual(single.shape, tokens.shape)
        self.assertEqual(batched.shape, (3, 8, 32))
        for batch in range(3):
            np.testing.assert_allclose(
                batched[batch], single[0], rtol=1e-5, atol=1e-6
            )

    def test_block_matches_explicit_pre_ln_composition(self) -> None:
        block = Block(32, 4, seed=0)
        rng = np.random.default_rng(13)
        tokens = rng.normal(size=(1, 6, 32)).astype(np.float32)

        out = block(tokens)
        after_attention = tokens + block.attn(block.norm1(tokens))
        expected = after_attention + block.ffn(block.norm2(after_attention))

        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


class FeedForwardTests(unittest.TestCase):
    def test_feed_forward_roundtrip_and_expansion(self) -> None:
        feed_forward = FeedForward(32, expansion=4, seed=0)
        rng = np.random.default_rng(14)
        tokens = rng.normal(size=(2, 5, 32)).astype(np.float32)

        out = feed_forward(tokens)

        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(feed_forward.fc1.weight.shape, (128, 32))
        self.assertEqual(feed_forward.fc2.weight.shape, (32, 128))


class EncoderTests(unittest.TestCase):
    def test_encoder_returns_contextual_and_pooled_shapes(self) -> None:
        encoder = make_encoder()
        rng = np.random.default_rng(15)
        images = rng.random((2, 3, 32, 32), dtype=np.float32)

        tokens, pooled = encoder(images)

        self.assertEqual(tokens.shape, (2, 17, 32))
        self.assertEqual(pooled.shape, (2, 32))
        np.testing.assert_array_equal(pooled, tokens[:, 0, :])
        self.assertEqual(len(encoder.vit.cls_history), 3)

    def test_four_layer_stack_output_shape(self) -> None:
        encoder = make_encoder(hidden=24, depth=4)
        rng = np.random.default_rng(16)
        images = rng.random((1, 3, 32, 32), dtype=np.float32)

        tokens, pooled = encoder(images)

        self.assertEqual(tokens.shape, (1, 17, 24))
        self.assertEqual(pooled.shape, (1, 24))
        self.assertEqual(len(encoder.vit.cls_history), 5)

    def test_cls_pools_information_from_every_patch(self) -> None:
        encoder = make_encoder()
        rng = np.random.default_rng(17)
        images = rng.random((1, 3, 32, 32), dtype=np.float32)
        base = encoder(images)[1].copy()

        patch_size = 8
        for row in range(4):
            for col in range(4):
                changed = images.copy()
                changed[
                    0,
                    :,
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ] += 0.5
                pooled = encoder(changed)[1]
                self.assertGreater(
                    float(np.max(np.abs(pooled - base))),
                    1e-7,
                    msg=f"CLS ignored patch ({row}, {col})",
                )

    def test_zero_image_cls_is_nonzero_and_learned(self) -> None:
        encoder = make_encoder()
        images = np.zeros((2, 3, 32, 32), dtype=np.float32)

        tokens, pooled = encoder(images)
        front_end_tokens = encoder.front_end(images)
        learned_cls = encoder.front_end.cls_token.reshape(-1)

        self.assertTrue(np.all(np.isfinite(pooled)))
        self.assertGreater(float(np.linalg.norm(pooled, axis=1).min()), 0.0)
        for batch in range(2):
            np.testing.assert_allclose(
                front_end_tokens[batch, 0], learned_cls, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                tokens[batch, 0], pooled[batch], rtol=0.0, atol=0.0
            )

    def test_vit_base_parameter_count(self) -> None:
        encoder = VisionEncoder()

        self.assertEqual(encoder.num_parameters(), 85_647_360)
        self.assertEqual(encoder.vit.blocks[0].num_parameters(), 7_087_872)
        self.assertEqual(encoder.front_end.num_parameters(), 591_360)
        self.assertEqual(encoder.vit.norm.num_parameters(), 1_536)


class GradientTests(unittest.TestCase):
    def test_backward_matches_finite_difference(self) -> None:
        encoder = make_encoder(patch_size=4, hidden=16, depth=2, heads=4)
        rng = np.random.default_rng(18)
        image = rng.random((1, 3, 8, 8), dtype=np.float32)
        probe = rng.normal(size=16).astype(np.float32)

        encoder.zero_grad()
        _, pooled = encoder(image)
        encoder.backward(np.ones_like(pooled) * probe)

        parameters = [
            (
                "patch_projection",
                encoder.front_end.patch_embed.proj.weight,
                encoder.front_end.patch_embed.proj.weight_grad,
            ),
            (
                "qkv",
                encoder.vit.blocks[0].attn.qkv.weight,
                encoder.vit.blocks[0].attn.qkv.weight_grad,
            ),
            (
                "fc1",
                encoder.vit.blocks[0].ffn.fc1.weight,
                encoder.vit.blocks[0].ffn.fc1.weight_grad,
            ),
            (
                "norm1_gamma",
                encoder.vit.blocks[0].norm1.gamma,
                encoder.vit.blocks[0].norm1.gamma_grad,
            ),
            (
                "cls_token",
                encoder.front_end.cls_token,
                encoder.front_end.cls_token_grad,
            ),
        ]

        def loss() -> float:
            return float(encoder(image)[1][0] @ probe)

        for name, parameter, gradient in parameters:
            flat_parameter = parameter.reshape(-1)
            flat_gradient = gradient.reshape(-1)
            index = int(np.argmax(np.abs(flat_gradient)))
            analytic = float(flat_gradient[index])
            self.assertNotEqual(analytic, 0.0, msg=name)
            old = float(flat_parameter[index])
            eps = 1e-3
            flat_parameter[index] = old + eps
            plus = loss()
            flat_parameter[index] = old - eps
            minus = loss()
            flat_parameter[index] = old
            finite_difference = (plus - minus) / (2.0 * eps)
            relative_error = abs(analytic - finite_difference) / max(
                1e-12, abs(finite_difference)
            )
            self.assertLess(relative_error, 1e-2, msg=name)

        encoder.zero_grad()
        encoder(image)
        image_gradient = encoder.backward(probe[None, :])
        flat_image = image.reshape(-1)
        flat_image_gradient = image_gradient.reshape(-1)
        index = int(np.argmax(np.abs(flat_image_gradient)))
        analytic = float(flat_image_gradient[index])
        self.assertNotEqual(analytic, 0.0)
        old = float(flat_image[index])
        eps = 1e-2
        flat_image[index] = old + eps
        plus = loss()
        flat_image[index] = old - eps
        minus = loss()
        flat_image[index] = old
        finite_difference = (plus - minus) / (2.0 * eps)
        relative_error = abs(analytic - finite_difference) / max(
            1e-12, abs(finite_difference)
        )
        self.assertLess(relative_error, 1e-2, msg="image_pixel")

    def test_gradients_reach_patch_projection_and_all_blocks(self) -> None:
        encoder = make_encoder(patch_size=4, hidden=16, depth=2, heads=4)
        rng = np.random.default_rng(19)
        image = rng.random((1, 3, 8, 8), dtype=np.float32)

        encoder.zero_grad()
        _, pooled = encoder(image)
        encoder.backward(np.ones_like(pooled))

        self.assertGreater(
            float(np.linalg.norm(encoder.front_end.patch_embed.proj.weight_grad)),
            0.0,
        )
        self.assertGreater(
            float(np.linalg.norm(encoder.front_end.cls_token_grad)), 0.0
        )
        for block in encoder.vit.blocks:
            self.assertGreater(float(np.linalg.norm(block.attn.qkv.weight_grad)), 0.0)
            self.assertGreater(float(np.linalg.norm(block.ffn.fc1.weight_grad)), 0.0)

        encoder.zero_grad()
        encoder(image)
        image_gradient = encoder.backward(np.ones((1, 16), dtype=np.float32))
        patch_size = 4
        for row in range(2):
            for col in range(2):
                patch_gradient = image_gradient[
                    0,
                    :,
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ]
                self.assertGreater(float(np.linalg.norm(patch_gradient)), 0.0)


class ValidationTests(unittest.TestCase):
    def test_input_validation(self) -> None:
        encoder = make_encoder()

        with self.assertRaises(ValueError):
            encoder.forward(np.zeros((3, 32, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            encoder.forward(np.zeros((1, 4, 32, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            encoder.forward(np.zeros((1, 3, 30, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            make_encoder(patch_size=16).forward(
                np.zeros((1, 3, 8, 8), dtype=np.float32)
            )
        with self.assertRaises(ValueError):
            VisionEncoder(patch_size=8, hidden=30)
        with self.assertRaises(ValueError):
            VisionEncoder(patch_size=8, hidden=30, heads=4)
        with self.assertRaises(ValueError):
            VisionEncoder(patch_size=8, hidden=32, heads=0)
        with self.assertRaises(ValueError):
            VisionEncoder(patch_size=8, hidden=32, depth=0)
        with self.assertRaises(ValueError):
            VisionEncoder(patch_size=8, hidden=32, expansion=0)


class FixtureTests(unittest.TestCase):
    def test_fixture_is_deterministic(self) -> None:
        first = synthesize_image(seed=77, size=48)
        second = synthesize_image(seed=77, size=48)
        different = synthesize_image(seed=78, size=48)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.shape, (1, 3, 48, 48))
        self.assertEqual(first.dtype, np.float32)

        np.random.seed(1234)
        expected_stream = [float(np.random.random()) for _ in range(3)]
        np.random.seed(1234)
        synthesize_image(seed=9, size=8)
        observed_stream = [float(np.random.random()) for _ in range(3)]
        self.assertEqual(observed_stream, expected_stream)


if __name__ == "__main__":
    unittest.main()
