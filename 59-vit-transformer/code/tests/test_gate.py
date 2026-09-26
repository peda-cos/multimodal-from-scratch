"""Top-level gate tests for the vision transformer encoder."""

from __future__ import annotations

import unittest

import numpy as np

from code.main import Block, MultiHeadSelfAttention, VisionEncoder


def test_attention_weights_normalized() -> None:
    """Every attention row is a probability distribution over keys."""
    attention = MultiHeadSelfAttention(32, 4, seed=5)
    rng = np.random.default_rng(30)
    tokens = rng.normal(size=(2, 6, 32)).astype(np.float32)

    _, probs = attention.forward_with_weights(tokens)

    assert probs.shape == (2, 4, 6, 6)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=-1), 1.0, rtol=0.0, atol=1e-5)


def test_cls_pooling_shape_and_cls_identity() -> None:
    """The pooled output is the final CLS token and starts from the learned CLS."""
    encoder = VisionEncoder(patch_size=8, hidden=32, depth=2, heads=4, seed=1)
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)

    tokens, pooled = encoder(images)
    front_end_tokens = encoder.front_end(images)

    assert tokens.shape == (2, 17, 32)
    assert pooled.shape == (2, 32)
    assert np.array_equal(pooled, tokens[:, 0, :])
    assert np.allclose(
        front_end_tokens[:, 0, :],
        np.broadcast_to(encoder.front_end.cls_token, (2, 1, 32))[:, 0, :],
        rtol=0.0,
        atol=0.0,
    )


def test_pre_ln_residual_composition() -> None:
    """A block equals both explicit pre-LN residual passes."""
    block = Block(32, 4, seed=2)
    rng = np.random.default_rng(31)
    tokens = rng.normal(size=(1, 5, 32)).astype(np.float32)

    out = block(tokens)
    after_attention = tokens + block.attn(block.norm1(tokens))
    expected = after_attention + block.ffn(block.norm2(after_attention))

    assert np.allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_cls_uses_every_patch() -> None:
    """Perturbing any single patch changes the pooled CLS vector."""
    encoder = VisionEncoder(patch_size=8, hidden=32, depth=2, heads=4, seed=3)
    rng = np.random.default_rng(32)
    images = rng.random((1, 3, 32, 32), dtype=np.float32)
    base = encoder(images)[1].copy()

    for row in range(4):
        for col in range(4):
            changed = images.copy()
            changed[0, :, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] += 0.5
            pooled = encoder(changed)[1]
            assert np.max(np.abs(pooled - base)) > 1e-7


def test_gradients_nonzero_for_patch_projection() -> None:
    """The CLS loss reaches the patch projection and every input patch."""
    encoder = VisionEncoder(patch_size=4, hidden=16, depth=2, heads=4, seed=4)
    rng = np.random.default_rng(33)
    image = rng.random((1, 3, 8, 8), dtype=np.float32)

    encoder.zero_grad()
    _, pooled = encoder(image)
    image_gradient = encoder.backward(np.ones_like(pooled))

    assert np.linalg.norm(encoder.front_end.patch_embed.proj.weight_grad) > 0.0
    assert np.linalg.norm(encoder.front_end.cls_token_grad) > 0.0
    for row in range(2):
        for col in range(2):
            patch_gradient = image_gradient[
                0, :, row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4
            ]
            assert np.linalg.norm(patch_gradient) > 0.0


def test_small_encoder_forward_shape() -> None:
    """A small encoder preserves batch and sequence dimensions end to end."""
    encoder = VisionEncoder(patch_size=8, hidden=32, depth=2, heads=4, seed=5)
    rng = np.random.default_rng(34)
    images = rng.random((2, 3, 32, 32), dtype=np.float32)

    tokens, pooled = encoder(images)

    assert tokens.shape == (2, 17, 32)
    assert pooled.shape == (2, 32)
    assert len(encoder.vit.cls_history) == 3
    assert np.all(np.isfinite(tokens))


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Run the module-level contract tests through unittest."""
    del loader, pattern
    for name, candidate in sorted(globals().items()):
        if name.startswith("test_") and callable(candidate):
            tests.addTest(unittest.FunctionTestCase(candidate, description=name))
    return tests


if __name__ == "__main__":
    unittest.main()
