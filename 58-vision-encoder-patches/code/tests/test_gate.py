"""Tests for the NumPy patch-tokenizer front end."""

from __future__ import annotations

import unittest

import numpy as np

from code.main import Conv2d, VisionFrontEnd, sinusoidal_2d, synthesize_image


def test_default_sequence_shape() -> None:
    """The default ViT-Base/16 front end yields 196 patches plus CLS."""
    model = VisionFrontEnd(seed=0)
    images = np.zeros((2, 3, 224, 224), dtype=np.float32)
    tokens = model(images)

    assert tokens.shape == (2, 197, 768)
    assert tokens.dtype == np.float32


def test_rectangular_conv_matches_manual_loop() -> None:
    """A rectangular 10x14 image matches explicit per-patch dot products."""
    rng = np.random.default_rng(21)
    image = rng.normal(size=(1, 2, 10, 14)).astype(np.float32)
    conv = Conv2d(2, 4, kernel_size=2, seed=12)
    output = conv(image)

    assert output.shape == (1, 4, 5, 7)
    for row in range(5):
        for col in range(7):
            patch = image[0, :, row * 2 : (row + 1) * 2, col * 2 : (col + 1) * 2]
            expected = np.einsum("cuv,ocuv->o", patch, conv.weight) + conv.bias
            assert np.allclose(
                output[0, :, row, col], expected, rtol=1e-6, atol=1e-6
            )


def test_position_norms_uniform_at_vit_resolution() -> None:
    """All 196 position rows have the same norm at hidden=768."""
    positions = sinusoidal_2d(14, 14, 768)
    norms = np.linalg.norm(positions, axis=1)

    assert positions.shape == (196, 768)
    assert np.allclose(norms, np.sqrt(384.0), rtol=0.0, atol=1e-4)
    assert np.array_equal(
        positions[0, :8],
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    assert not np.array_equal(positions[0], positions[-1])


def test_cls_token_is_shared_and_unpositioned() -> None:
    """CLS is identical across a batch and does not receive a position."""
    model = VisionFrontEnd(patch_size=16, hidden=32, seed=3)
    rng = np.random.default_rng(33)
    images = rng.normal(size=(3, 3, 32, 32)).astype(np.float32)
    tokens = model(images)
    learned_cls = model.cls_token.reshape(-1)

    for batch in range(3):
        assert np.array_equal(tokens[batch, 0], learned_cls)
    assert np.array_equal(model.cls_token.reshape(-1), learned_cls)
    assert not np.array_equal(tokens[0, 1:], tokens[1, 1:])


def test_patch_count_scales_with_resolution() -> None:
    """A wider input changes the grid and sequence length deterministically."""
    model = VisionFrontEnd(patch_size=16, hidden=32, seed=4)
    tokens = model(np.zeros((1, 3, 320, 160), dtype=np.float32))

    assert model.patch_embed.grid_shape(320, 160) == (20, 10)
    assert tokens.shape == (1, 201, 32)


def test_fixture_seed_changes_pixels_without_global_rng() -> None:
    """The fixture is seed-stable, seed-sensitive, and RNG-isolated."""
    first = synthesize_image(seed=77, size=48)
    second = synthesize_image(seed=77, size=48)
    different = synthesize_image(seed=78, size=48)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (1, 3, 48, 48)
    assert first.dtype == np.float32

    np.random.seed(1234)
    expected_stream = [float(np.random.random()) for _ in range(3)]
    np.random.seed(1234)
    synthesize_image(seed=9, size=8)
    observed_stream = [float(np.random.random()) for _ in range(3)]
    assert observed_stream == expected_stream


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
