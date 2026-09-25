"""Tests for the NumPy patch-tokenizer front end."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from code.main import (
    Conv2d,
    PatchEmbed,
    VisionFrontEnd,
    sinusoidal_2d,
    synthesize_image,
)


def unfold_patches(
    x: np.ndarray,
    kernel_size: int,
    stride: int | None = None,
) -> np.ndarray:
    """Independent unfold reference returning ``(B, C*k*k, L)`` like torch."""
    step = kernel_size if stride is None else stride
    batch, channels, height, width = x.shape
    out_h = (height - kernel_size) // step + 1
    out_w = (width - kernel_size) // step + 1
    windows = sliding_window_view(x, (kernel_size, kernel_size), axis=(2, 3))
    windows = windows[:, :, ::step, ::step]
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        batch, out_h * out_w, channels * kernel_size * kernel_size
    )
    return columns.transpose(0, 2, 1)


class ReferenceLinear:
    """Test-only affine layer with explicit weight/bias arrays."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T + self.bias


def sinusoidal_reference(positions: np.ndarray, dim: int) -> np.ndarray:
    """Independent 1D sinusoidal computation used to pin the 2D layout."""
    frequencies = np.exp(-np.log(10_000.0) * np.arange(0, dim, 2) / dim)
    angles = positions.astype(np.float64)[:, None] * frequencies[None, :]
    table = np.empty((positions.shape[0], dim), dtype=np.float64)
    table[:, 0::2] = np.sin(angles)
    table[:, 1::2] = np.cos(angles)
    return table


class PatchEmbedTests(unittest.TestCase):
    def test_patch_and_sequence_counts(self) -> None:
        model = VisionFrontEnd(patch_size=16, hidden=32, seed=1)
        image = np.zeros((1, 3, 224, 224), dtype=np.float32)
        tokens = model(image)

        self.assertEqual(model.patch_embed.grid_shape(224, 224), (14, 14))
        self.assertEqual(tokens.shape, (1, 197, 32))

    def test_batch_shape_and_dtype(self) -> None:
        model = VisionFrontEnd(patch_size=16, hidden=32, seed=2)
        image = np.zeros((2, 3, 224, 224), dtype=np.float32)
        tokens = model(image)

        self.assertEqual(tokens.shape, (2, 197, 32))
        self.assertEqual(tokens.dtype, np.dtype(np.float32))

    def test_conv_matches_unfold_linear(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.normal(size=(2, 3, 8, 12)).astype(np.float32)
        conv = Conv2d(3, 5, kernel_size=4, seed=7)
        conv_output = conv(image)

        unfolded = unfold_patches(image, kernel_size=4)
        linear = ReferenceLinear(conv.weight.reshape(5, -1), conv.bias)
        linear_output = linear(unfolded.transpose(0, 2, 1))
        expected = linear_output.reshape(2, 2, 3, 5).transpose(0, 3, 1, 2)

        np.testing.assert_allclose(conv_output, expected, rtol=1e-5, atol=1e-6)

    def test_conv_matches_manual_patch_loop(self) -> None:
        image = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        conv = Conv2d(1, 3, kernel_size=2, seed=3)
        output = conv(image)

        for row in range(2):
            for col in range(2):
                patch = image[0, :, row * 2 : (row + 1) * 2, col * 2 : (col + 1) * 2]
                expected = (
                    np.einsum("cuv,ocuv->o", patch, conv.weight) + conv.bias
                )
                np.testing.assert_allclose(
                    output[0, :, row, col], expected, rtol=1e-6, atol=1e-6
                )

    def test_non_square_image_grid(self) -> None:
        model = VisionFrontEnd(patch_size=16, hidden=32, seed=2)
        image = np.zeros((2, 3, 96, 64), dtype=np.float32)
        tokens = model(image)

        self.assertEqual(model.patch_embed.grid_shape(96, 64), (6, 4))
        self.assertEqual(tokens.shape, (2, 25, 32))
        self.assertEqual(sinusoidal_2d(6, 4, 32).shape, (24, 32))


class SinusoidalPositionTests(unittest.TestCase):
    def test_position_is_deterministic(self) -> None:
        first = sinusoidal_2d(3, 5, 16)
        np.random.random(100)
        second = sinusoidal_2d(3, 5, 16)

        np.testing.assert_array_equal(first, second)

    def test_position_norm_and_structure(self) -> None:
        grid_h, grid_w, dim = 7, 5, 16
        positions = sinusoidal_2d(grid_h, grid_w, dim)
        norms = np.linalg.norm(positions, axis=1)

        np.testing.assert_allclose(
            norms, np.sqrt(dim / 2), rtol=0.0, atol=1e-4
        )

        grid = positions.reshape(grid_h, grid_w, dim)
        row_half = grid[:, :, : dim // 2]
        col_half = grid[:, :, dim // 2 :]
        for row in range(grid_h):
            np.testing.assert_allclose(
                row_half[row],
                np.broadcast_to(row_half[row, 0], (grid_w, dim // 2)),
                rtol=0.0,
                atol=1e-6,
            )
        for col in range(grid_w):
            np.testing.assert_allclose(
                col_half[:, col],
                np.broadcast_to(col_half[0, col], (grid_h, dim // 2)),
                rtol=0.0,
                atol=1e-6,
            )

        self.assertEqual(np.unique(positions, axis=0).shape[0], grid_h * grid_w)

    def test_position_matches_manual_formula(self) -> None:
        grid_h, grid_w, dim = 4, 6, 12
        positions = sinusoidal_2d(grid_h, grid_w, dim)
        row_half = sinusoidal_reference(np.arange(grid_h), dim // 2)
        col_half = sinusoidal_reference(np.arange(grid_w), dim // 2)
        expected = np.concatenate((row_half[2], col_half[3])).astype(np.float32)

        np.testing.assert_allclose(
            positions[2 * grid_w + 3], expected, rtol=0.0, atol=1e-6
        )


class VisionFrontEndTests(unittest.TestCase):
    def test_cls_broadcast_without_cross_sample_leakage(self) -> None:
        model = VisionFrontEnd(patch_size=4, hidden=8, seed=5)
        rng = np.random.default_rng(11)
        images = rng.normal(size=(3, 3, 8, 8)).astype(np.float32)
        output = model(images)

        singles = np.concatenate([model(images[i : i + 1]) for i in range(3)])
        np.testing.assert_allclose(output, singles, rtol=0.0, atol=1e-6)

        cls = model.cls_token.reshape(-1)
        np.testing.assert_allclose(
            output[:, 0], np.broadcast_to(cls, (3, model.hidden)), rtol=0.0, atol=0.0
        )

        perturbed = images.copy()
        perturbed[1, :, 0, 0] += 1_000.0
        changed = model(perturbed)
        np.testing.assert_array_equal(changed[0], output[0])
        np.testing.assert_array_equal(changed[2], output[2])
        np.testing.assert_array_equal(changed[1, 0], output[1, 0])
        self.assertFalse(np.allclose(changed[1, 1], output[1, 1]))

    def test_parameter_counts_match_vit_base(self) -> None:
        model = VisionFrontEnd(seed=0)

        projection_parameters = sum(
            int(parameter.size) for parameter in model.patch_embed.parameters()
        )
        total_parameters = sum(
            int(parameter.size) for parameter in model.parameters()
        )

        self.assertEqual(model.patch_embed.proj.weight.shape, (768, 3, 16, 16))
        self.assertEqual(projection_parameters, 590_592)
        self.assertEqual(int(model.cls_token.size), 768)
        self.assertEqual(total_parameters, 591_360)
        self.assertEqual(len(model.parameters()), 3)

    def test_front_end_matches_manual_composition(self) -> None:
        model = VisionFrontEnd(patch_size=4, hidden=8, seed=9)
        rng = np.random.default_rng(4)
        images = rng.normal(size=(2, 3, 8, 8)).astype(np.float32)
        output = model(images)

        conv = model.patch_embed.proj
        unfolded = unfold_patches(images, kernel_size=4).transpose(0, 2, 1)
        linear = ReferenceLinear(conv.weight.reshape(model.hidden, -1), conv.bias)
        patch_tokens = linear(unfolded) + sinusoidal_2d(2, 2, model.hidden)[None]
        cls = np.broadcast_to(model.cls_token, (2, 1, model.hidden))
        expected = np.concatenate((cls, patch_tokens), axis=1)

        np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-6)


class FixtureAndValidationTests(unittest.TestCase):
    def test_synthesize_image_deterministic(self) -> None:
        first = synthesize_image(seed=123, size=32)
        second = synthesize_image(seed=123, size=32)
        different = synthesize_image(seed=124, size=32)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.shape, (1, 3, 32, 32))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLess(float(first.max()), 1.0)
        self.assertEqual(synthesize_image(seed=1).shape, (1, 3, 224, 224))

        np.random.seed(999)
        expected_stream = [float(np.random.random()) for _ in range(3)]
        np.random.seed(999)
        synthesize_image(seed=5)
        observed_stream = [float(np.random.random()) for _ in range(3)]
        self.assertEqual(expected_stream, observed_stream)

    def test_input_and_argument_validation(self) -> None:
        model = VisionFrontEnd(patch_size=16, hidden=32, seed=0)

        with self.assertRaises(ValueError):
            model(np.zeros((1, 4, 32, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            model(np.zeros((1, 3, 30, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            model(np.zeros((1, 3, 32, 32, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            sinusoidal_2d(0, 4, 16)
        with self.assertRaises(ValueError):
            sinusoidal_2d(4, 4, 30)
        with self.assertRaises(ValueError):
            sinusoidal_2d(4, 4, 0)
        with self.assertRaises(ValueError):
            sinusoidal_2d(4, 4, 16, base=0.0)
        with self.assertRaises(ValueError):
            Conv2d(3, 4, kernel_size=0)
        with self.assertRaises(ValueError):
            Conv2d(3, 4, kernel_size=16, stride=0)
        with self.assertRaises(ValueError):
            PatchEmbed(patch_size=0)
        with self.assertRaises(ValueError):
            VisionFrontEnd(hidden=30)


if __name__ == "__main__":
    unittest.main()
