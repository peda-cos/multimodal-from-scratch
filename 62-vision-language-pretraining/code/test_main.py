"""Detailed tests for losses, gradients, shapes, and training behavior."""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np

from code.main import (
    BOS_ID,
    CAPTION_LEN,
    CONTENT_LEN,
    CONTENT_OFFSET,
    CONTENT_VOCAB,
    DEFAULT_LR,
    EOS_ID,
    IMAGE_SIZE,
    PAD_ID,
    Adam,
    Attention,
    CrossAttentionDecoder,
    ImageProjector,
    LayerNorm,
    MultimodalModel,
    SmallViT,
    TextEncoder,
    clip_grad_norm,
    info_nce_loss,
    info_nce_loss_and_grad,
    lm_loss,
    lm_loss_and_grad,
    make_mock_corpus,
    patchify,
    similarity_accuracy,
    train,
    unpatchify,
)


def _central_difference(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Central finite differences for an arbitrary-shape float64 array."""
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = float(values[index])
        values[index] = original + eps
        plus = float(function(values))
        values[index] = original - eps
        minus = float(function(values))
        values[index] = original
        gradient[index] = (plus - minus) / (2.0 * eps)
    return gradient


def _small_model() -> MultimodalModel:
    return MultimodalModel(
        vision_encoder=SmallViT(
            hidden=16,
            heads=2,
            layers=1,
            ff_hidden=32,
            seed=0,
        ),
        image_projector=ImageProjector(16, 32, 16, seed=1),
        text_encoder=TextEncoder(32, 16, 16, seed=2),
        decoder=CrossAttentionDecoder(
            vocab_size=32,
            hidden=32,
            heads=4,
            layers=1,
            ff_hidden=64,
            max_seq_len=10,
            memory_dim=16,
            seed=3,
        ),
        init_temperature=0.07,
    )


class InfoNCETests(unittest.TestCase):
    def test_symmetric_under_modality_swap(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.normal(size=(5, 4))
        text = rng.normal(size=(5, 4))

        forward = info_nce_loss(image, text, 0.2)
        reverse = info_nce_loss(text, image, 0.2)

        self.assertAlmostEqual(forward, reverse, places=10)
        self.assertGreaterEqual(forward, 0.0)

    def test_perfect_diagonal_and_uniform_baseline(self) -> None:
        identity = np.eye(4, dtype=np.float64)
        diagonal_loss = info_nce_loss(identity, identity, 0.05)
        self.assertLess(abs(diagonal_loss), 1e-6)

        uniform = np.ones((4, 3), dtype=np.float64)
        uniform_loss = info_nce_loss(uniform, uniform, 0.2)
        self.assertAlmostEqual(uniform_loss, math.log(4.0), places=6)

    def test_single_pair_is_zero(self) -> None:
        self.assertEqual(
            info_nce_loss(
                np.ones((1, 3), dtype=np.float64),
                np.ones((1, 3), dtype=np.float64),
                0.2,
            ),
            0.0,
        )

    def test_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.normal(size=(4, 3))
        text = rng.normal(size=(4, 3))
        temperature = 0.2
        _, d_image, d_text, d_temperature = info_nce_loss_and_grad(
            image, text, temperature
        )

        numeric_image = _central_difference(
            lambda values: info_nce_loss(values, text, temperature),
            image.copy(),
            eps=1e-6,
        )
        numeric_text = _central_difference(
            lambda values: info_nce_loss(image, values, temperature),
            text.copy(),
            eps=1e-6,
        )
        numeric_temperature = _central_difference(
            lambda values: info_nce_loss(
                image, text, float(values)
            ),
            np.array(temperature, dtype=np.float64),
            eps=1e-6,
        )

        np.testing.assert_allclose(
            d_image, numeric_image, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            d_text, numeric_text, rtol=1e-6, atol=1e-8
        )
        self.assertAlmostEqual(
            d_temperature, float(numeric_temperature), places=6
        )

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            info_nce_loss(
                np.ones((3, 2)), np.ones((4, 2)), 0.2
            )
        with self.assertRaisesRegex(ValueError, "zero"):
            info_nce_loss(
                np.zeros((3, 2)), np.ones((3, 2)), 0.2
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            info_nce_loss(
                np.ones((3, 2)), np.ones((3, 2)), 0.0
            )
        with self.assertRaises(ValueError):
            info_nce_loss(
                np.array(
                    [[1.0, 2.0], [np.nan, 1.0]], dtype=np.float64
                ),
                np.ones((2, 2)),
                0.2,
            )


class LMLossTests(unittest.TestCase):
    def test_masks_padding_and_matches_manual(self) -> None:
        logits = np.array(
            [
                [
                    [0.2, -0.1, 0.5, 0.0, 0.3],
                    [0.1, 0.4, -0.2, 0.6, 0.0],
                    [0.7, 0.2, 0.1, -0.3, 0.4],
                ],
                [
                    [-0.2, 0.3, 0.6, 0.1, 0.0],
                    [0.5, -0.4, 0.2, 0.3, 0.1],
                    [0.0, 0.2, 0.4, -0.1, 0.5],
                ],
            ],
            dtype=np.float64,
        )
        targets = np.array(
            [[1, 2, PAD_ID], [3, PAD_ID, PAD_ID]], dtype=np.int64
        )

        loss = lm_loss(logits, targets, PAD_ID)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(
            np.exp(shifted).sum(axis=-1, keepdims=True)
        )
        active = targets != PAD_ID
        gathered = np.take_along_axis(
            log_probs, targets[..., None], axis=-1
        )[..., 0]
        expected = -float(gathered[active].mean())
        self.assertAlmostEqual(loss, expected, places=10)

    def test_gradient_masks_padding(self) -> None:
        rng = np.random.default_rng(8)
        logits = rng.normal(size=(2, 3, 5))
        targets = np.array(
            [[1, 2, PAD_ID], [3, PAD_ID, PAD_ID]], dtype=np.int64
        )
        loss, dlogits = lm_loss_and_grad(logits, targets, PAD_ID)

        numeric = _central_difference(
            lambda values: lm_loss(values, targets, PAD_ID),
            logits.copy(),
            eps=1e-6,
        )

        np.testing.assert_allclose(
            dlogits, numeric, rtol=1e-6, atol=1e-8
        )
        self.assertTrue(np.all(dlogits[1, 1:] == 0.0))
        self.assertTrue(np.all(dlogits[0, 2] == 0.0))
        np.testing.assert_allclose(
            dlogits.sum(axis=-1)[targets != PAD_ID],
            0.0,
            rtol=0.0,
            atol=1e-7,
        )
        self.assertGreater(loss, 0.0)

    def test_validation(self) -> None:
        logits = np.zeros((2, 3, 5))
        with self.assertRaisesRegex(ValueError, "no non-padding"):
            lm_loss(
                logits,
                np.zeros((2, 3), dtype=np.int64),
                PAD_ID,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            lm_loss(
                logits,
                np.array(
                    [[1, 2, 5], [3, 4, 1]], dtype=np.int64
                ),
                PAD_ID,
            )
        with self.assertRaises(ValueError):
            lm_loss(
                logits,
                np.array([[1, 2], [3, 4]], dtype=np.int64),
                PAD_ID,
            )
        with self.assertRaises(ValueError):
            lm_loss(
                logits,
                np.array(
                    [[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]
                ),
                PAD_ID,
            )


class GenericLayerTests(unittest.TestCase):
    def test_layer_norm_gradient(self) -> None:
        rng = np.random.default_rng(11)
        norm = LayerNorm(4)
        values = rng.normal(size=(2, 3, 4))
        weights = rng.normal(size=(2, 3, 4))
        norm.forward(values)
        dinput = norm.backward(weights)

        numeric = _central_difference(
            lambda candidate: float(
                np.sum(norm.forward(candidate) * weights)
            ),
            values.copy(),
            eps=1e-3,
        )
        np.testing.assert_allclose(
            dinput, numeric, rtol=5e-3, atol=5e-4
        )

    def test_attention_gradient_and_causality(self) -> None:
        rng = np.random.default_rng(12)
        attention = Attention(8, 2, seed=5)
        values = rng.normal(size=(2, 4, 8))
        weights = rng.normal(size=(2, 4, 8))
        attention.forward(values, causal=True)
        dquery, dkey_value = attention.backward(weights)
        dinput = dquery + dkey_value

        numeric = _central_difference(
            lambda candidate: float(
                np.sum(attention.forward(candidate, causal=True) * weights)
            ),
            values.copy(),
            eps=1e-3,
        )
        np.testing.assert_allclose(
            dinput, numeric, rtol=2e-2, atol=1e-4
        )

    def test_patchify_roundtrip_and_order(self) -> None:
        rng = np.random.default_rng(13)
        images = rng.normal(size=(2, 1, 8, 8)).astype(np.float32)
        patches = patchify(images, 2)

        self.assertEqual(patches.shape, (2, 16, 4))
        np.testing.assert_allclose(
            patches[0, 0],
            images[0, :, 0:2, 0:2].reshape(-1),
            rtol=0.0,
            atol=0.0,
        )
        restored = unpatchify(patches, 1, 8, 8, 2)
        np.testing.assert_allclose(
            restored, images, rtol=0.0, atol=0.0
        )


class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_losses(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        output = model.forward(corpus.images[:4], corpus.captions[:4])
        contrastive, language_model, total = model.losses(output)

        self.assertEqual(output.image_emb.shape, (4, 16))
        self.assertEqual(output.text_emb.shape, (4, 16))
        self.assertEqual(output.logits.shape, (4, 9, 32))
        self.assertEqual(output.targets.shape, (4, 9))
        self.assertGreaterEqual(contrastive, 0.0)
        self.assertGreaterEqual(language_model, 0.0)
        self.assertAlmostEqual(
            total, contrastive + language_model, places=10
        )
        self.assertTrue(np.isfinite(output.logits).all())

    def test_backward_populates_finite_gradients(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        model.loss_and_backward(
            corpus.images[:8], corpus.captions[:8]
        )

        nonzero = 0
        for parameter in model.parameters():
            self.assertEqual(parameter.grad.shape, parameter.data.shape)
            self.assertTrue(np.isfinite(parameter.grad).all())
            if np.any(parameter.grad != 0.0):
                nonzero += 1
        self.assertGreater(nonzero, 0)
        self.assertTrue(
            np.isfinite(model.log_temperature.grad).all()
        )

    def test_contrastive_only_does_not_touch_decoder(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
        decoder_before = [
            parameter.data.copy()
            for parameter in model.decoder.parameters()
        ]
        other_before = [
            parameter.data.copy()
            for parameter in (
                model.vision_encoder.parameters()
                + model.image_projector.parameters()
                + model.text_encoder.parameters()
            )
        ]

        model.loss_and_backward(
            corpus.images[:8], corpus.captions[:8], lm_weight=0.0
        )
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

        for before, parameter in zip(
            decoder_before, model.decoder.parameters()
        ):
            np.testing.assert_array_equal(before, parameter.data)
        changed = [
            not np.array_equal(before, parameter.data)
            for before, parameter in zip(
                other_before,
                (
                    model.vision_encoder.parameters()
                    + model.image_projector.parameters()
                    + model.text_encoder.parameters()
                ),
            )
        ]
        self.assertTrue(any(changed))

    def test_five_step_training_reduces_combined_loss(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
        images = corpus.images
        captions = corpus.captions
        before = model.losses(model.forward(images, captions))

        for _ in range(5):
            model.loss_and_backward(images, captions)
            clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        after = model.losses(model.forward(images, captions))
        self.assertLess(after[2], before[2])

    def test_train_reports_five_step_progress(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        report = train(
            model,
            corpus,
            steps=5,
            batch_size=8,
            seed=0,
            log_every=10,
            print_fn=lambda _: None,
        )

        self.assertEqual(len(report.records), 5)
        self.assertLess(report.final.total, report.initial.total)
        self.assertGreaterEqual(report.final.contrastive, 0.0)
        self.assertGreaterEqual(report.final.lm, 0.0)
        self.assertGreaterEqual(report.final.temperature, 0.01)
        self.assertLessEqual(report.final.temperature, 1.0)

    def test_temperature_stays_in_range_and_moves(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(0, 8)
        optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
        before = float(model.log_temperature.data.item())
        model.loss_and_backward(
            corpus.images[:8], corpus.captions[:8]
        )
        self.assertTrue(
            np.isfinite(model.log_temperature.grad).all()
        )
        self.assertNotEqual(float(model.log_temperature.grad.item()), 0.0)
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        after = float(model.log_temperature.data.item())
        tau, _ = model.temperature()

        self.assertNotEqual(before, after)
        self.assertGreaterEqual(tau, 0.01)
        self.assertLessEqual(tau, 1.0)

    def test_causal_attention_does_not_leak_future(self) -> None:
        model = _small_model()
        rng = np.random.default_rng(14)
        memory = rng.normal(size=(2, 17, 16)).astype(np.float32)
        ids = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 3, 5, 7, 9, 11, 13, 15, 17],
            ],
            dtype=np.int64,
        )
        changed = ids.copy()
        changed[:, -1] = 18
        first = model.decoder.forward(ids, memory)
        second = model.decoder.forward(changed, memory)

        np.testing.assert_allclose(
            first[:, :-1], second[:, :-1], rtol=0.0, atol=0.0
        )
        self.assertGreater(
            float(np.max(np.abs(first[:, -1] - second[:, -1]))),
            1e-6,
        )

    def test_memory_changes_logits(self) -> None:
        model = _small_model()
        rng = np.random.default_rng(15)
        ids = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64
        )
        memory_a = rng.normal(size=(1, 17, 16)).astype(np.float32)
        memory_b = rng.normal(size=(1, 17, 16)).astype(np.float32)
        logits_a = model.decoder.forward(ids, memory_a)
        logits_b = model.decoder.forward(ids, memory_b)

        self.assertGreater(
            float(np.max(np.abs(logits_a - logits_b))), 1e-5
        )

    def test_similarity_accuracy(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        perfect = similarity_accuracy(identity, identity)
        shifted_text = identity[[1, 2, 0]]
        broken = similarity_accuracy(identity, shifted_text)

        self.assertEqual(perfect, (1.0, 1.0))
        self.assertEqual(broken, (0.0, 0.0))
        with self.assertRaises(ValueError):
            similarity_accuracy(
                np.zeros((2, 3)), np.ones((2, 3))
            )


class CorpusTests(unittest.TestCase):
    def test_deterministic_and_well_formed(self) -> None:
        first = make_mock_corpus(7, 12)
        second = make_mock_corpus(7, 12)
        other = make_mock_corpus(8, 12)

        np.testing.assert_array_equal(first.images, second.images)
        np.testing.assert_array_equal(first.captions, second.captions)
        self.assertFalse(np.array_equal(first.images, other.images))
        self.assertEqual(first.images.shape, (12, 1, 8, 8))
        self.assertEqual(first.captions.shape, (12, CAPTION_LEN))
        self.assertEqual(first.images.dtype, np.dtype(np.float32))
        self.assertEqual(first.captions.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(
            first.captions[:, 0], np.full(12, BOS_ID)
        )
        np.testing.assert_array_equal(
            first.captions[:, 7], np.full(12, EOS_ID)
        )
        np.testing.assert_array_equal(
            first.captions[:, 8:], np.full((12, 2), PAD_ID)
        )
        content = first.captions[:, 1 : 1 + CONTENT_LEN]
        self.assertTrue((content >= CONTENT_OFFSET).all())
        self.assertTrue(
            (content < CONTENT_OFFSET + CONTENT_VOCAB).all()
        )
        self.assertTrue(np.isfinite(first.images).all())

    def test_renders_tokens_in_patch_order(self) -> None:
        corpus = make_mock_corpus(3, 4)
        image = corpus.images[0, 0]
        tokens = corpus.captions[0, 1 : 1 + CONTENT_LEN]

        for position, token in enumerate(tokens):
            index = int(token) - CONTENT_OFFSET
            angle = 2.0 * math.pi * index / CONTENT_VOCAB
            expected = np.array(
                [
                    math.cos(angle),
                    math.sin(angle),
                    math.cos(2.0 * angle),
                    math.sin(2.0 * angle),
                ],
                dtype=np.float32,
            ).reshape(2, 2)
            row = (position // 2) * 2
            column = (position % 2) * 2
            np.testing.assert_allclose(
                image[row : row + 2, column : column + 2],
                expected,
                rtol=0.0,
                atol=0.1,
            )

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            make_mock_corpus(-1, 4)
        with self.assertRaises(ValueError):
            make_mock_corpus(0, 0)


if __name__ == "__main__":
    unittest.main()
