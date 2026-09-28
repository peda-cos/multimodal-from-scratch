"""Detailed tests for the projection layer, loss, and training loop."""

from __future__ import annotations

import unittest

import numpy as np

from code.main import (
    CAPTION_LEN,
    MLPProjector,
    MockTextEmbedding,
    MockVisionEncoder,
    NUM_TOKENS,
    TEXT_HIDDEN,
    VISION_HIDDEN,
    cls_pool,
    cosine_alignment_grad,
    cosine_alignment_loss,
    cosine_alignment_loss_and_grad,
    make_pair,
    sgd_step,
    train,
)


class ProjectorTests(unittest.TestCase):
    def test_output_shape_and_exact_parameter_count(self) -> None:
        projector = MLPProjector(768, 1024, 512, seed=0)
        features = np.zeros((4, 768), dtype=np.float32)

        output = projector(features)

        self.assertEqual(output.shape, (4, 512))
        self.assertEqual(projector.num_parameters(), 1_312_256)
        self.assertEqual(
            [parameter.data.shape for parameter in projector.parameters()],
            [(1024, 768), (1024,), (512, 1024), (512,)],
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in projector.parameters())
        )

    def test_gradient_flows_to_all_parameters_and_input(self) -> None:
        projector = MLPProjector(8, 12, 5, seed=2)
        rng = np.random.default_rng(10)
        features = rng.normal(size=(3, 8)).astype(np.float32)
        target = rng.normal(size=(3, 5)).astype(np.float32)

        projected = projector(features)
        loss, gradient = cosine_alignment_loss_and_grad(projected, target)
        projector.zero_grad()
        input_gradient = projector.backward(gradient)

        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(input_gradient.shape, features.shape)
        self.assertGreater(float(np.linalg.norm(input_gradient)), 0.0)
        for parameter in projector.parameters():
            self.assertEqual(parameter.grad.shape, parameter.data.shape)
            self.assertGreater(float(np.linalg.norm(parameter.grad)), 0.0)

    def test_sgd_step_uses_gradients_and_learning_rate(self) -> None:
        projector = MLPProjector(4, 3, 2, seed=0)
        for parameter in projector.parameters():
            parameter.grad.fill(1.0)
        before = [parameter.data.copy() for parameter in projector.parameters()]

        sgd_step(projector, 0.5)

        for old, parameter in zip(before, projector.parameters(), strict=True):
            np.testing.assert_allclose(parameter.data, old - 0.5, rtol=0.0, atol=1e-7)

    def test_rejects_bad_input_and_backward_order(self) -> None:
        projector = MLPProjector(4, 3, 2, seed=0)
        with self.assertRaises(ValueError):
            projector(np.zeros((2, 5), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            projector.backward(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            MLPProjector(0, 3, 2)


class TextTableTests(unittest.TestCase):
    def test_table_is_deterministic_and_frozen(self) -> None:
        first = MockTextEmbedding(64, 16, seed=3)
        second = MockTextEmbedding(64, 16, seed=3)
        third = MockTextEmbedding(64, 16, seed=4)

        np.testing.assert_array_equal(first.table, second.table)
        self.assertFalse(np.array_equal(first.table, third.table))
        self.assertFalse(first.requires_grad)
        parameters = first.parameters()
        self.assertEqual(len(parameters), 1)
        self.assertFalse(parameters[0].requires_grad)
        self.assertEqual(int(np.count_nonzero(parameters[0].grad)), 0)

    def test_caption_embedding_is_mean_pooled(self) -> None:
        table = MockTextEmbedding(32, 8, seed=1)
        ids = np.array([1, 5, 7], dtype=np.int64)
        expected = table.table[ids].mean(axis=0)

        pooled = table.caption_embedding(ids)
        batched = table.caption_embedding(
            np.array([[1, 5, 7], [2, 3, 4]], dtype=np.int64)
        )

        np.testing.assert_allclose(pooled, expected, rtol=0.0, atol=1e-7)
        self.assertEqual(batched.shape, (2, 8))
        np.testing.assert_allclose(batched[0], expected, rtol=0.0, atol=1e-7)

    def test_rejects_out_of_range_and_empty_captions(self) -> None:
        table = MockTextEmbedding(32, 8, seed=1)
        with self.assertRaises(ValueError):
            table.caption_embedding(np.array([32], dtype=np.int64))
        with self.assertRaises(ValueError):
            table.caption_embedding(np.array([], dtype=np.int64))
        with self.assertRaises(ValueError):
            table.embed(np.zeros((1, 2, 3), dtype=np.int64))


class LossTests(unittest.TestCase):
    def test_zero_anti_parallel_and_orthogonal(self) -> None:
        paired = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        orthogonal_image = np.array([[1.0, 0.0]], dtype=np.float32)
        orthogonal_text = np.array([[0.0, 1.0]], dtype=np.float32)

        self.assertAlmostEqual(cosine_alignment_loss(paired, paired), 0.0, places=6)
        self.assertAlmostEqual(cosine_alignment_loss(paired, -paired), 2.0, places=6)
        self.assertAlmostEqual(
            cosine_alignment_loss(orthogonal_image, orthogonal_text),
            1.0,
            places=6,
        )

    def test_one_dimensional_inputs_are_one_pair(self) -> None:
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        self.assertAlmostEqual(
            cosine_alignment_loss(vector, vector), 0.0, places=6
        )
        self.assertEqual(cosine_alignment_grad(vector, vector).shape, (1, 3))

    def test_rejects_mismatched_shapes(self) -> None:
        image = np.zeros((2, 3), dtype=np.float32)
        text = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            cosine_alignment_loss(image, text)
        with self.assertRaises(ValueError):
            cosine_alignment_loss(np.zeros((2, 3, 1), dtype=np.float32), image)


class GradientTests(unittest.TestCase):
    def test_projector_backward_matches_finite_differences(self) -> None:
        projector = MLPProjector(6, 5, 4, seed=3)
        rng = np.random.default_rng(20)
        features = rng.normal(size=(3, 6)).astype(np.float32)
        upstream = rng.normal(size=(3, 4)).astype(np.float32)
        projector.zero_grad()
        projector(features)
        projector.backward(upstream)

        checks = [
            (projector.fc1, (0, 0)),
            (projector.fc1, (4, 5)),
            (projector.b1, (2,)),
            (projector.fc2, (1, 3)),
            (projector.b2, (0,)),
        ]
        for parameter, index in checks:
            old = parameter.data[index]
            epsilon = 1e-2
            parameter.data[index] = old + epsilon
            plus = float(np.sum(projector(features) * upstream))
            parameter.data[index] = old - epsilon
            minus = float(np.sum(projector(features) * upstream))
            parameter.data[index] = old
            numeric = (plus - minus) / (2.0 * epsilon)
            analytic = float(parameter.grad[index])
            self.assertAlmostEqual(numeric, analytic, delta=1e-3)

    def test_cosine_gradient_matches_finite_differences(self) -> None:
        image = np.array(
            [[0.3, -0.2, 1.1], [0.4, 0.7, -0.5]], dtype=np.float32
        )
        text = np.array(
            [[0.8, 0.1, 0.2], [-0.3, 0.6, 0.9]], dtype=np.float32
        )
        _, gradient = cosine_alignment_loss_and_grad(image, text)

        for index in ((0, 0), (1, 2)):
            old = image[index]
            epsilon = 1e-2
            image[index] = old + epsilon
            plus = cosine_alignment_loss(image, text)
            image[index] = old - epsilon
            minus = cosine_alignment_loss(image, text)
            image[index] = old
            numeric = (plus - minus) / (2.0 * epsilon)
            self.assertAlmostEqual(
                numeric, float(gradient[index]), delta=1e-3
            )


class FixtureTests(unittest.TestCase):
    def test_make_pair_shapes_and_determinism(self) -> None:
        table = MockTextEmbedding(128, TEXT_HIDDEN, seed=0)
        encoder = MockVisionEncoder(
            TEXT_HIDDEN, VISION_HIDDEN, NUM_TOKENS, seed=1
        )

        first_tokens, first_ids = make_pair(
            7, 128, text_table=table, vision_encoder=encoder
        )
        second_tokens, second_ids = make_pair(
            7, 128, text_table=table, vision_encoder=encoder
        )
        third_tokens, _ = make_pair(
            8, 128, text_table=table, vision_encoder=encoder
        )
        default_tokens, default_ids = make_pair(3, 64)

        self.assertEqual(first_tokens.shape, (NUM_TOKENS, VISION_HIDDEN))
        self.assertEqual(first_ids.shape, (CAPTION_LEN,))
        self.assertEqual(default_tokens.shape, (NUM_TOKENS, VISION_HIDDEN))
        self.assertEqual(default_ids.shape, (CAPTION_LEN,))
        np.testing.assert_array_equal(first_tokens, second_tokens)
        np.testing.assert_array_equal(first_ids, second_ids)
        self.assertFalse(np.array_equal(first_tokens, third_tokens))
        self.assertTrue(np.all(first_ids >= 0))
        self.assertTrue(np.all(first_ids < 128))

    def test_cls_pool_returns_only_the_first_token(self) -> None:
        tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        pooled = cls_pool(tokens)

        np.testing.assert_array_equal(pooled, tokens[:, 0, :])
        with self.assertRaises(ValueError):
            cls_pool(tokens[0])


class TrainingTests(unittest.TestCase):
    @staticmethod
    def _pairs(
        count: int,
    ) -> tuple[
        MockTextEmbedding,
        MockVisionEncoder,
        list[tuple[np.ndarray, np.ndarray]],
    ]:
        table = MockTextEmbedding(128, TEXT_HIDDEN, seed=0)
        encoder = MockVisionEncoder(
            TEXT_HIDDEN, VISION_HIDDEN, NUM_TOKENS, seed=1
        )
        pairs = [
            make_pair(seed, 128, text_table=table, vision_encoder=encoder)
            for seed in range(count)
        ]
        return table, encoder, pairs

    def test_training_reduces_loss_over_200_steps(self) -> None:
        table, _, pairs = self._pairs(8)
        projector = MLPProjector(VISION_HIDDEN, 64, TEXT_HIDDEN, seed=0)

        reports = train(
            projector,
            pairs,
            table,
            steps=200,
            batch_size=8,
            lr=0.1,
            report_every=25,
        )

        self.assertEqual(reports[0].step, 0)
        self.assertEqual(reports[-1].step, 200)
        self.assertEqual(
            [report.step for report in reports],
            [0, 25, 50, 75, 100, 125, 150, 175, 200],
        )
        self.assertLess(reports[-1].loss, reports[0].loss - 0.05)
        self.assertGreater(
            reports[-1].mean_cosine, reports[0].mean_cosine + 0.05
        )

    def test_training_leaves_frozen_modules_unchanged(self) -> None:
        table, encoder, pairs = self._pairs(4)
        projector = MLPProjector(VISION_HIDDEN, 32, TEXT_HIDDEN, seed=2)
        table_before = table.table.copy()
        projector_before = [
            parameter.data.copy() for parameter in projector.parameters()
        ]
        encoder_before = encoder.projection.data.copy()
        tokens_before = [tokens.copy() for tokens, _ in pairs]

        train(
            projector,
            pairs,
            table,
            steps=20,
            batch_size=4,
            lr=0.1,
            report_every=10,
        )

        np.testing.assert_array_equal(table.table, table_before)
        np.testing.assert_array_equal(encoder.projection.data, encoder_before)
        self.assertFalse(table.requires_grad)
        self.assertFalse(encoder.requires_grad)
        for before, (tokens, _) in zip(tokens_before, pairs, strict=True):
            np.testing.assert_array_equal(tokens, before)
        self.assertTrue(
            any(
                not np.array_equal(before, parameter.data)
                for before, parameter in zip(
                    projector_before, projector.parameters(), strict=True
                )
            )
        )

    def test_training_rejects_empty_pairs_and_bad_hyperparameters(self) -> None:
        table = MockTextEmbedding(64, TEXT_HIDDEN, seed=0)
        projector = MLPProjector(VISION_HIDDEN, 4, TEXT_HIDDEN, seed=0)
        pairs = [make_pair(0, 64, text_table=table)]
        with self.assertRaises(ValueError):
            train(projector, [], table)
        with self.assertRaises(ValueError):
            train(projector, pairs, table, steps=0)
        with self.assertRaises(ValueError):
            train(projector, pairs, table, batch_size=0)


if __name__ == "__main__":
    unittest.main()
