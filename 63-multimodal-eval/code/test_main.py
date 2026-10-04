"""Unit and integration tests for the multimodal evaluation surfaces."""

from __future__ import annotations

import unittest
from math import exp

import numpy as np

from code.main import (
    BOS_ID,
    CAPTION_LEN,
    CONTENT_LEN,
    CONTENT_OFFSET,
    CONTENT_VOCAB,
    DEFAULT_EVAL_SEED,
    DEFAULT_LR,
    EOS_ID,
    PAD_ID,
    VQA_HIDDEN,
    VOCAB_SIZE,
    Adam,
    CaptionHead,
    CrossAttentionDecoder,
    ImageProjector,
    MultimodalModel,
    SmallViT,
    TextEncoder,
    VQAHead,
    bleu4,
    build_default_model,
    build_eval_suite,
    content_patch_indices,
    content_token_indices,
    evaluate,
    make_mock_corpus,
    patchify,
    recall_at_k,
    render_image,
    train,
    vqa_exact_match,
)


def _small_model() -> MultimodalModel:
    small_vocab = CONTENT_OFFSET + CONTENT_VOCAB
    return MultimodalModel(
        vision_encoder=SmallViT(
            hidden=16,
            heads=2,
            layers=1,
            ff_hidden=32,
            seed=0,
        ),
        image_projector=ImageProjector(16, 32, 16, seed=1),
        text_encoder=TextEncoder(small_vocab, 16, 16, seed=2),
        decoder=CrossAttentionDecoder(
            vocab_size=small_vocab,
            hidden=32,
            heads=4,
            layers=1,
            ff_hidden=64,
            max_seq_len=CAPTION_LEN,
            memory_dim=16,
            seed=3,
        ),
        caption_head=CaptionHead(
            input_dim=16,
            vocab_size=CONTENT_VOCAB,
            hidden=32,
            seed=4,
        ),
        vqa_head=VQAHead(
            joint_dim=16,
            hidden=VQA_HIDDEN,
            answer_vocab=CONTENT_VOCAB,
            question_vocab=CONTENT_LEN,
            seed=5,
        ),
    )


def _noop(_: str) -> None:
    return None


class RecallTests(unittest.TestCase):
    def test_identity_and_reversed_diagonal(self) -> None:
        size = 6
        identity = np.eye(size)
        for k in (1, 3, 6):
            self.assertEqual(recall_at_k(identity, k), 1.0)
            self.assertEqual(
                recall_at_k(identity, k, direction="image_to_text"),
                1.0,
            )
        worst = np.zeros((size, size))
        np.fill_diagonal(worst, -1.0)
        self.assertEqual(recall_at_k(worst, 1), 0.0)
        self.assertEqual(recall_at_k(worst, 2), 0.0)
        self.assertEqual(recall_at_k(worst, size), 1.0)

    def test_directions_can_differ(self) -> None:
        scores = np.array(
            [
                [0.9, 0.0, 0.0],
                [0.8, 0.5, 0.1],
                [0.7, 0.2, 0.6],
            ]
        )
        self.assertAlmostEqual(recall_at_k(scores, 1), 1.0 / 3.0)
        self.assertAlmostEqual(
            recall_at_k(scores, 1, direction="image_to_text"), 1.0
        )

    def test_validation_and_k_upper_bound(self) -> None:
        scores = np.eye(4)
        for k in (0, -1, 5):
            with self.assertRaises(ValueError):
                recall_at_k(scores, k)
        with self.assertRaises(ValueError):
            recall_at_k(np.ones((2, 3)), 1)
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[np.nan]]), 1)
        with self.assertRaises(ValueError):
            recall_at_k(scores, 1, direction="sideways")
        self.assertEqual(recall_at_k(scores, 4), 1.0)


class VQATests(unittest.TestCase):
    def test_fraction_of_exact_matches(self) -> None:
        self.assertEqual(vqa_exact_match([1, 2, 3, 4], [1, 2, 9, 9]), 0.5)
        self.assertEqual(vqa_exact_match([5], [5]), 1.0)
        self.assertEqual(vqa_exact_match([5], [6]), 0.0)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            vqa_exact_match([], [])
        with self.assertRaises(ValueError):
            vqa_exact_match([1, 2], [1])


class BleuTests(unittest.TestCase):
    def test_exact_reference_is_one(self) -> None:
        generated = [1, 2, 3, 4]
        references = [[9, 9, 9, 9], [1, 2, 3, 4]]
        self.assertEqual(bleu4(generated, references), 1.0)

    def test_disjoint_vocabulary_is_zero(self) -> None:
        for smoothing in (True, False):
            self.assertEqual(
                bleu4([1, 2, 3, 4], [[5, 6, 7, 8]], smoothing),
                0.0,
            )

    def test_multi_reference_clipping_and_smoothing(self) -> None:
        score = bleu4([1, 1, 1, 1], [[1, 1, 2, 2]])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        self.assertAlmostEqual(score, 0.408248, places=5)
        smoothed = bleu4([1, 2, 3, 4], [[1, 2, 9, 9]], smoothing=True)
        unsmoothed = bleu4(
            [1, 2, 3, 4], [[1, 2, 9, 9]], smoothing=False
        )
        self.assertGreater(smoothed, 0.0)
        self.assertEqual(unsmoothed, 0.0)

    def test_brevity_penalty_uses_closest_reference(self) -> None:
        generated = [1, 2, 3, 4, 5]
        references = [[1, 2, 3, 4, 5, 6, 7, 8]]
        self.assertAlmostEqual(
            bleu4(generated, references), exp(1.0 - 8.0 / 5.0), places=6
        )

    def test_empty_generated_is_zero(self) -> None:
        self.assertEqual(bleu4([], [[1, 2, 3, 4]]), 0.0)
        with self.assertRaises(ValueError):
            bleu4([1], [])


class SuiteTests(unittest.TestCase):
    def test_counts_and_types(self) -> None:
        suite = build_eval_suite(
            DEFAULT_EVAL_SEED, 50, VOCAB_SIZE, CAPTION_LEN
        )
        self.assertEqual(len(suite.pairs), 50)
        self.assertEqual(len(suite.vqa), 50)
        self.assertEqual(len(suite.caps), 50)
        self.assertEqual(suite.content_len, CONTENT_LEN)
        self.assertEqual(len(list(suite)), 3)
        for pair in suite.pairs:
            self.assertEqual(pair.image.shape, (1, 8, 8))
            self.assertEqual(pair.caption_ids.shape, (CAPTION_LEN,))
        for sample in suite.vqa:
            self.assertEqual(sample.question_ids[0], BOS_ID)
            slot = int(sample.question_ids[-1]) - suite.question_offset
            self.assertTrue(0 <= slot < CONTENT_LEN)
            self.assertTrue(
                CONTENT_OFFSET
                <= sample.answer_id
                < CONTENT_OFFSET + CONTENT_VOCAB
            )
        for caption_sample in suite.caps:
            self.assertEqual(len(caption_sample.reference_ids), 3)
            for reference in caption_sample.reference_ids:
                self.assertEqual(reference[0], BOS_ID)
                self.assertEqual(reference[1 + CONTENT_LEN], EOS_ID)

    def test_deterministic_and_seed_sensitive(self) -> None:
        first = build_eval_suite(DEFAULT_EVAL_SEED, 8, VOCAB_SIZE, 10)
        second = build_eval_suite(DEFAULT_EVAL_SEED, 8, VOCAB_SIZE, 10)
        other = build_eval_suite(DEFAULT_EVAL_SEED + 1, 8, VOCAB_SIZE, 10)
        np.testing.assert_array_equal(
            first.pairs[0].image, second.pairs[0].image
        )
        np.testing.assert_array_equal(
            first.pairs[0].caption_ids, second.pairs[0].caption_ids
        )
        self.assertFalse(
            np.array_equal(first.pairs[0].image, other.pairs[0].image)
        )

    def test_short_caption_length(self) -> None:
        suite = build_eval_suite(3, 4, VOCAB_SIZE, 6)
        self.assertEqual(suite.content_len, 4)
        for sample in suite.vqa:
            slot = int(sample.question_ids[-1]) - suite.question_offset
            self.assertTrue(0 <= slot < 4)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_eval_suite(0, 0, VOCAB_SIZE, CAPTION_LEN)
        with self.assertRaises(ValueError):
            build_eval_suite(0, 4, 10, CAPTION_LEN)
        with self.assertRaises(ValueError):
            build_eval_suite(0, 4, VOCAB_SIZE, 2)
        with self.assertRaises(ValueError):
            build_eval_suite(-1, 4, VOCAB_SIZE, CAPTION_LEN)


class RenderTests(unittest.TestCase):
    def test_content_codes_land_in_expected_patch(self) -> None:
        tokens = [3, 4, 5, 6, 7, 8]
        rng = np.random.default_rng(0)
        image = render_image(tokens, rng)[0]
        patches = patchify(image[None, None, :, :], 2)[0]
        self.assertEqual(content_token_indices(2), (1, 2))
        for position, patch_index in enumerate(
            content_patch_indices(CONTENT_LEN)
        ):
            patch = patches[patch_index]
            self.assertAlmostEqual(
                float(patch[0]),
                float(np.cos(2.0 * np.pi * (tokens[position] - 3) / CONTENT_VOCAB)),
                places=1,
            )


class ModelIntegrationTests(unittest.TestCase):
    def test_forward_shapes_and_loss_breakdown(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(seed=7, n_pairs=16)
        output = model.forward(
            corpus.images[:4], corpus.captions[:4]
        )
        self.assertEqual(output.image_emb.shape, (4, 16))
        self.assertEqual(output.text_emb.shape, (4, 16))
        self.assertEqual(output.projected.shape, (4, 17, 16))
        self.assertEqual(output.vision_tokens.shape, (4, 17, 16))
        self.assertEqual(output.logits.shape, (4, CAPTION_LEN - 1, 67))
        losses = model.losses(output)
        self.assertGreaterEqual(losses.contrastive, 0.0)
        self.assertGreaterEqual(losses.lm, 0.0)
        self.assertGreaterEqual(losses.caption, 0.0)
        self.assertGreaterEqual(losses.vqa, 0.0)
        self.assertAlmostEqual(
            losses.total,
            losses.contrastive
            + losses.lm
            + losses.caption
            + losses.vqa,
            places=5,
        )

    def test_backward_populates_head_gradients(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(seed=8, n_pairs=16)
        model.loss_and_backward(
            corpus.images[:8], corpus.captions[:8]
        )
        for parameter in model.parameters():
            self.assertTrue(
                np.all(np.isfinite(parameter.grad)),
                parameter.name,
            )
        self.assertGreater(
            float(np.abs(model.caption_head.fc1.weight.grad).sum()),
            0.0,
        )
        self.assertGreater(
            float(
                np.abs(
                    model.vqa_head.question_embedding.table.grad
                ).sum()
            ),
            0.0,
        )

    def test_head_only_updates_reduce_caption_and_vqa_loss(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(seed=9, n_pairs=32)
        optimizer = Adam(model.head_parameters(), lr=3e-2)
        first = model.loss_and_backward(
            corpus.images[:16],
            corpus.captions[:16],
            lm_weight=0.0,
            caption_weight=1.0,
            vqa_weight=1.0,
        )
        latest = first
        for _ in range(30):
            latest = model.loss_and_backward(
                corpus.images[:16],
                corpus.captions[:16],
                lm_weight=0.0,
                caption_weight=1.0,
                vqa_weight=1.0,
            )
            optimizer.step()
        self.assertLess(latest.caption, first.caption * 0.5)
        self.assertLess(latest.vqa, first.vqa * 0.5)

    def test_evaluate_returns_bounded_metrics(self) -> None:
        model = build_default_model(seed=62)
        suite = build_eval_suite(
            DEFAULT_EVAL_SEED, 12, VOCAB_SIZE, CAPTION_LEN
        )
        first = evaluate(model, suite)
        second = evaluate(model, suite)
        self.assertEqual(first, second)
        self.assertEqual(first["n_samples"], 12.0)
        expected = {
            "retrieval_r1",
            "retrieval_r5",
            "retrieval_r10",
            "retrieval_r1_i2t",
            "retrieval_r5_i2t",
            "retrieval_r10_i2t",
            "vqa_em",
            "bleu4",
            "n_samples",
        }
        self.assertEqual(set(first), expected)
        for key, value in first.items():
            if key != "n_samples":
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_train_reduces_corpus_and_head_losses(self) -> None:
        model = _small_model()
        corpus = make_mock_corpus(seed=10, n_pairs=32)
        report = train(
            model,
            corpus,
            steps=8,
            batch_size=8,
            lr=DEFAULT_LR,
            head_lr=3e-2,
            log_every=100,
            print_fn=_noop,
        )
        self.assertEqual(len(report.records), 8)
        self.assertLess(report.final.total, report.initial.total)
        self.assertLess(report.final.caption, report.initial.caption)
        self.assertLess(report.final.vqa, report.initial.vqa)


class CorpusTests(unittest.TestCase):
    def test_mock_corpus_is_deterministic_and_well_formed(self) -> None:
        first = make_mock_corpus(seed=11, n_pairs=12)
        second = make_mock_corpus(seed=11, n_pairs=12)
        np.testing.assert_array_equal(first.images, second.images)
        np.testing.assert_array_equal(first.captions, second.captions)
        self.assertEqual(first.images.shape, (12, 1, 8, 8))
        self.assertEqual(first.captions.shape, (12, CAPTION_LEN))
        self.assertTrue(np.all(first.captions[:, 0] == BOS_ID))
        self.assertTrue(
            np.all(first.captions[:, 1 + CONTENT_LEN] == EOS_ID)
        )
        self.assertTrue(
            np.all(first.captions[:, 2 + CONTENT_LEN :] == PAD_ID)
        )
        content = first.captions[:, 1 : 1 + CONTENT_LEN]
        self.assertTrue(np.all(content >= CONTENT_OFFSET))
        self.assertTrue(
            np.all(content < CONTENT_OFFSET + CONTENT_VOCAB)
        )


if __name__ == "__main__":
    unittest.main()
