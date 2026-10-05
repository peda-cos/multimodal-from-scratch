"""Module-level contract tests counted by the portfolio verification gate."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from math import exp

import numpy as np

from code.main import (
    BOS_ID,
    CAPTION_LEN,
    CONTENT_LEN,
    CONTENT_OFFSET,
    CONTENT_VOCAB,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_EVAL_SEED,
    DEFAULT_LR,
    EOS_ID,
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
    evaluate,
    make_mock_corpus,
    recall_at_k,
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


def test_eval_suite_contract() -> None:
    """Counts, references, question ids, determinism, and held-out split."""
    suite = build_eval_suite(
        DEFAULT_EVAL_SEED,
        DEFAULT_EVAL_SAMPLES,
        VOCAB_SIZE,
        CAPTION_LEN,
    )
    repeat = build_eval_suite(
        DEFAULT_EVAL_SEED,
        DEFAULT_EVAL_SAMPLES,
        VOCAB_SIZE,
        CAPTION_LEN,
    )
    assert len(suite.pairs) == DEFAULT_EVAL_SAMPLES
    assert len(suite.vqa) == DEFAULT_EVAL_SAMPLES
    assert len(suite.caps) == DEFAULT_EVAL_SAMPLES
    assert suite.question_offset == VOCAB_SIZE - suite.content_len
    np.testing.assert_array_equal(
        suite.pairs[0].image, repeat.pairs[0].image
    )
    np.testing.assert_array_equal(
        suite.pairs[0].caption_ids, repeat.pairs[0].caption_ids
    )
    for sample in suite.vqa:
        slot = int(sample.question_ids[-1]) - suite.question_offset
        assert 0 <= slot < suite.content_len
        assert CONTENT_OFFSET <= sample.answer_id < (
            CONTENT_OFFSET + CONTENT_VOCAB
        )
    for caption_sample in suite.caps:
        assert len(caption_sample.reference_ids) == 3
        for reference in caption_sample.reference_ids:
            assert reference[0] == BOS_ID
            assert reference[1 + suite.content_len] == EOS_ID

    corpus = make_mock_corpus(seed=62, n_pairs=200)
    trained = {
        tuple(int(token) for token in row[1 : 1 + CONTENT_LEN])
        for row in corpus.captions
    }
    held_out = {
        tuple(
            int(token)
            for token in pair.caption_ids[1 : 1 + CONTENT_LEN]
        )
        for pair in suite.pairs
    }
    assert not trained & held_out


def test_recall_identity_and_worst_diagonal() -> None:
    """R@K reaches 1.0 for identity and 0.0 when the diagonal is worst."""
    size = 6
    identity = np.eye(size)
    assert recall_at_k(identity, 1) == 1.0
    assert recall_at_k(identity, 3, direction="image_to_text") == 1.0
    worst = np.zeros((size, size))
    np.fill_diagonal(worst, -1.0)
    assert recall_at_k(worst, 1) == 0.0
    assert recall_at_k(worst, 2) == 0.0
    assert recall_at_k(worst, size) == 1.0
    try:
        recall_at_k(identity, size + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("k above the matrix size must be rejected")


def test_bleu_contract() -> None:
    """Exact match, disjoint vocabulary, smoothing, and brevity."""
    assert bleu4([1, 2, 3, 4], [[1, 2, 3, 4]]) == 1.0
    assert bleu4([1, 2, 3, 4], [[5, 6, 7, 8]]) == 0.0
    assert bleu4([1, 2, 3, 4], [[1, 2, 9, 9]], smoothing=True) > 0.0
    assert bleu4([1, 2, 3, 4], [[1, 2, 9, 9]], smoothing=False) == 0.0
    assert abs(
        bleu4([1, 2, 3, 4, 5], [[1, 2, 3, 4, 5, 6, 7, 8]])
        - exp(1.0 - 8.0 / 5.0)
    ) < 1e-6


def test_vqa_exact_match_contract() -> None:
    """Exact match is the mean of one-bit integer equality."""
    assert vqa_exact_match([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5
    assert vqa_exact_match([7], [7]) == 1.0
    assert vqa_exact_match([7], [8]) == 0.0


def test_evaluate_returns_all_surfaces() -> None:
    """The suite evaluator exposes bounded, deterministic retrieval/VQA/BLEU."""
    model = build_default_model(seed=62)
    suite = build_eval_suite(
        DEFAULT_EVAL_SEED, 12, VOCAB_SIZE, CAPTION_LEN
    )
    first = evaluate(model, suite)
    second = evaluate(model, suite)
    assert first == second
    assert first["n_samples"] == 12.0
    assert {
        "retrieval_r1",
        "retrieval_r5",
        "retrieval_r10",
        "retrieval_r1_i2t",
        "retrieval_r5_i2t",
        "retrieval_r10_i2t",
        "vqa_em",
        "bleu4",
        "n_samples",
    } == set(first)
    for key, value in first.items():
        if key != "n_samples":
            assert 0.0 <= value <= 1.0


def test_head_gradients_are_live_and_finite() -> None:
    """Both readout heads receive finite, non-zero gradients."""
    model = _small_model()
    corpus = make_mock_corpus(seed=8, n_pairs=16)
    model.loss_and_backward(
        corpus.images[:8], corpus.captions[:8]
    )
    for parameter in model.parameters():
        assert np.all(np.isfinite(parameter.grad)), parameter.name
    assert np.abs(model.caption_head.fc1.weight.grad).sum() > 0.0
    assert (
        np.abs(model.vqa_head.question_embedding.table.grad).sum() > 0.0
    )


def test_head_only_training_reduces_losses() -> None:
    """Frozen base features still let the heads fit the synthetic codebook."""
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
    for _ in range(25):
        latest = model.loss_and_backward(
            corpus.images[:16],
            corpus.captions[:16],
            lm_weight=0.0,
            caption_weight=1.0,
            vqa_weight=1.0,
        )
        optimizer.step()
    assert latest.caption < first.caption * 0.6
    assert latest.vqa < first.vqa * 0.6


def test_short_training_reduces_total_loss() -> None:
    """One short deterministic training run reduces the combined loss."""
    model = _small_model()
    corpus = make_mock_corpus(seed=10, n_pairs=32)
    report = train(
        model,
        corpus,
        steps=6,
        batch_size=8,
        lr=DEFAULT_LR,
        head_lr=3e-2,
        log_every=100,
        print_fn=_noop,
    )
    assert len(report.records) == 6
    assert report.final.total < report.initial.total


_GATE_TESTS: tuple[Callable[[], None], ...] = (
    test_eval_suite_contract,
    test_recall_identity_and_worst_diagonal,
    test_bleu_contract,
    test_vqa_exact_match_contract,
    test_evaluate_returns_all_surfaces,
    test_head_gradients_are_live_and_finite,
    test_head_only_training_reduces_losses,
    test_short_training_reduces_total_loss,
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Run the module-level contract functions under unittest discovery."""
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for test in _GATE_TESTS:
        suite.addTest(unittest.FunctionTestCase(test))
    return suite
