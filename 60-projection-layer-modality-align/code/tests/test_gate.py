"""Top-level contract tests counted by the portfolio verification gate."""

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
    cosine_alignment_loss,
    cosine_alignment_loss_and_grad,
    make_pair,
    train,
)


def _pairs(
    count: int,
) -> tuple[MockTextEmbedding, MockVisionEncoder, list[tuple[np.ndarray, np.ndarray]]]:
    table = MockTextEmbedding(128, TEXT_HIDDEN, seed=0)
    encoder = MockVisionEncoder(
        TEXT_HIDDEN, VISION_HIDDEN, NUM_TOKENS, seed=1
    )
    pairs = [
        make_pair(seed, 128, text_table=table, vision_encoder=encoder)
        for seed in range(count)
    ]
    return table, encoder, pairs


def test_projector_output_and_parameter_budget() -> None:
    """The 768-1024-512 projector has the expected shape and 1.3M parameters."""
    projector = MLPProjector(768, 1024, 512, seed=0)

    output = projector(np.zeros((4, 768), dtype=np.float32))

    assert output.shape == (4, 512)
    assert output.dtype == np.float32
    assert projector.num_parameters() == 1_312_256


def test_text_table_is_frozen_and_seeded() -> None:
    """The mock table is reproducible and exposes no trainable parameters."""
    first = MockTextEmbedding(64, 16, seed=3)
    second = MockTextEmbedding(64, 16, seed=3)
    third = MockTextEmbedding(64, 16, seed=4)

    assert np.array_equal(first.table, second.table)
    assert not np.array_equal(first.table, third.table)
    assert first.requires_grad is False
    assert all(not parameter.requires_grad for parameter in first.parameters())


def test_caption_embedding_is_mean_pooled() -> None:
    """A caption embedding is the arithmetic mean of its token rows."""
    table = MockTextEmbedding(32, 8, seed=1)
    ids = np.array([1, 5, 7], dtype=np.int64)

    pooled = table.caption_embedding(ids)

    expected = table.table[ids].mean(axis=0)
    np.testing.assert_allclose(pooled, expected, rtol=0.0, atol=1e-7)
    assert pooled.shape == (8,)


def test_cosine_loss_extremes() -> None:
    """Cosine loss is zero, one, and two at the expected angles."""
    paired = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    image = np.array([[1.0, 0.0]], dtype=np.float32)
    text = np.array([[0.0, 1.0]], dtype=np.float32)

    assert abs(cosine_alignment_loss(paired, paired)) < 1e-6
    assert abs(cosine_alignment_loss(paired, -paired) - 2.0) < 1e-6
    assert abs(cosine_alignment_loss(image, text) - 1.0) < 1e-6


def test_projector_gradient_reaches_both_layers() -> None:
    """One backward pass fills every projector gradient and the input gradient."""
    projector = MLPProjector(8, 12, 5, seed=2)
    rng = np.random.default_rng(10)
    features = rng.normal(size=(3, 8)).astype(np.float32)
    target = rng.normal(size=(3, 5)).astype(np.float32)

    projected = projector(features)
    _, gradient = cosine_alignment_loss_and_grad(projected, target)
    projector.zero_grad()
    input_gradient = projector.backward(gradient)

    assert input_gradient.shape == features.shape
    assert float(np.linalg.norm(input_gradient)) > 0.0
    for parameter in projector.parameters():
        assert parameter.grad.shape == parameter.data.shape
        assert float(np.linalg.norm(parameter.grad)) > 0.0


def test_make_pair_shapes_and_determinism() -> None:
    """The fixture emits 197x768 image tokens and deterministic caption ids."""
    table, encoder, pairs = _pairs(2)
    tokens, caption_ids = pairs[0]
    repeated_tokens, repeated_ids = make_pair(
        0, 128, text_table=table, vision_encoder=encoder
    )

    assert tokens.shape == (NUM_TOKENS, VISION_HIDDEN)
    assert caption_ids.shape == (CAPTION_LEN,)
    assert np.array_equal(tokens, repeated_tokens)
    assert np.array_equal(caption_ids, repeated_ids)
    assert np.all(caption_ids >= 0)
    assert np.all(caption_ids < 128)


def test_training_reduces_loss_by_step_200() -> None:
    """The projection-only loop lowers the full-dataset loss and raises cosine."""
    table, _, pairs = _pairs(8)
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

    assert reports[0].step == 0
    assert reports[-1].step == 200
    assert reports[-1].loss < reports[0].loss - 0.05
    assert reports[-1].mean_cosine > reports[0].mean_cosine + 0.05


def test_frozen_modules_unchanged_after_training() -> None:
    """Training moves the projector and no frozen table or token fixture."""
    table, encoder, pairs = _pairs(4)
    projector = MLPProjector(VISION_HIDDEN, 32, TEXT_HIDDEN, seed=2)
    table_before = table.table.copy()
    encoder_before = encoder.projection.data.copy()
    tokens_before = [tokens.copy() for tokens, _ in pairs]
    projector_before = [
        parameter.data.copy() for parameter in projector.parameters()
    ]

    train(
        projector,
        pairs,
        table,
        steps=20,
        batch_size=4,
        lr=0.1,
        report_every=10,
    )

    assert np.array_equal(table.table, table_before)
    assert np.array_equal(encoder.projection.data, encoder_before)
    assert all(
        np.array_equal(tokens, before)
        for before, (tokens, _) in zip(tokens_before, pairs, strict=True)
    )
    assert all(
        not parameter.requires_grad
        for parameter in table.parameters() + encoder.parameters()
    )
    assert any(
        not np.array_equal(before, parameter.data)
        for before, parameter in zip(
            projector_before, projector.parameters(), strict=True
        )
    )


def test_finite_difference_checks_projector_and_cosine() -> None:
    """Analytic MLP and cosine gradients match central differences."""
    projector = MLPProjector(6, 5, 4, seed=3)
    rng = np.random.default_rng(20)
    features = rng.normal(size=(3, 6)).astype(np.float32)
    upstream = rng.normal(size=(3, 4)).astype(np.float32)
    projector.zero_grad()
    projector(features)
    projector.backward(upstream)

    parameter = projector.fc2
    index = (1, 3)
    old = parameter.data[index]
    epsilon = 1e-2
    parameter.data[index] = old + epsilon
    plus = float(np.sum(projector(features) * upstream))
    parameter.data[index] = old - epsilon
    minus = float(np.sum(projector(features) * upstream))
    parameter.data[index] = old
    numeric = (plus - minus) / (2.0 * epsilon)
    assert abs(numeric - float(parameter.grad[index])) < 1e-3

    image = np.array(
        [[0.3, -0.2, 1.1], [0.4, 0.7, -0.5]], dtype=np.float32
    )
    text = np.array(
        [[0.8, 0.1, 0.2], [-0.3, 0.6, 0.9]], dtype=np.float32
    )
    _, gradient = cosine_alignment_loss_and_grad(image, text)
    old_value = image[0, 0]
    image[0, 0] = old_value + epsilon
    plus_loss = cosine_alignment_loss(image, text)
    image[0, 0] = old_value - epsilon
    minus_loss = cosine_alignment_loss(image, text)
    image[0, 0] = old_value
    numeric_loss_gradient = (plus_loss - minus_loss) / (2.0 * epsilon)
    assert abs(numeric_loss_gradient - float(gradient[0, 0])) < 1e-3


def test_cls_pool_returns_first_token() -> None:
    """CLS pooling returns exactly row zero for every batch item."""
    tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    pooled = cls_pool(tokens)

    np.testing.assert_array_equal(pooled, tokens[:, 0, :])
    assert pooled.shape == (2, 4)


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
