"""Top-level contract tests counted by the portfolio verification gate."""

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
    DEFAULT_LR,
    EOS_ID,
    PAD_ID,
    Adam,
    CrossAttentionDecoder,
    ImageProjector,
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
    unpatchify,
)


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


def _central_difference(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
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


def test_mock_corpus_is_deterministic() -> None:
    """Equal seeds reproduce identical pairs with fixed sentinels."""
    first = make_mock_corpus(5, 10)
    second = make_mock_corpus(5, 10)
    other = make_mock_corpus(6, 10)

    assert np.array_equal(first.images, second.images)
    assert np.array_equal(first.captions, second.captions)
    assert not np.array_equal(first.images, other.images)
    assert first.images.shape == (10, 1, 8, 8)
    assert first.captions.shape == (10, CAPTION_LEN)
    assert np.all(first.captions[:, 0] == BOS_ID)
    assert np.all(first.captions[:, 7] == EOS_ID)
    assert np.all(first.captions[:, 8:] == PAD_ID)
    content = first.captions[:, 1 : 1 + CONTENT_LEN]
    assert np.all(content >= CONTENT_OFFSET)
    assert np.all(content < CONTENT_OFFSET + 16)


def test_infonce_is_symmetric() -> None:
    """Swapping the two modalities leaves the symmetric loss unchanged."""
    rng = np.random.default_rng(0)
    image = rng.normal(size=(5, 4))
    text = rng.normal(size=(5, 4))

    assert abs(
        info_nce_loss(image, text, 0.2)
        - info_nce_loss(text, image, 0.2)
    ) < 1e-10


def test_infonce_perfect_diagonal_is_zero() -> None:
    """One-hot matches at a low temperature drive the loss to zero."""
    identity = np.eye(4, dtype=np.float64)
    loss = info_nce_loss(identity, identity, 0.05)

    assert abs(loss) < 1e-6


def test_infonce_gradient_matches_finite_difference() -> None:
    """The image gradient and temperature gradient are exact."""
    rng = np.random.default_rng(2)
    image = rng.normal(size=(4, 3))
    text = rng.normal(size=(4, 3))
    temperature = 0.2
    _, d_image, _, d_temperature = info_nce_loss_and_grad(
        image, text, temperature
    )
    numeric_image = _central_difference(
        lambda values: info_nce_loss(values, text, temperature),
        image.copy(),
        eps=1e-6,
    )
    numeric_temperature = _central_difference(
        lambda values: info_nce_loss(
            image, text, float(values)
        ),
        np.array(temperature, dtype=np.float64),
        eps=1e-6,
    )

    assert np.allclose(d_image, numeric_image, rtol=1e-6, atol=1e-8)
    assert abs(d_temperature - float(numeric_temperature)) < 1e-6


def test_lm_loss_masks_padding() -> None:
    """Padding targets are excluded from the mean loss."""
    logits = np.array(
        [
            [
                [0.2, -0.1, 0.5, 0.0, 0.3],
                [0.1, 0.4, -0.2, 0.6, 0.0],
                [0.7, 0.2, 0.1, -0.3, 0.4],
            ]
        ],
        dtype=np.float64,
    )
    targets = np.array([[1, 2, PAD_ID]], dtype=np.int64)
    loss = lm_loss(logits, targets, PAD_ID)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(
        np.exp(shifted).sum(axis=-1, keepdims=True)
    )
    expected = -float(
        np.mean(
            [
                log_probs[0, 0, 1],
                log_probs[0, 1, 2],
            ]
        )
    )
    assert abs(loss - expected) < 1e-10


def test_lm_padding_gradient_is_zero() -> None:
    """No gradient reaches padded positions."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2, 3, 5))
    targets = np.array(
        [[1, 2, PAD_ID], [3, PAD_ID, PAD_ID]], dtype=np.int64
    )
    loss, dlogits = lm_loss_and_grad(logits, targets, PAD_ID)

    assert np.all(dlogits[0, 2] == 0.0)
    assert np.all(dlogits[1, 1:] == 0.0)
    assert loss > 0.0
    numeric = _central_difference(
        lambda values: lm_loss(values, targets, PAD_ID),
        logits.copy(),
        eps=1e-6,
    )
    assert np.allclose(dlogits, numeric, rtol=1e-6, atol=1e-8)


def test_model_forward_returns_both_losses() -> None:
    """The forward pass produces image/text embeddings and caption logits."""
    model = _small_model()
    corpus = make_mock_corpus(0, 6)
    output = model.forward(corpus.images, corpus.captions)
    contrastive, language_model, total = model.losses(output)

    assert output.image_emb.shape == (6, 16)
    assert output.text_emb.shape == (6, 16)
    assert output.logits.shape == (6, 9, 32)
    assert output.targets.shape == (6, 9)
    assert math.isfinite(contrastive)
    assert math.isfinite(language_model)
    assert contrastive >= 0.0
    assert language_model >= 0.0
    assert abs(total - (contrastive + language_model)) < 1e-10


def test_all_parameter_gradients_are_finite() -> None:
    """Backward touches every parameter with finite values."""
    model = _small_model()
    corpus = make_mock_corpus(0, 6)
    model.loss_and_backward(corpus.images, corpus.captions)

    nonzero = 0
    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.data.shape
        assert np.all(np.isfinite(parameter.grad))
        if np.any(parameter.grad != 0.0):
            nonzero += 1
    assert nonzero > 0
    assert np.all(np.isfinite(model.log_temperature.grad))


def test_contrastive_only_leaves_decoder_unchanged() -> None:
    """lm_weight=0 blocks every decoder update while contrastive still trains."""
    model = _small_model()
    corpus = make_mock_corpus(0, 6)
    optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
    before = [
        parameter.data.copy()
        for parameter in model.decoder.parameters()
    ]
    model.loss_and_backward(
        corpus.images, corpus.captions, lm_weight=0.0
    )
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()

    for original, parameter in zip(
        before, model.decoder.parameters()
    ):
        assert np.array_equal(original, parameter.data)


def test_five_steps_reduce_combined_loss() -> None:
    """Five optimizer steps lower the combined loss on a fixed batch."""
    model = _small_model()
    corpus = make_mock_corpus(0, 8)
    optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
    before = model.losses(
        model.forward(corpus.images, corpus.captions)
    )[2]
    for _ in range(5):
        model.loss_and_backward(corpus.images, corpus.captions)
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
    after = model.losses(
        model.forward(corpus.images, corpus.captions)
    )[2]

    assert after < before


def test_temperature_stays_in_range_and_moves() -> None:
    """The learned log-temperature changes and projects into [0.01, 1.0]."""
    model = _small_model()
    corpus = make_mock_corpus(0, 6)
    optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
    before = float(model.log_temperature.data.item())
    model.loss_and_backward(corpus.images, corpus.captions)
    assert float(model.log_temperature.grad.item()) != 0.0
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    tau, _ = model.temperature()

    assert float(model.log_temperature.data.item()) != before
    assert 0.01 <= tau <= 1.0


def test_causal_attention_blocks_future_tokens() -> None:
    """Changing the final token leaves earlier decoder logits untouched."""
    model = _small_model()
    rng = np.random.default_rng(4)
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

    assert np.array_equal(first[:, :-1], second[:, :-1])
    assert float(np.max(np.abs(first[:, -1] - second[:, -1]))) > 1e-6


def test_patchify_roundtrip() -> None:
    """Patch extraction is row-major and exactly invertible."""
    rng = np.random.default_rng(5)
    images = rng.normal(size=(2, 1, 8, 8)).astype(np.float32)
    patches = patchify(images, 2)

    assert patches.shape == (2, 16, 4)
    assert np.array_equal(
        patches[0, 0], images[0, :, 0:2, 0:2].reshape(-1)
    )
    assert np.array_equal(unpatchify(patches, 1, 8, 8, 2), images)


def test_retrieval_metric() -> None:
    """Perfect matches score 1 and a derangement scores 0."""
    identity = np.eye(3, dtype=np.float64)
    assert similarity_accuracy(identity, identity) == (1.0, 1.0)
    deranged = identity[[1, 2, 0]]
    assert similarity_accuracy(identity, deranged) == (0.0, 0.0)


def test_corpus_renders_token_patches() -> None:
    """The first six patches carry the six caption tokens in order."""
    corpus = make_mock_corpus(9, 3)
    image = corpus.images[0, 0]
    tokens = corpus.captions[0, 1 : 1 + CONTENT_LEN]

    for position, token in enumerate(tokens):
        index = int(token) - CONTENT_OFFSET
        angle = 2.0 * math.pi * index / 16
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
        assert np.allclose(
            image[row : row + 2, column : column + 2],
            expected,
            rtol=0.0,
            atol=0.1,
        )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Register the top-level functions with unittest discovery."""
    del loader, pattern
    for name, candidate in sorted(globals().items()):
        if name.startswith("test_") and callable(candidate):
            tests.addTest(
                unittest.FunctionTestCase(candidate, description=name)
            )
    return tests


if __name__ == "__main__":
    unittest.main()
