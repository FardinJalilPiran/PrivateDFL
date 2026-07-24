"""Tests for the HD model.

The encoder is pinned deliberately. The sensitivity bound sqrt(D) used in
Proofs 1-4 relies on cosine outputs lying in [-1, 1]; binarising or rescaling
would silently invalidate the noise calibration, so these tests lock the
encoder to exactly cos(X @ basis).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from privatedfl.hdc import Encoder, HDModel

# ------------------------------------------------------------------- encoder


def test_encoder_is_exactly_cosine_of_the_projection():
    """Locks Eq. (1): h_d = cos(F . B_d), with no binarisation."""
    encoder = Encoder(n_features=8, dimensions=64, seed=0)
    x = torch.rand(5, 8)
    assert torch.allclose(encoder(x), torch.cos(x @ encoder.basis), atol=1e-6)


def test_encoder_output_is_continuous_not_bipolar():
    """A regression guard: hypervectors must not be sign()-ed."""
    encoder = Encoder(n_features=16, dimensions=512, seed=0)
    encoded = encoder(torch.rand(32, 16))
    assert encoded.abs().max() <= 1.0 + 1e-6
    assert len(encoded.unique()) > 100  # would be exactly 2 if binarised


def test_encoder_respects_the_sensitivity_bound():
    """||H|| <= sqrt(D), the bound the DP proofs assume."""
    dimensions = 256
    encoder = Encoder(n_features=12, dimensions=dimensions, seed=0)
    encoded = encoder(torch.rand(64, 12))
    assert encoded.norm(dim=1).max().item() <= np.sqrt(dimensions) + 1e-4


def test_basis_is_standard_normal():
    encoder = Encoder(n_features=64, dimensions=4096, seed=0)
    assert encoder.basis.mean().item() == pytest.approx(0.0, abs=0.02)
    assert encoder.basis.std().item() == pytest.approx(1.0, abs=0.02)


def test_encoder_shape_and_single_sample():
    encoder = Encoder(n_features=10, dimensions=128, seed=0)
    assert encoder(torch.rand(7, 10)).shape == (7, 128)
    assert encoder(torch.rand(10)).shape == (1, 128)


def test_encoder_is_deterministic_for_a_fixed_seed():
    x = torch.rand(4, 6)
    a = Encoder(n_features=6, dimensions=128, seed=7)(x)
    b = Encoder(n_features=6, dimensions=128, seed=7)(x)
    assert torch.equal(a, b)


def test_chunked_encoding_matches_a_single_pass():
    encoder = Encoder(n_features=6, dimensions=128, seed=1)
    x = torch.rand(50, 6)
    assert torch.allclose(encoder.encode(x, batch_size=7), encoder(x), atol=1e-6)


def test_encoding_preserves_locality():
    encoder = Encoder(n_features=32, dimensions=4096, seed=3)
    anchor = torch.full((1, 32), 0.5)
    encoded = encoder(torch.cat([anchor, anchor + 0.01, torch.full((1, 32), 0.99)]))
    assert torch.dot(encoded[0], encoded[1]) > torch.dot(encoded[0], encoded[2])


# --------------------------------------------------------------------- model


def test_from_samples_bundles_per_class():
    encoded = torch.ones((4, 8))
    labels = torch.tensor([0, 0, 1, 2])
    model = HDModel.from_samples(encoded, labels, n_classes=3)
    assert model.class_hypervectors[0].sum().item() == pytest.approx(16.0)
    assert model.class_hypervectors[1].sum().item() == pytest.approx(8.0)


def test_model_learns_a_separable_problem():
    torch.manual_seed(0)
    encoder = Encoder(n_features=10, dimensions=2048, seed=0)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0.0, 0.3, (60, 10)), rng.uniform(0.7, 1.0, (60, 10))])
    y = np.array([0] * 60 + [1] * 60)
    encoded = encoder(torch.from_numpy(x.astype(np.float32)))
    labels = torch.from_numpy(y).long()

    model = HDModel.from_samples(encoded, labels, n_classes=2)
    assert model.accuracy(encoded, labels) > 0.9


def test_retraining_does_not_hurt_a_learnable_problem():
    torch.manual_seed(0)
    encoder = Encoder(n_features=12, dimensions=2048, seed=1)
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(0.0, 0.45, (80, 12)), rng.uniform(0.55, 1.0, (80, 12))])
    y = np.array([0] * 80 + [1] * 80)
    encoded = encoder(torch.from_numpy(x.astype(np.float32)))
    labels = torch.from_numpy(y).long()

    model = HDModel.from_samples(encoded, labels, n_classes=2)
    before = model.accuracy(encoded, labels)
    model.retrain(encoded, labels, n_epochs=3, generator=torch.Generator().manual_seed(0))
    assert model.accuracy(encoded, labels) >= before - 1e-9


def test_retrain_defaults_to_a_single_pass():
    """The released implementation makes one shuffled pass per round."""
    import inspect

    assert inspect.signature(HDModel.retrain).parameters["n_epochs"].default == 1


def test_predict_returns_one_label_per_sample():
    model = HDModel.zeros(4, 64)
    model.class_hypervectors[2] += 1.0
    predictions = model.predict(torch.ones((7, 64)))
    assert predictions.shape == (7,)
    assert torch.all(predictions == 2)


def test_cosine_normalises_prototype_magnitude_but_dot_does_not():
    """A large-norm prototype wins under dot but not under cosine."""
    model = HDModel.zeros(2, 4)
    model.class_hypervectors[0] = torch.tensor([1.0, 1.0, 1.0, 1.0]) * 10  # big norm
    model.class_hypervectors[1] = torch.tensor([1.0, 1.0, 1.0, 1.0]) * 1  # aligned, small
    query = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert model.predict(query, "dot").item() == 0
    assert model._score(query, "cosine").std().item() == pytest.approx(0.0, abs=1e-5)


def test_unknown_similarity_is_rejected():
    model = HDModel.zeros(2, 8)
    with pytest.raises(ValueError):
        model.predict(torch.ones((1, 8)), similarity="euclidean")


def test_clone_is_independent():
    model = HDModel.zeros(2, 8)
    copy = model.clone()
    copy.class_hypervectors += 1
    assert model.class_hypervectors.sum().item() == 0
