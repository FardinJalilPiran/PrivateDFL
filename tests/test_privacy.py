"""Tests for the noise accountant (Theorems 2-5 and Section 4.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from privatedfl.privacy import (
    NoiseAccountant,
    blackbox_cumulative_variance,
    cumulative_variance,
    incremental_variance,
    required_variance,
    step_index,
)

BASE = dict(dimensions=2000, epsilon=0.5, delta0=1e-3, samples_per_client=73)
K = 100


# ------------------------------------------------------------------ step index


@pytest.mark.parametrize(
    ("r", "k", "expected"),
    [(1, 1, 1), (1, 100, 100), (2, 1, 101), (2, 50, 150), (30, 100, 3000)],
)
def test_step_index_maps_round_and_client(r, k, expected):
    assert step_index(r, k, K) == expected


def test_step_index_rejects_out_of_range():
    with pytest.raises(ValueError):
        step_index(0, 1, K)
    with pytest.raises(ValueError):
        step_index(1, 0, K)
    with pytest.raises(ValueError):
        step_index(1, K + 1, K)


# ------------------------------------------------------------------- theorems


def test_theorem_2_first_client_first_round():
    """Gamma(1,1) = (2D/eps^2) ln(1.25 N / delta0)."""
    expected = (2 * 2000 / 0.25) * math.log(1.25 * 73 / 1e-3)
    assert incremental_variance(**BASE, step=1) == pytest.approx(expected)


@pytest.mark.parametrize("k", [2, 5, 50, 100])
def test_theorem_3_remaining_clients_first_round(k):
    """Gamma(1,k) = (2D/eps^2) ln(k / (k-1))."""
    expected = (2 * 2000 / 0.25) * math.log(k / (k - 1))
    assert incremental_variance(**BASE, step=step_index(1, k, K)) == pytest.approx(expected)


@pytest.mark.parametrize("r", [2, 5, 30])
def test_theorem_4_first_client_later_rounds(r):
    """Gamma(r,1) = (2D/eps^2) ln((K(r-1)+1) / (K(r-1)))."""
    expected = (2 * 2000 / 0.25) * math.log((K * (r - 1) + 1) / (K * (r - 1)))
    assert incremental_variance(**BASE, step=step_index(r, 1, K)) == pytest.approx(expected)


@pytest.mark.parametrize(("r", "k"), [(2, 7), (5, 50), (30, 100)])
def test_theorem_5_general_case(r, k):
    """Gamma(r,k) = (2D/eps^2) ln((K(r-1)+k) / (K(r-1)+k-1))."""
    t = K * (r - 1) + k
    expected = (2 * 2000 / 0.25) * math.log(t / (t - 1))
    assert incremental_variance(**BASE, step=t) == pytest.approx(expected)


def test_theorems_3_and_4_are_special_cases_of_theorem_5():
    """The paper notes Theorem 5 subsumes 3 (r=1) and 4 (k=1)."""
    for r, k in [(1, 9), (4, 1), (12, 63)]:
        t = K * (r - 1) + k
        general = (2 * 2000 / 0.25) * math.log(t / (t - 1))
        assert incremental_variance(**BASE, step=t) == pytest.approx(general)


# ----------------------------------------------------------------- accounting


@pytest.mark.parametrize("steps", [1, 2, 10, 137, 3000])
def test_increments_telescope_to_the_requirement(steps):
    """Eq. 28: independent increments sum exactly to the required variance."""
    total = sum(incremental_variance(**BASE, step=t) for t in range(1, steps + 1))
    assert total == pytest.approx(required_variance(**BASE, step=steps), rel=1e-9)


def test_cumulative_equals_required():
    for t in (1, 5, 500):
        assert cumulative_variance(**BASE, step=t) == pytest.approx(
            required_variance(**BASE, step=t)
        )


def test_requirement_grows_only_logarithmically():
    """Doubling the work adds a constant, not a multiple."""
    a = required_variance(**BASE, step=1000)
    b = required_variance(**BASE, step=2000)
    c = required_variance(**BASE, step=4000)
    assert (b - a) == pytest.approx(c - b, rel=1e-9)


def test_increments_shrink_as_the_run_progresses():
    values = [incremental_variance(**BASE, step=t) for t in range(2, 50)]
    assert values == sorted(values, reverse=True)


def test_blackbox_exceeds_tracked_and_diverges():
    """Section 4.2.3: ln(t!) vs ln(t)."""
    ratios = []
    for t in (10, 100, 1000, 3000):
        tracked = required_variance(**BASE, step=t)
        blackbox = blackbox_cumulative_variance(**BASE, step=t)
        assert blackbox > tracked
        ratios.append(blackbox / tracked)
    assert ratios == sorted(ratios)  # the gap widens


def test_blackbox_matches_the_closed_form():
    """Xi(t) = (2D/eps^2)[ t ln(1.25N/delta0) + ln(t!) ]."""
    t = 20
    naive = sum((2 * 2000 / 0.25) * math.log(1.25 * 73 * j / 1e-3) for j in range(1, t + 1))
    assert blackbox_cumulative_variance(**BASE, step=t) == pytest.approx(naive, rel=1e-9)


def test_blackbox_stays_finite_for_large_steps():
    """lgamma keeps ln(t!) computable where t! would overflow."""
    assert math.isfinite(blackbox_cumulative_variance(**BASE, step=100_000))


# ------------------------------------------------------------------- scaling


def test_variance_scales_linearly_with_dimensions():
    a = required_variance(dimensions=1000, epsilon=0.5, delta0=1e-3, samples_per_client=50, step=5)
    b = required_variance(dimensions=2000, epsilon=0.5, delta0=1e-3, samples_per_client=50, step=5)
    assert b == pytest.approx(2 * a)


def test_variance_scales_inversely_with_epsilon_squared():
    tight = required_variance(
        dimensions=1000, epsilon=0.25, delta0=1e-3, samples_per_client=50, step=5
    )
    loose = required_variance(
        dimensions=1000, epsilon=0.5, delta0=1e-3, samples_per_client=50, step=5
    )
    assert tight == pytest.approx(4 * loose)


def test_delta0_has_only_a_logarithmic_effect():
    """The paper reports delta0 barely moves accuracy; the maths says why."""
    a = required_variance(dimensions=2000, epsilon=0.5, delta0=1e-3, samples_per_client=73, step=10)
    b = required_variance(
        dimensions=2000, epsilon=0.5, delta0=1e-18, samples_per_client=73, step=10
    )
    assert b > a
    assert b / a < 4  # 15 orders of magnitude in delta0, under 4x in variance


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dimensions=0, epsilon=0.5, delta0=1e-3, samples_per_client=10),
        dict(dimensions=10, epsilon=0.0, delta0=1e-3, samples_per_client=10),
        dict(dimensions=10, epsilon=0.5, delta0=0.0, samples_per_client=10),
        dict(dimensions=10, epsilon=0.5, delta0=1.0, samples_per_client=10),
        dict(dimensions=10, epsilon=0.5, delta0=1e-3, samples_per_client=0),
    ],
)
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        required_variance(**kwargs, step=1)


# ---------------------------------------------------------------- perturbation


def test_perturb_draws_the_prescribed_variance():
    accountant = NoiseAccountant(**BASE, n_clients=K)
    model = torch.zeros((6, BASE["dimensions"]), dtype=torch.float64)
    noisy = accountant.perturb(model, round_index=3, client_index=40)
    expected = incremental_variance(**BASE, step=step_index(3, 40, K))
    assert noisy.var().item() == pytest.approx(expected, rel=0.05)


def test_disabled_accountant_is_a_no_op():
    accountant = NoiseAccountant(**BASE, n_clients=K, enabled=False)
    model = torch.randn((6, 128))
    assert torch.equal(accountant.perturb(model, 1, 1), model)
    assert accountant.report(1, 1)["added"] == 0.0


def test_perturb_is_reproducible_with_a_seeded_generator():
    accountant = NoiseAccountant(**BASE, n_clients=K)
    model = torch.zeros((6, 256))
    a = accountant.perturb(model, 2, 3, generator=torch.Generator().manual_seed(0))
    b = accountant.perturb(model, 2, 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_report_exposes_the_full_accounting():
    accountant = NoiseAccountant(**BASE, n_clients=K)
    report = accountant.report(2, 5)
    assert report["step"] == 105
    assert report["added"] < report["required"] < report["blackbox"]


def test_noise_is_calibrated_against_the_largest_shard():
    """delta = delta_0 / N, so uneven shards must be covered by the biggest one.

    Calibrating on the smallest shard would under-noise the clients holding the
    most data, silently weakening the guarantee they were promised.
    """
    from privatedfl.data import ClientData, FederatedDataset

    rng = np.random.default_rng(0)
    clients = [
        ClientData(client_id=f"c{i}", x=rng.random((n, 4)).astype(np.float32), y=np.zeros(n, int))
        for i, n in enumerate((10, 40, 25))
    ]
    dataset = FederatedDataset(
        name="uneven",
        clients=clients,
        test_x=rng.random((5, 4)).astype(np.float32),
        test_y=np.zeros(5, int),
        n_features=4,
        n_classes=2,
    )
    assert dataset.min_client_samples() == 10
    assert dataset.max_client_samples() == 40

    small = required_variance(2000, 1.0, 1e-3, 10, 1)
    large = required_variance(2000, 1.0, 1e-3, 40, 1)
    assert large > small, "a larger N must demand strictly more noise"
