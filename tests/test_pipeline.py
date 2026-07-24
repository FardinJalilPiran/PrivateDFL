"""End-to-end tests for the ring loop, metrics, config and device handling."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from privatedfl import ExperimentConfig, evaluate, load_dataset, run_privatedfl
from privatedfl.utils import resolve_device

DATASET_ROOT = "Dataset"
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@pytest.fixture(scope="module")
def small_dataset():
    return load_dataset(
        "UCIHAR",
        n_clients=6,
        partition="non-IID",
        root=DATASET_ROOT,
        seed=0,
        samples_per_client=60,
    )


def small_config(**overrides):
    base = dict(
        dataset="UCIHAR",
        n_clients=6,
        rounds=3,
        dimensions=500,
        local_epochs=1,
        epsilon=0.5,
        delta0=1e-3,
        device="cpu",
        seed=0,
        data_root=DATASET_ROOT,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# -------------------------------------------------------------------- metrics


def test_perfect_predictions_give_zero_error_rates():
    y = np.array([0, 1, 2, 0, 1, 2])
    report = evaluate(y, y, n_classes=3)
    assert report.accuracy == 1.0
    assert report.macro_fpr == 0.0
    assert report.macro_fnr == 0.0


def test_metrics_match_a_hand_computed_confusion_matrix():
    report = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=2)
    assert report.accuracy == pytest.approx(0.75)
    assert report.per_class_fnr[0] == pytest.approx(0.5)
    assert report.per_class_fpr[1] == pytest.approx(0.5)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array([0, 1, 1]), n_classes=2)


# --------------------------------------------------------------------- config


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"dataset": "UCIHAR", "not_a_key": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rounds=0),
        dict(n_clients=0),
        dict(dimensions=0),
        dict(local_epochs=0),
        dict(epsilon=-1.0),
        dict(delta0=0.0),
        dict(delta0=1.0),
        dict(similarity="euclidean"),
    ],
)
def test_config_validates_ranges(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_config_roundtrips_through_a_dict():
    config = ExperimentConfig(dataset="UCIHAR", rounds=2)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_epsilon_is_not_validated_when_dp_is_off():
    ExperimentConfig(differential_privacy=False, epsilon=-1.0)  # must not raise


# ----------------------------------------------------------------- ring loop


def test_training_runs_and_beats_chance(small_dataset):
    history = run_privatedfl(small_dataset, small_config(), progress=False)
    assert len(history.rounds) == 3
    assert history.final_report.accuracy > 1.0 / small_dataset.n_classes


def test_step_counter_advances_by_one_ring_per_round(small_dataset):
    history = run_privatedfl(small_dataset, small_config(rounds=4), progress=False)
    assert [r.steps_completed for r in history.rounds] == [6, 12, 18, 24]


def test_blackbox_penalty_grows_across_rounds(small_dataset):
    history = run_privatedfl(small_dataset, small_config(rounds=4), progress=False)
    ratios = [r.noise_blackbox / r.noise_required for r in history.rounds]
    assert ratios == sorted(ratios)
    assert all(r > 1.0 for r in ratios)


def test_first_round_dominates_the_injected_noise(small_dataset):
    """Theorem 2 is a one-off cost; later increments telescope to almost nothing."""
    history = run_privatedfl(small_dataset, small_config(rounds=4), progress=False)
    first, *rest = [r.noise_added_this_round for r in history.rounds]
    assert all(first > 10 * later for later in rest)


def test_disabling_dp_injects_no_noise(small_dataset):
    history = run_privatedfl(
        small_dataset, small_config(differential_privacy=False), progress=False
    )
    assert all(r.noise_added_this_round == 0.0 for r in history.rounds)


def test_privacy_costs_accuracy(small_dataset):
    """A tight budget must not beat the non-private ceiling."""
    private = run_privatedfl(small_dataset, small_config(epsilon=0.05), progress=False)
    clear = run_privatedfl(small_dataset, small_config(differential_privacy=False), progress=False)
    assert private.final_report.accuracy <= clear.final_report.accuracy


def test_runs_are_reproducible_under_a_fixed_seed(small_dataset):
    a = run_privatedfl(small_dataset, small_config(seed=123), progress=False).accuracies
    b = run_privatedfl(small_dataset, small_config(seed=123), progress=False).accuracies
    assert a == b


def test_different_seeds_give_different_runs(small_dataset):
    a = run_privatedfl(small_dataset, small_config(seed=1), progress=False).accuracies
    b = run_privatedfl(small_dataset, small_config(seed=2), progress=False).accuracies
    assert a != b


def test_both_similarities_run(small_dataset):
    for similarity in ("dot", "cosine"):
        history = run_privatedfl(small_dataset, small_config(similarity=similarity), progress=False)
        assert np.isfinite(history.final_report.accuracy)


def test_eval_every_still_reports_each_round(small_dataset):
    history = run_privatedfl(small_dataset, small_config(rounds=4, eval_every=2), progress=False)
    assert len(history.rounds) == 4


def test_history_serialises_to_json_safe_types(small_dataset):
    history = run_privatedfl(small_dataset, small_config(), progress=False)
    json.dumps(history.to_dict())


def test_iid_partition_runs_end_to_end():
    dataset = load_dataset(
        "UCIHAR", n_clients=6, partition="IID", root=DATASET_ROOT, seed=0, samples_per_client=60
    )
    history = run_privatedfl(dataset, small_config(partition="IID"), progress=False)
    assert history.final_report.accuracy > 1.0 / dataset.n_classes


# --------------------------------------------------------------------- device


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_full_run_on_each_device(small_dataset, device):
    """A CPU generator must work against a CUDA model."""
    history = run_privatedfl(small_dataset, small_config(device=device), progress=False)
    assert np.isfinite(history.final_report.accuracy)


def test_resolve_device_falls_back_when_cuda_is_absent():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda").type == expected
    assert resolve_device("auto").type == expected
