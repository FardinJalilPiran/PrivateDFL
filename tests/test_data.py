"""Tests for the .choir_dat reader, normalisation and partitioning."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from privatedfl.data import _l2_normalise, available_datasets, load_dataset, read_choir_file
from privatedfl.partition import partition_clients

DATASET_ROOT = "Dataset"


def write_choir(path, x: np.ndarray, y: np.ndarray, n_classes: int) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack("ii", x.shape[1], n_classes))
        for row, label in zip(x, y):
            handle.write(row.astype(np.float32).tobytes())
            handle.write(struct.pack("i", int(label)))


# -------------------------------------------------------------------- reader


def test_reader_roundtrips(tmp_path):
    x = np.random.default_rng(0).random((25, 7)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 3, 25)
    path = tmp_path / "toy_train.choir_dat"
    write_choir(path, x, y, n_classes=3)

    n_features, n_classes, xr, yr = read_choir_file(path)
    assert (n_features, n_classes) == (7, 3)
    assert np.allclose(xr, x)
    assert np.array_equal(yr, y)
    assert yr.dtype == np.int64


def test_reader_rejects_a_truncated_file(tmp_path):
    x = np.random.default_rng(0).random((5, 4)).astype(np.float32)
    path = tmp_path / "t_train.choir_dat"
    write_choir(path, x, np.zeros(5), n_classes=2)
    data = path.read_bytes()
    path.write_bytes(data[:-3])  # lop off part of the last record
    with pytest.raises(ValueError, match="trailing bytes"):
        read_choir_file(path)


def test_reader_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        read_choir_file(tmp_path / "absent.choir_dat")


def test_reader_rejects_a_nonsense_header(tmp_path):
    path = tmp_path / "bad.choir_dat"
    path.write_bytes(struct.pack("ii", -1, 3))
    with pytest.raises(ValueError, match="implausible header"):
        read_choir_file(path)


# ------------------------------------------------------------- normalisation


def test_l2_normalisation_gives_unit_rows():
    x = np.random.default_rng(0).normal(size=(20, 9)).astype(np.float32)
    a, b = _l2_normalise(x, x * 3)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-5)


def test_l2_normalisation_survives_a_zero_row():
    x = np.zeros((2, 4), dtype=np.float32)
    a, _ = _l2_normalise(x, x)
    assert np.isfinite(a).all()


# ---------------------------------------------------------------- partition


def _toy(n=1200, features=5, classes=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, features)).astype(np.float32), np.tile(np.arange(classes), n // classes)


def test_iid_split_is_equal_and_disjoint():
    x, y = _toy()
    clients = partition_clients(x, y, n_clients=10, strategy="IID", seed=0)
    assert len({len(c) for c in clients}) == 1
    seen = np.concatenate([c.x for c in clients])
    assert len(np.unique(seen, axis=0)) == len(seen)


def test_iid_clients_see_every_class():
    x, y = _toy(n=6000)
    clients = partition_clients(x, y, n_clients=5, strategy="IID", n_classes=6, seed=0)
    for client in clients:
        assert (client.class_counts(6) > 0).sum() == 6


def test_non_iid_clients_are_restricted_to_two_classes():
    x, y = _toy(n=6000)
    clients = partition_clients(
        x, y, n_clients=20, strategy="non-IID", n_classes=6, seed=0, classes_per_client=2
    )
    for client in clients:
        assert (client.class_counts(6) > 0).sum() <= 2


def test_classes_per_client_is_honoured():
    x, y = _toy(n=6000)
    clients = partition_clients(
        x, y, n_clients=12, strategy="non-IID", n_classes=6, seed=0, classes_per_client=3
    )
    for client in clients:
        assert (client.class_counts(6) > 0).sum() <= 3


def test_balanced_non_iid_gives_every_client_the_same_N():
    """The accountant calibrates delta against a single N."""
    x, y = _toy(n=6000)
    clients = partition_clients(
        x, y, n_clients=50, strategy="non-IID", n_classes=6, seed=0, balance=True
    )
    assert len({len(c) for c in clients}) == 1


def test_unbalanced_non_iid_produces_uneven_shards():
    """The released split consumes pools, so later clients get less."""
    x, y = _toy(n=6000)
    clients = partition_clients(
        x, y, n_clients=50, strategy="non-IID", n_classes=6, seed=0, balance=False
    )
    assert len({len(c) for c in clients}) > 1


def test_unbalanced_split_never_starves_a_client():
    """Pool exhaustion must degrade to reuse, not to an empty shard."""
    x, y = _toy(n=1200)
    clients = partition_clients(
        x, y, n_clients=100, strategy="non-IID", n_classes=6, seed=3, balance=False
    )
    assert all(len(c) > 0 for c in clients)


def test_too_many_clients_is_reported_clearly():
    x, y = _toy(n=50)
    with pytest.raises(ValueError, match="cannot be split"):
        partition_clients(x, y, n_clients=1000, strategy="IID")


def test_unknown_partition_rejected():
    x, y = _toy()
    with pytest.raises(ValueError, match="partition must be one of"):
        partition_clients(x, y, n_clients=2, strategy="stratified")


def test_partition_name_is_case_insensitive():
    x, y = _toy()
    assert len(partition_clients(x, y, n_clients=4, strategy="iid", seed=0)) == 4
    assert len(partition_clients(x, y, n_clients=4, strategy="NON-IID", seed=0)) == 4


# ------------------------------------------------------------ shipped dataset


def test_shipped_ucihar_is_discoverable():
    assert "UCIHAR" in available_datasets(DATASET_ROOT)


def test_shipped_ucihar_has_the_expected_shape():
    dataset = load_dataset("UCIHAR", n_clients=10, partition="IID", root=DATASET_ROOT, seed=0)
    assert dataset.n_features == 561
    assert dataset.n_classes == 6
    assert len(dataset.test_y) == 2947
    assert sum(len(c) for c in dataset.clients) <= 7352


def test_shipped_ucihar_rows_are_l2_normalised():
    dataset = load_dataset("UCIHAR", n_clients=4, partition="IID", root=DATASET_ROOT, seed=0)
    norms = np.linalg.norm(dataset.clients[0].x, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_missing_dataset_gives_a_helpful_error():
    with pytest.raises(FileNotFoundError):
        load_dataset("NOT_A_DATASET", n_clients=2, root=DATASET_ROOT)


def test_l2_normalisation_matches_sklearn_normalizer():
    """Pin equivalence to the released code's `Normalizer(norm="l2")`.

    scikit-learn is not a dependency (the operation is one row-wise division),
    so this check is skipped when it is absent. It exists to stop the
    preprocessing silently drifting away from the published implementation.
    """
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

    from privatedfl.data import _l2_normalise

    rng = np.random.default_rng(0)
    train = rng.uniform(-1, 1, (200, 64)).astype(np.float32)
    test = rng.uniform(-1, 1, (50, 64)).astype(np.float32)

    mine_train, mine_test = _l2_normalise(train.copy(), test.copy())

    normalizer = sklearn_preprocessing.Normalizer(norm="l2")
    theirs_train = normalizer.fit_transform(train)
    theirs_test = normalizer.transform(test)

    # float32 reduction order differs by ~1 ULP; the operation is the same.
    assert np.allclose(mine_train, theirs_train, rtol=1e-6, atol=1e-7)
    assert np.allclose(mine_test, theirs_test, rtol=1e-6, atol=1e-7)
    assert np.allclose(np.linalg.norm(mine_train, axis=1), 1.0)
