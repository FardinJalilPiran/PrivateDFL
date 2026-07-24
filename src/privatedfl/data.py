"""Dataset loading for PrivateDFL.

Data lives in the ``.choir_dat`` binary format under a ``Dataset/`` directory::

    int32   n_features
    int32   n_classes
    repeated:
        float32 * n_features    feature vector
        int32                   label

The repository ships the preprocessed UCI-HAR split used in the paper. MNIST and
ISOLET follow the same convention: drop ``MNIST_train.choir_dat`` and
``MNIST_test.choir_dat`` into ``Dataset/`` and pass ``--dataset MNIST``.

Features are L2-normalised with the scaler fitted on the training split, exactly
as in the released implementation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ClientData",
    "FederatedDataset",
    "read_choir_file",
    "load_dataset",
    "available_datasets",
]


@dataclass
class ClientData:
    """Data held locally by one client in the ring."""

    client_id: str
    x: np.ndarray  # (n_samples, n_features), float32
    y: np.ndarray  # (n_samples,), int64

    def __len__(self) -> int:
        return len(self.y)

    def class_counts(self, n_classes: int) -> np.ndarray:
        return np.bincount(self.y, minlength=n_classes)


@dataclass
class FederatedDataset:
    """A client partition plus the held-out test split."""

    name: str
    clients: list[ClientData]
    test_x: np.ndarray
    test_y: np.ndarray
    n_classes: int
    n_features: int
    metadata: dict = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    def min_client_samples(self) -> int:
        return min(len(c) for c in self.clients)

    def max_client_samples(self) -> int:
        """Largest shard held by any client.

        This is the value the DP accounting must calibrate against: delta scales
        as ``delta_0 / N``, so the client holding the most data is the one whose
        privacy is hardest to guarantee.
        """
        return max(len(c) for c in self.clients)

    def summary(self) -> str:
        sizes = [len(c) for c in self.clients]
        classes_held = [int((c.class_counts(self.n_classes) > 0).sum()) for c in self.clients]
        return (
            f"{self.name}: {self.n_clients} clients, {self.n_features} features, "
            f"{self.n_classes} classes | client sizes min={min(sizes)} max={max(sizes)} "
            f"total={sum(sizes)} | classes/client min={min(classes_held)} "
            f"max={max(classes_held)} | test={len(self.test_y)}"
        )


def read_choir_file(path: str | Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Read a ``.choir_dat`` file.

    Returns ``(n_features, n_classes, X, y)``. The whole payload is read at once
    and reinterpreted, rather than unpacked value by value: for UCI-HAR that is
    4.1 million ``struct.unpack`` calls avoided.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Expected a .choir_dat file. The UCI-HAR split ships with this "
            "repository under Dataset/; for MNIST or ISOLET, place the "
            "corresponding files there using the same naming convention."
        )

    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) < 8:
            raise ValueError(f"{path} is too short to contain a valid header.")
        n_features, n_classes = struct.unpack("ii", header)
        payload = handle.read()

    if n_features <= 0 or n_classes <= 0:
        raise ValueError(f"{path} has an implausible header: {n_features=}, {n_classes=}")

    record_bytes = n_features * 4 + 4
    n_records, remainder = divmod(len(payload), record_bytes)
    if remainder:
        raise ValueError(
            f"{path} has {remainder} trailing bytes that do not form a complete record; "
            "the file may be truncated or the header may be wrong."
        )

    raw = np.frombuffer(payload, dtype=np.uint8).reshape(n_records, record_bytes)
    x = raw[:, : n_features * 4].copy().view(np.float32)
    y = raw[:, n_features * 4 :].copy().view(np.int32).ravel().astype(np.int64)
    return n_features, n_classes, x, y


def available_datasets(root: str | Path = "Dataset") -> list[str]:
    """Names for which both a train and a test file are present."""
    root = Path(root)
    if not root.exists():
        return []
    names = {p.name[: -len("_train.choir_dat")] for p in root.glob("*_train.choir_dat")}
    return sorted(n for n in names if (root / f"{n}_test.choir_dat").exists())


def load_dataset(
    name: str = "UCIHAR",
    n_clients: int = 100,
    partition: str = "non-IID",
    root: str | Path = "Dataset",
    seed: int = 42,
    samples_per_client: int | None = None,
    classes_per_client: int = 2,
    balance_clients: bool = True,
) -> FederatedDataset:
    """Load a ``.choir_dat`` dataset and split it across the ring.

    Parameters
    ----------
    name:
        Dataset stem, e.g. ``"UCIHAR"``. See :func:`available_datasets`.
    n_clients:
        Ring size ``K``.
    partition:
        ``"IID"`` or ``"non-IID"``.
    samples_per_client:
        Cap on ``N``. ``None`` uses the natural shard size. The accountant
        assumes a common ``N``, so shards are equalised either way.
    classes_per_client:
        Labels per client under ``non-IID``.
    balance_clients:
        ``True`` gives every client the same ``N``, as the accountant assumes.
        ``False`` reproduces the released split, whose shards vary in size.
    """
    from .partition import partition_clients  # local import avoids a cycle

    root = Path(root)
    train_path = root / f"{name}_train.choir_dat"
    test_path = root / f"{name}_test.choir_dat"

    n_features, n_classes, train_x, train_y = read_choir_file(train_path)
    test_features, test_classes, test_x, test_y = read_choir_file(test_path)
    if n_features != test_features or n_classes != test_classes:
        raise ValueError(
            f"Train/test mismatch for {name}: features {n_features} vs {test_features}, "
            f"classes {n_classes} vs {test_classes}."
        )

    train_x, test_x = _l2_normalise(train_x, test_x)

    clients = partition_clients(
        train_x,
        train_y,
        n_clients=n_clients,
        strategy=partition,
        n_classes=n_classes,
        seed=seed,
        samples_per_client=samples_per_client,
        classes_per_client=classes_per_client,
        balance=balance_clients,
    )

    return FederatedDataset(
        name=name,
        clients=clients,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
        n_features=n_features,
        metadata={"partition": partition, "seed": seed},
    )


def _l2_normalise(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each row to unit L2 norm.

    Matches ``sklearn.preprocessing.Normalizer(norm="l2")``. Row-wise scaling
    carries no cross-sample state, so fitting on train and applying to test is
    exactly the same operation - no information crosses the split.
    """

    def scale(a: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        return (a / np.maximum(norms, 1e-12)).astype(np.float32)

    return scale(train_x), scale(test_x)
