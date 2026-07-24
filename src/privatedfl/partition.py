"""Client partitioning: IID and label-skewed non-IID.

The accountant calibrates ``delta = delta0 / (t N)`` against a single ``N``, so
both strategies produce equal-sized shards. Under ``non-IID`` each client is
restricted to ``classes_per_client`` labels, drawn round-robin so that class
coverage across the ring stays balanced even when ``K`` greatly exceeds the
number of classes.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import ClientData

LOGGER = logging.getLogger(__name__)

__all__ = ["partition_clients", "VALID_PARTITIONS"]

VALID_PARTITIONS = ("IID", "non-IID")


def partition_clients(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    strategy: str = "non-IID",
    n_classes: int | None = None,
    seed: int = 42,
    samples_per_client: int | None = None,
    classes_per_client: int = 2,
    balance: bool = True,
) -> list[ClientData]:
    """Split ``(x, y)`` into ``n_clients`` shards.

    ``balance=True`` gives every client the same ``N``, which is what the
    accountant's ``delta = delta0 / (t N)`` assumes. Class pools are reused when
    a class is shared by more clients than it has samples for.

    ``balance=False`` reproduces the released implementation: pools are consumed
    without replacement, so once a class runs dry the remaining clients that
    hold it receive smaller shards. Shard sizes then vary, and ``N`` is only
    well defined for the very first client. See ``docs/REPRODUCING.md``.
    """
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    canonical = {p.lower(): p for p in VALID_PARTITIONS}
    key = strategy.lower()
    if key not in canonical:
        raise ValueError(f"partition must be one of {VALID_PARTITIONS}, got {strategy!r}")
    strategy = canonical[key]

    if n_classes is None:
        n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)

    budget = len(y) // n_clients
    if samples_per_client is not None:
        budget = min(budget, samples_per_client)
    if budget < 1:
        raise ValueError(
            f"{len(y)} training samples cannot be split across {n_clients} clients. "
            "Reduce the client count."
        )

    if strategy == "IID":
        shards = _split_iid(len(y), n_clients, budget, rng)
    else:
        shards = _split_label_skewed(
            y, n_clients, n_classes, budget, classes_per_client, rng, balance
        )

    return [
        ClientData(client_id=f"client_{i + 1:03d}", x=x[idx], y=y[idx])
        for i, idx in enumerate(shards)
    ]


def _split_iid(n_samples: int, n_clients: int, budget: int, rng) -> list[np.ndarray]:
    order = rng.permutation(n_samples)
    return [order[k * budget : (k + 1) * budget] for k in range(n_clients)]


def _split_label_skewed(
    y: np.ndarray,
    n_clients: int,
    n_classes: int,
    budget: int,
    classes_per_client: int,
    rng,
    balance: bool = True,
) -> list[np.ndarray]:
    """Give each client a small, fixed set of labels.

    Classes are handed out round-robin over a shuffled ordering, so with
    ``K > n_classes`` every class is reused a similar number of times rather
    than a few classes being exhausted first.
    """
    if classes_per_client < 1:
        raise ValueError(f"classes_per_client must be >= 1, got {classes_per_client}")
    classes_per_client = min(classes_per_client, n_classes)

    order = rng.permutation(n_classes)
    assignments = [
        [int(order[(k * classes_per_client + j) % n_classes]) for j in range(classes_per_client)]
        for k in range(n_clients)
    ]

    by_class = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(n_classes)}
    original = {c: list(pool) for c, pool in by_class.items()}
    cursors = dict.fromkeys(range(n_classes), 0)
    per_class = max(1, budget // classes_per_client)
    starved = 0

    def draw_consuming(cls: int, count: int) -> np.ndarray:
        """Released behaviour: take from the pool and remove; may return fewer."""
        pool = by_class[cls]
        taken = pool[:count]
        by_class[cls] = pool[count:]
        return np.array(taken, dtype=np.int64)

    def draw_wrapping(cls: int, count: int) -> np.ndarray:
        """Balanced behaviour: walk the pool, restarting when it runs out."""
        pool = original[cls] if balance is False else by_class[cls]
        if not pool:
            return np.empty(0, dtype=np.int64)
        taken, start = [], cursors[cls] % len(pool)
        while len(taken) < count:
            end = min(start + count - len(taken), len(pool))
            taken.extend(pool[start:end])
            start = end % len(pool)
        cursors[cls] = start
        return np.array(taken, dtype=np.int64)

    draw = draw_wrapping if balance else draw_consuming

    shards = []
    for client_classes in assignments:
        picked = [p for p in (draw(c, per_class) for c in client_classes) if len(p)]
        idx = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)

        if balance and len(idx) < budget:  # top up from the client's own classes
            extra = draw(client_classes[0], budget - len(idx))
            if len(extra):
                idx = np.concatenate([idx, extra])
            idx = idx[:budget]

        if len(idx) == 0:
            # Consuming draws can exhaust a class entirely, leaving a client
            # with nothing. Reuse samples for that client rather than aborting;
            # an empty shard would make its update a no-op anyway.
            starved += 1
            idx = draw_wrapping(client_classes[0], per_class)
            if len(idx) == 0:
                raise ValueError(
                    f"Class {client_classes[0]} has no samples at all; check the dataset."
                )
        shards.append(rng.permutation(idx))

    if starved:
        LOGGER.warning(
            "%d client(s) exhausted their class pools and were given reused samples. "
            "This is inherent to the unbalanced split; set balance_clients=true to avoid it.",
            starved,
        )

    if balance:
        smallest = min(len(s) for s in shards)
        if smallest < budget:
            LOGGER.warning(
                "Shards trimmed to %d samples (requested %d) because some class pools are small.",
                smallest,
                budget,
            )
        shards = [s[:smallest] for s in shards]
    return shards
