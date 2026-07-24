"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["ExperimentConfig"]


@dataclass
class ExperimentConfig:
    """Every knob for one PrivateDFL run.

    Attributes
    ----------
    n_clients:
        Ring size ``K``. Clients update in sequence; there is no server.
    rounds:
        Number of full passes ``R`` around the ring.
    dimensions:
        Hypervector size ``D``. Larger is more expressive but also scales the
        injected noise variance linearly, so the benefit saturates.
    epsilon:
        Privacy budget. Smaller means stronger privacy and more noise.
    delta0:
        Privacy-loss coefficient; effective ``delta = delta0 / (t N)``.
    similarity:
        ``"dot"`` reproduces the released implementation; ``"cosine"`` follows
        Eq. (3) of the paper literally.
    """

    # data
    dataset: str = "UCIHAR"
    partition: str = "non-IID"
    classes_per_client: int = 2
    samples_per_client: int | None = None
    balance_clients: bool = True
    data_root: str = "Dataset"

    # ring
    n_clients: int = 100
    rounds: int = 30
    local_epochs: int = 1

    # model
    dimensions: int = 2000
    similarity: str = "dot"

    # privacy
    differential_privacy: bool = True
    epsilon: float = 0.5
    delta0: float = 1e-3

    # runtime
    seed: int = 55
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    output_dir: str = "results"
    run_name: str | None = None
    eval_every: int = 1

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if self.similarity not in ("dot", "cosine"):
            raise ValueError(f"similarity must be 'dot' or 'cosine', got {self.similarity!r}")
        if self.differential_privacy:
            if self.epsilon <= 0:
                raise ValueError("epsilon must be > 0 when differential privacy is enabled")
            if not 0 < self.delta0 < 1:
                raise ValueError("delta0 must lie in (0, 1)")

    # ------------------------------------------------------------------ io

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        import yaml  # noqa: PLC0415

        with Path(path).open() as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> ExperimentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def describe(self) -> str:
        privacy = (
            f"eps={self.epsilon:g}, delta0={self.delta0:g}"
            if self.differential_privacy
            else "DP disabled"
        )
        return (
            f"{self.dataset} ({self.partition}) | K={self.n_clients} | R={self.rounds} | "
            f"D={self.dimensions} | {privacy} | seed={self.seed}"
        )
