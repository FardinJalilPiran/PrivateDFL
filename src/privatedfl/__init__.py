"""PrivateDFL: privacy-preserving decentralized federated learning.

Reference implementation of

    F. Jalil Piran, Z. Chen, Y. Zhang, Q. Zhou, J. Tang, F. Imani,
    "Privacy-Preserving Decentralized Federated Learning via Explainable
    Adaptive Differential Privacy", arXiv:2509.10691, 2025.

Quick start
-----------
>>> from privatedfl import ExperimentConfig, load_dataset, run_privatedfl
>>> config = ExperimentConfig(n_clients=10, rounds=5, dimensions=1000)
>>> data = load_dataset("UCIHAR", n_clients=config.n_clients)
>>> history = run_privatedfl(data, config)
"""

from .config import ExperimentConfig
from .data import (
    ClientData,
    FederatedDataset,
    available_datasets,
    load_dataset,
    read_choir_file,
)
from .decentralized import RoundResult, TrainingHistory, noise_comparison, run_privatedfl
from .hdc import Encoder, HDModel
from .metrics import ClassificationReport, evaluate
from .privacy import (
    NoiseAccountant,
    blackbox_cumulative_variance,
    cumulative_variance,
    incremental_variance,
    required_variance,
    step_index,
)

__version__ = "1.0.2"

__all__ = [
    "ClassificationReport",
    "ClientData",
    "Encoder",
    "ExperimentConfig",
    "FederatedDataset",
    "HDModel",
    "NoiseAccountant",
    "RoundResult",
    "TrainingHistory",
    "__version__",
    "available_datasets",
    "blackbox_cumulative_variance",
    "cumulative_variance",
    "evaluate",
    "incremental_variance",
    "load_dataset",
    "noise_comparison",
    "read_choir_file",
    "required_variance",
    "run_privatedfl",
    "step_index",
]
