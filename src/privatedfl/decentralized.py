"""The PrivateDFL training loop: a serverless ring.

There is no aggregation step. Clients are arranged in a ring and the model is
handed from one to the next:

Round 1
    Client ``k`` builds class prototypes from its own data, perturbs them, and
    adds them into the shared model. The model accumulates contributions rather
    than averaging them.

Round r >= 2
    Client ``k`` receives the model, corrects it against its local data
    (Eq. 5), perturbs it, and passes it on.

Every client perturbs with the *incremental* variance for its global step
``t = K(r-1) + k`` - the difference between what the budget now requires and
what the model it just received already carries. See
:mod:`privatedfl.privacy`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import ExperimentConfig
from .data import FederatedDataset
from .hdc import Encoder, HDModel
from .metrics import ClassificationReport, evaluate
from .privacy import NoiseAccountant
from .utils import resolve_device, set_seed

LOGGER = logging.getLogger(__name__)

__all__ = ["RoundResult", "TrainingHistory", "run_privatedfl"]


@dataclass
class RoundResult:
    round_index: int
    accuracy: float
    macro_fpr: float
    macro_fnr: float
    steps_completed: int
    noise_required: float
    noise_added_this_round: float
    noise_blackbox: float
    seconds: float

    def to_dict(self) -> dict:
        return {
            "round": self.round_index,
            "accuracy": self.accuracy,
            "macro_fpr": self.macro_fpr,
            "macro_fnr": self.macro_fnr,
            "steps_completed": self.steps_completed,
            "noise_variance_required": self.noise_required,
            "noise_variance_added_this_round": self.noise_added_this_round,
            "noise_variance_blackbox": self.noise_blackbox,
            "seconds": self.seconds,
        }


@dataclass
class TrainingHistory:
    config: ExperimentConfig
    rounds: list[RoundResult] = field(default_factory=list)
    final_report: ClassificationReport | None = None
    dataset_summary: str = ""
    samples_per_client: int = 0
    total_seconds: float = 0.0

    @property
    def accuracies(self) -> list[float]:
        return [r.accuracy for r in self.rounds]

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracies) if self.rounds else 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "dataset_summary": self.dataset_summary,
            "samples_per_client": self.samples_per_client,
            "total_seconds": self.total_seconds,
            "best_accuracy": self.best_accuracy,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_report": self.final_report.to_dict() if self.final_report else None,
        }


def run_privatedfl(
    dataset: FederatedDataset,
    config: ExperimentConfig,
    progress: bool = True,
) -> TrainingHistory:
    """Run ``config.rounds`` passes around the ring and return the history."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    generator = torch.Generator().manual_seed(config.seed)

    n_classes = dataset.n_classes
    # delta = delta_0 / N, so a larger N demands MORE noise. When shards are
    # uneven the accountant must calibrate against the largest one, otherwise
    # the clients holding the most data are under-protected relative to the
    # budget they were promised. The proofs in the paper take worst cases
    # throughout; this is the same principle applied to unequal shard sizes.
    samples_per_client = dataset.max_client_samples()

    LOGGER.info("Device: %s", device)
    LOGGER.info("%s", dataset.summary())
    LOGGER.info(
        "Samples per client for DP calibration (N): %d (largest shard; smallest is %d)",
        samples_per_client,
        dataset.min_client_samples(),
    )

    encoder = Encoder(
        n_features=dataset.n_features,
        dimensions=config.dimensions,
        device=device,
        seed=config.seed,
    )
    accountant = NoiseAccountant(
        dimensions=config.dimensions,
        epsilon=config.epsilon,
        delta0=config.delta0,
        samples_per_client=samples_per_client,
        n_clients=dataset.n_clients,
        enabled=config.differential_privacy,
    )

    # Encode once up front. The basis is fixed, so a client's hypervectors never
    # change between rounds; re-encoding every round is pure waste.
    LOGGER.info("Encoding client data...")
    encoded_clients = [
        (
            encoder.encode(torch.from_numpy(client.x)),
            torch.from_numpy(client.y).long().to(device),
        )
        for client in dataset.clients
    ]
    test_encoded = encoder.encode(torch.from_numpy(dataset.test_x))
    test_labels = torch.from_numpy(dataset.test_y).long()

    history = TrainingHistory(
        config=config,
        dataset_summary=dataset.summary(),
        samples_per_client=samples_per_client,
    )
    model = HDModel.zeros(n_classes, config.dimensions, device=device)
    started = time.perf_counter()

    for round_index in range(1, config.rounds + 1):
        round_started = time.perf_counter()
        added_this_round = 0.0

        for client_index, (encoded, labels) in enumerate(encoded_clients, start=1):
            if round_index == 1:
                # Build local prototypes, perturb, and fold into the shared model.
                local = HDModel.from_samples(encoded, labels, n_classes)
                noisy = accountant.perturb(
                    local.class_hypervectors, round_index, client_index, generator=generator
                )
                model.class_hypervectors += noisy
            else:
                # Correct the received model, perturb, and pass it along.
                model.retrain(
                    encoded,
                    labels,
                    n_epochs=config.local_epochs,
                    similarity=config.similarity,
                    generator=generator,
                )
                model = HDModel(
                    accountant.perturb(
                        model.class_hypervectors, round_index, client_index, generator=generator
                    )
                )
            added_this_round += accountant.report(round_index, client_index)["added"]

        report_at_end = accountant.report(round_index, dataset.n_clients)
        due = round_index % config.eval_every == 0 or round_index == config.rounds

        if due:
            predictions = model.predict(test_encoded, config.similarity).cpu().numpy()
            report = evaluate(test_labels.numpy(), predictions, n_classes)
            history.final_report = report
        else:
            report = history.final_report or evaluate(
                test_labels.numpy(),
                model.predict(test_encoded, config.similarity).cpu().numpy(),
                n_classes,
            )

        result = RoundResult(
            round_index=round_index,
            accuracy=report.accuracy,
            macro_fpr=report.macro_fpr,
            macro_fnr=report.macro_fnr,
            steps_completed=int(report_at_end["step"]),
            noise_required=report_at_end["required"],
            noise_added_this_round=added_this_round,
            noise_blackbox=report_at_end["blackbox"],
            seconds=time.perf_counter() - round_started,
        )
        history.rounds.append(result)

        if progress and due:
            if report_at_end["blackbox"] > 0:
                ratio = report_at_end["blackbox"] / max(report_at_end["required"], 1e-12)
                saving = f"  blackbox/tracked={ratio:8.1f}x"
            else:
                saving = ""
            LOGGER.info(
                "round %3d/%d  acc=%.4f%s  (%.1fs)",
                round_index,
                config.rounds,
                report.accuracy,
                saving,
                result.seconds,
            )

    history.total_seconds = time.perf_counter() - started
    return history


def noise_comparison(config: ExperimentConfig, samples_per_client: int) -> dict[str, np.ndarray]:
    """Tracked vs. black-box cumulative variance per step, without training.

    Reproduces the comparison in Section 4.2: the accountant grows like
    ``ln(KR)`` while the untracked baseline grows like ``ln((KR)!)``.
    """
    from .privacy import blackbox_cumulative_variance, required_variance

    total = config.n_clients * config.rounds
    steps = np.arange(1, total + 1)
    args = (config.dimensions, config.epsilon, config.delta0, samples_per_client)
    return {
        "step": steps,
        "tracked": np.array([required_variance(*args, int(t)) for t in steps]),
        "blackbox": np.array([blackbox_cumulative_variance(*args, int(t)) for t in steps]),
    }
