"""Command-line entry point.

Examples
--------
    privatedfl --config configs/ucihar_noniid.yaml
    privatedfl --dataset UCIHAR --clients 100 --rounds 30 --epsilon 0.5
    privatedfl --partition IID --dimensions 5000 --no-dp
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ExperimentConfig
from .data import available_datasets, load_dataset
from .decentralized import run_privatedfl
from .utils import configure_logging, save_history, timestamp

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privatedfl",
        description="Privacy-preserving decentralized federated learning with "
        "hyperdimensional computing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file to start from")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default=None, help="dataset stem, e.g. UCIHAR")
    data.add_argument("--partition", choices=["IID", "non-IID"], default=None)
    data.add_argument("--classes-per-client", type=int, default=None)
    data.add_argument(
        "--samples-per-client", type=int, default=None, help="cap on N; default uses the full shard"
    )
    data.add_argument("--data-root", type=str, default=None, help="directory of .choir_dat files")
    data.add_argument(
        "--unbalanced-clients",
        action="store_true",
        help="reproduce the released split, whose shard sizes vary",
    )
    data.add_argument(
        "--list-datasets", action="store_true", help="show available datasets and exit"
    )

    ring = parser.add_argument_group("ring")
    ring.add_argument("--clients", type=int, default=None, dest="n_clients")
    ring.add_argument("--rounds", type=int, default=None)
    ring.add_argument("--local-epochs", type=int, default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--dimensions", type=int, default=None, help="hypervector size D")
    model.add_argument(
        "--similarity",
        choices=["dot", "cosine"],
        default=None,
        help="dot matches the released code; cosine follows Eq. (3)",
    )

    privacy = parser.add_argument_group("privacy")
    privacy.add_argument("--epsilon", type=float, default=None, help="privacy budget")
    privacy.add_argument("--delta0", type=float, default=None, help="privacy loss coefficient")
    privacy.add_argument(
        "--no-dp", action="store_true", help="disable DP (non-private accuracy ceiling)"
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=None)
    runtime.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    runtime.add_argument("--eval-every", type=int, default=None, help="evaluate every N rounds")
    runtime.add_argument("--output-dir", type=str, default=None)
    runtime.add_argument("--run-name", type=str, default=None)
    runtime.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Start from a YAML file (or defaults) and apply explicit CLI flags."""
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    for name in (
        "dataset",
        "partition",
        "classes_per_client",
        "samples_per_client",
        "data_root",
        "n_clients",
        "rounds",
        "local_epochs",
        "dimensions",
        "similarity",
        "epsilon",
        "delta0",
        "seed",
        "device",
        "eval_every",
        "output_dir",
        "run_name",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)

    if args.no_dp:
        config.differential_privacy = False
    if args.unbalanced_clients:
        config.balance_clients = False

    config.__post_init__()  # re-validate after the overrides
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)

    root = args.data_root or ExperimentConfig().data_root
    if args.list_datasets:
        found = available_datasets(root)
        print(f"Datasets in {root}/: " + (", ".join(found) if found else "none found"))
        return 0

    config = config_from_args(args)
    LOGGER.info("Configuration: %s", config.describe())

    dataset = load_dataset(
        name=config.dataset,
        n_clients=config.n_clients,
        partition=config.partition,
        root=config.data_root,
        seed=config.seed,
        samples_per_client=config.samples_per_client,
        classes_per_client=config.classes_per_client,
        balance_clients=config.balance_clients,
    )

    history = run_privatedfl(dataset, config, progress=not args.quiet)

    name = (
        config.run_name
        or f"{config.dataset}-{config.partition}-eps{config.epsilon:g}-{timestamp()}"
    )
    path = save_history(history, config.output_dir, name)

    print()
    print("Final model")
    print("-----------")
    print(f"  {history.final_report}")
    print(f"  best accuracy: {history.best_accuracy * 100:.2f}%")
    print(f"  wall clock:    {history.total_seconds:.1f}s")
    print(f"  results:       {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
