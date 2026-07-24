#!/usr/bin/env python3
"""Report accuracy across random seeds.

Under a tight privacy budget the first-round perturbation is large relative to
the signal, so a single run is a noisy estimate. This script reports the mean
and spread so results can be quoted honestly.

Usage
-----
    python scripts/seed_variance.py --seeds 10 --clients 100 --rounds 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from privatedfl import ExperimentConfig, load_dataset, run_privatedfl
from privatedfl.utils import configure_logging, timestamp

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="UCIHAR")
    parser.add_argument("--partition", choices=["IID", "non-IID"], default="non-IID")
    parser.add_argument("--seeds", type=int, default=10, help="run seeds 0..S-1")
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--delta0", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    accuracies = []
    for seed in range(args.seeds):
        dataset = load_dataset(
            args.dataset,
            n_clients=args.clients,
            partition=args.partition,
            root=args.data_root,
            seed=seed,
        )
        config = ExperimentConfig(
            dataset=args.dataset,
            partition=args.partition,
            n_clients=args.clients,
            rounds=args.rounds,
            dimensions=args.dimensions,
            epsilon=args.epsilon,
            delta0=args.delta0,
            seed=seed,
            data_root=args.data_root,
            eval_every=args.rounds,
        )
        history = run_privatedfl(dataset, config, progress=False)
        accuracies.append(history.final_report.accuracy * 100)
        LOGGER.info("seed %2d  accuracy = %.2f%%", seed, accuracies[-1])

    values = np.array(accuracies)
    print()
    print(
        f"{args.dataset} ({args.partition}), K={args.clients}, R={args.rounds}, "
        f"D={args.dimensions}, eps={args.epsilon:g}"
    )
    print(f"  mean   {values.mean():.2f}%")
    print(f"  sd     {values.std(ddof=1):.2f}")
    print(f"  range  [{values.min():.2f}, {values.max():.2f}]")
    print(f"  median {np.median(values):.2f}%")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"seed-variance-{timestamp()}.json"
    path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "accuracies": accuracies,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            },
            indent=2,
        )
    )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
