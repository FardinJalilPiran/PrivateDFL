#!/usr/bin/env python3
"""Sweep the privacy budget and privacy-loss coefficient (Figure 5).

Accuracy should fall sharply as ``epsilon`` tightens (variance ~ 1/eps^2) and
barely move with ``delta0`` (variance ~ ln(1/delta0)).

Usage
-----
    python scripts/sweep_privacy.py --epsilons 1 0.5 0.1 0.05 --delta0s 1e-3 1e-9
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

from privatedfl import ExperimentConfig, load_dataset, run_privatedfl
from privatedfl.utils import configure_logging, timestamp

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="UCIHAR")
    parser.add_argument("--partition", choices=["IID", "non-IID"], default="non-IID")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 0.5, 0.1, 0.05, 0.01])
    parser.add_argument("--delta0s", type=float, nargs="+", default=[1e-3, 1e-9, 1e-15])
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    dataset = load_dataset(
        args.dataset,
        n_clients=args.clients,
        partition=args.partition,
        root=args.data_root,
        seed=args.seed,
    )
    LOGGER.info("%s", dataset.summary())

    grid = list(itertools.product(args.epsilons, args.delta0s))
    records = []
    for index, (epsilon, delta0) in enumerate(grid, start=1):
        LOGGER.info("[%d/%d] eps=%g  delta0=%g", index, len(grid), epsilon, delta0)
        config = ExperimentConfig(
            dataset=args.dataset,
            partition=args.partition,
            n_clients=args.clients,
            rounds=args.rounds,
            dimensions=args.dimensions,
            epsilon=epsilon,
            delta0=delta0,
            seed=args.seed,
            data_root=args.data_root,
            eval_every=args.rounds,
        )
        history = run_privatedfl(dataset, config, progress=False)
        records.append(
            {
                "epsilon": epsilon,
                "delta0": delta0,
                "accuracy": history.final_report.accuracy,
                "macro_fpr": history.final_report.macro_fpr,
                "macro_fnr": history.final_report.macro_fnr,
                "seconds": history.total_seconds,
            }
        )
        LOGGER.info("    accuracy = %.2f%%", records[-1]["accuracy"] * 100)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"privacy-sweep-{timestamp()}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(
            {"dataset": args.dataset, "partition": args.partition, "results": records}, indent=2
        )
    )
    LOGGER.info("Wrote %s", output_dir / f"{stem}.json")
    _maybe_plot(records, args, output_dir / f"{stem}.png")
    return 0


def _maybe_plot(records, args, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        LOGGER.info("matplotlib not installed; skipping the heatmap.")
        return

    grid = np.zeros((len(args.epsilons), len(args.delta0s)))
    for record in records:
        grid[args.epsilons.index(record["epsilon"]), args.delta0s.index(record["delta0"])] = (
            record["accuracy"] * 100
        )

    fig, ax = plt.subplots(figsize=(1.6 * len(args.delta0s) + 3, 0.8 * len(args.epsilons) + 2))
    image = ax.imshow(grid, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(args.delta0s)), [f"{d:g}" for d in args.delta0s])
    ax.set_yticks(range(len(args.epsilons)), [f"{e:g}" for e in args.epsilons])
    ax.set_xlabel(r"Privacy loss coefficient  $\delta_0$")
    ax.set_ylabel(r"Privacy budget  $\epsilon$")
    ax.set_title(f"PrivateDFL accuracy (%) - {args.dataset} ({args.partition})")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
