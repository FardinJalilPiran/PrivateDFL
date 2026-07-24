#!/usr/bin/env python3
"""Tracked vs. black-box noise accumulation (Section 4.2).

The accountant's cumulative variance grows like ``ln(KR)``; a scheme that cannot
see the noise already in the model must re-inject the full requirement every
step and grows like ``ln((KR)!)``. No training is involved, so this runs
instantly.

Usage
-----
    python scripts/plot_noise_accounting.py --clients 100 --rounds 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privatedfl.privacy import (
    blackbox_cumulative_variance,
    incremental_variance,
    required_variance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--samples", type=int, default=73, help="samples per client N")
    parser.add_argument("--dimensions", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--delta0", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="results/noise_accounting.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = dict(
        dimensions=args.dimensions,
        epsilon=args.epsilon,
        delta0=args.delta0,
        samples_per_client=args.samples,
    )
    total = args.clients * args.rounds
    steps = list(range(1, total + 1))

    tracked = [required_variance(**kwargs, step=t) for t in steps]
    blackbox = [blackbox_cumulative_variance(**kwargs, step=t) for t in steps]
    added = [incremental_variance(**kwargs, step=t) for t in steps]

    print(f"{'step':>8} {'tracked':>16} {'added':>16} {'black-box':>18} {'ratio':>12}")
    marks = sorted({1, 2, 10, total // 10 or 1, total // 2 or 1, total})
    for t in marks:
        print(
            f"{t:>8} {tracked[t - 1]:>16.2f} {added[t - 1]:>16.4f} "
            f"{blackbox[t - 1]:>18.2f} {blackbox[t - 1] / tracked[t - 1]:>11.1f}x"
        )
    print(
        f"\nAfter {total} updates the untracked scheme carries "
        f"{blackbox[-1] / tracked[-1]:.0f}x the noise variance."
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping the plot.")
        return 0

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))

    left.plot(steps, tracked, label="PrivateDFL (tracked)", lw=2)
    left.plot(steps, blackbox, label="black-box (untracked)", lw=2, ls="--")
    left.set_yscale("log")
    left.set_xlabel("Client update  $t = K(r-1)+k$")
    left.set_ylabel("Cumulative noise variance (log)")
    left.set_title("Tracked accounting vs. worst-case injection")
    left.legend()
    left.grid(alpha=0.3)

    right.plot(steps, added, color="crimson", lw=2)
    right.set_yscale("log")
    right.set_xlabel("Client update  $t$")
    right.set_ylabel(r"Injected variance  $\Gamma_k^r$ (log)")
    right.set_title("Per-update injection collapses after the first client")
    right.grid(alpha=0.3)

    fig.suptitle(
        f"K={args.clients}, R={args.rounds}, N={args.samples}, "
        f"D={args.dimensions}, $\\epsilon$={args.epsilon:g}"
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
