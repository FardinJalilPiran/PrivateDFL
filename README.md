# PrivateDFL

Reference implementation of **PrivateDFL** — a serverless decentralized federated learning framework that combines hyperdimensional computing with an explainable, adaptive differential privacy noise accountant.

[![arXiv](https://img.shields.io/badge/arXiv-2509.10691-b31b1b)](https://arxiv.org/abs/2509.10691)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

> Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani.
> *Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy.*
> arXiv:2509.10691, 2025.

---

## The idea in one paragraph

Decentralized federated learning removes the central server, but the model updates clients pass to one another still leak — inversion attacks reconstruct training data, membership inference reveals who contributed. Differential privacy fixes that, except every existing DP-DFL method is a black box: a client cannot see how much noise is already in the model it received, so it must assume the worst and inject a full dose. Over `K` clients and `R` rounds that compounds like `ln((KR)!)` and the model collapses. PrivateDFL makes the noise *auditable*. Because the hyperdimensional model is transparent, each client can compute exactly what the budget requires now, subtract what the model already carries, and inject only the difference. Cumulative noise then grows like `ln(KR)` instead — at 100 clients over 30 rounds, roughly **2,800× less noise variance** for the same formal `(ε, δ)` guarantee.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/FardinJalilPiran/PrivateDFL.git
cd PrivateDFL

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Core dependencies are just PyTorch, NumPy and PyYAML. The `all` extra adds matplotlib (for the scripts), pytest and ruff.

### Check it works

```bash
make smoke        # 10 clients, 3 rounds, ~10s on a laptop CPU
pytest            # full test suite
```

## Quick start

The UCI-HAR data ships with the repository, so there is nothing to download.

```bash
# Paper configuration: 100 clients in a ring, 30 rounds, non-IID
privatedfl --config configs/ucihar_noniid.yaml

# IID partition
privatedfl --config configs/ucihar_iid.yaml

# Non-private ceiling, to see what the guarantee costs
privatedfl --config configs/no_privacy.yaml

# Explore the budget
privatedfl --epsilon 0.05 --dimensions 5000 --rounds 50
```

From Python:

```python
from privatedfl import ExperimentConfig, load_dataset, run_privatedfl

config = ExperimentConfig(n_clients=100, rounds=30, dimensions=2000, epsilon=0.5)
dataset = load_dataset("UCIHAR", n_clients=config.n_clients, partition="non-IID")
history = run_privatedfl(dataset, config)

print(history.final_report)     # accuracy, macro FPR, macro FNR
print(history.accuracies)       # per-round accuracy
```

Each run writes a JSON record to `results/` with the configuration, per-round accuracy, and the required / injected / black-box noise variances.

## How the accountant works

In a ring there is no aggregation step, so the model's state is fully described by how many client updates have touched it. Writing that global step as `t = K(r−1) + k` collapses the paper's four theorems into two cases:

| Quantity | Variance |
| --- | --- |
| Required after step `t` | `(2D/ε²)·ln(1.25·t·N/δ₀)` |
| **Injected at step `t = 1`** | the whole requirement (Theorem 2) |
| **Injected at step `t ≥ 2`** | `(2D/ε²)·ln(t/(t−1))` (Theorems 3–5) |
| Black-box cumulative | `(2D/ε²)·[t·ln(1.25N/δ₀) + ln(t!)]` |

The logarithms telescope, so the injected increments sum *exactly* to the requirement — never more, never less. `tests/test_privacy.py` asserts this to nine significant figures.

Two consequences worth knowing before tuning anything:

- **The first client pays almost everything.** At the default settings step 1 injects a variance of ~183,000; step 3000 injects ~5. Round 1 accuracy is therefore near chance, and the model climbs out over subsequent rounds as retraining rebuilds signal against a nearly-frozen noise floor.
- **`ε` is the expensive knob, `δ₀` is nearly free.** Variance scales as `1/ε²` but only as `ln(1/δ₀)`. Tightening `δ₀` by fifteen orders of magnitude costs under 4× variance — which is why the paper's `δ₀` sweep is almost flat.

Inspect the whole schedule without training anything:

```bash
python scripts/plot_noise_accounting.py --clients 100 --rounds 30
```

## The encoder is deliberately unchanged

Encoding is exactly Eq. (1) of the paper and exactly the released code:

```python
basis = torch.randn(n_features, D)     # B_d ~ N(0, 1)
H = torch.cos(X @ basis)               # h_d = cos(F . B_d)
```

Hypervectors are **real-valued in [−1, 1], not binarised**. This is not incidental: the sensitivity bound `Δg = √D` in Proofs 1–4 follows precisely from cosine outputs being bounded by 1, so binarising or rescaling here would silently invalidate every noise calibration downstream. `tests/test_hdc.py` pins this behaviour.

## About the data

`Dataset/` holds the preprocessed UCI-HAR split in `.choir_dat` format — 7,352 training and 2,947 test samples, 561 features, 6 activity classes. The loader applies L2 row normalisation, matching the released implementation.

The paper also evaluates MNIST and ISOLET. Both use the same container format; drop `MNIST_train.choir_dat` and `MNIST_test.choir_dat` into `Dataset/` and pass `--dataset MNIST`. See [`Dataset/README.md`](Dataset/README.md) for the byte layout, and `privatedfl --list-datasets` to see what is present.

## Repository layout

```
PrivateDFL/
├── src/privatedfl/
│   ├── config.py         ExperimentConfig: every knob, validated
│   ├── data.py           .choir_dat reader, L2 normalisation
│   ├── partition.py      IID and label-skewed non-IID splits
│   ├── hdc.py            encoder, class prototypes, retrain, predict
│   ├── privacy.py        the noise accountant (Theorems 2-5)
│   ├── decentralized.py  the ring loop
│   ├── metrics.py        accuracy, macro FPR/FNR
│   ├── utils.py          seeding, device selection, result I/O
│   └── cli.py            the `privatedfl` command
├── Dataset/              preprocessed UCI-HAR (.choir_dat)
├── configs/              paper, IID, non-private, quick
├── scripts/              noise accounting, privacy sweep, seed variance
├── notebooks/            walkthrough of the framework
├── tests/                pytest suite
└── docs/                 algorithm notes, reproduction guide
```

## Configuration

Any config field can be overridden on the command line; run `privatedfl --help` for the full list.

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset` | `UCIHAR` | stem of the files in `Dataset/` |
| `--partition` | `non-IID` | `IID` or `non-IID` |
| `--classes-per-client` | `2` | labels per client under `non-IID` |
| `--clients` | `100` | ring size `K` |
| `--rounds` | `30` | passes around the ring `R` |
| `--dimensions` | `2000` | hypervector size `D` |
| `--epsilon` | `0.5` | privacy budget |
| `--delta0` | `1e-3` | privacy-loss coefficient |
| `--similarity` | `dot` | `dot` matches the released code; `cosine` follows Eq. (3) |
| `--unbalanced-clients` | off | reproduce the released split's uneven shard sizes |
| `--no-dp` | off | non-private baseline |
| `--device` | `auto` | CUDA if available |

## Reproducing

```bash
# Section 4.2: tracked vs. black-box noise growth
python scripts/plot_noise_accounting.py

# Figure 5: accuracy across the epsilon x delta0 grid
python scripts/sweep_privacy.py --epsilons 1 0.5 0.1 0.05 --delta0s 1e-3 1e-9 1e-15

# Spread across random seeds
python scripts/seed_variance.py --seeds 10
```

**Please read [`docs/REPRODUCING.md`](docs/REPRODUCING.md) before quoting a single number.** Under a tight budget the round-1 perturbation is large relative to the signal, and final accuracy varies by roughly ±3 points across seeds. Reporting a mean and spread over several seeds is much more defensible than reporting one run.

## Development

```bash
make dev          # editable install with all extras
make test         # pytest
make lint         # ruff check + format check
make format       # auto-fix
```

## Citation

```bibtex
@article{jalil2025privacy,
  title   = {Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy},
  author  = {Piran, Fardin Jalil and Chen, Zhiling and Zhang, Yang and Zhou, Qianyu and Tang, Jiong and Imani, Farhad},
  journal = {arXiv preprint arXiv:2509.10691},
  year    = {2025}
}
```

## Acknowledgments

Supported by the National Science Foundation, United States [grant number 2434519].

## License

MIT — see [LICENSE](LICENSE).
