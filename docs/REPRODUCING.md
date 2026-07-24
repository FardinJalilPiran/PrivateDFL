# Reproducing the paper

## What runs out of the box

The UCI-HAR experiments are fully reproducible: the preprocessed data ships in
`Dataset/`, so

```bash
privatedfl --config configs/ucihar_noniid.yaml
```

runs the paper's configuration — 100 clients, 30 rounds, `D = 2000`, `ε = 0.5`,
`δ₀ = 10⁻³`, non-IID with 2 classes per client.

Everything about the accountant is exactly reproducible because it depends only
on `(D, ε, δ₀, N, K, R)` and not on the data at all:

```bash
python scripts/plot_noise_accounting.py     # Section 4.2
pytest tests/test_privacy.py                # Theorems 2-5 against closed forms
```

MNIST and ISOLET are not included — convert them to `.choir_dat` and drop them
into `Dataset/`. See `Dataset/README.md`.

## Noise calibration under uneven shards

`delta = delta_0 / N`, so a larger `N` demands *more* noise. When shards are
uneven the accountant calibrates against the **largest** client, since that is
the participant whose privacy is hardest to guarantee. Calibrating on the
smallest shard would quietly under-protect everyone holding more data than the
minimum. With `--unbalanced-clients` on UCI-HAR this means `N = 72` rather than
`36`; it affects only the opening perturbation, because the Theorem 5 increment
does not depend on `N` at all.

## Read this before quoting a number

**Final accuracy varies by roughly ±3 points across random seeds.** Measured on
UCI-HAR, non-IID, `K=100`, `R=30`, `D=2000`, `ε=0.5`:

| Split | Seeds 0–5 | Mean | SD | Range |
| --- | --- | --- | --- | --- |
| Balanced (default) | 85.1, 84.5, 90.1, 90.6, 92.0, 83.6 | 87.6% | 3.3 | 83.6 – 92.0 |
| Released (unbalanced) | 87.1, 88.8, 89.2, 85.6, 90.7, 82.1 | 87.3% | 2.8 | 82.1 – 90.7 |

The released notebook reports **91.38%**, which sits at the top of that spread
rather than at its centre. This is not a bug in either implementation — it is
what a single seed looks like when the round-1 perturbation is large relative to
the signal.

Accuracy also oscillates from round to round *within* a single run, because
fresh noise is drawn at every step. In one verbatim replay of the released
notebook the model reads 91.11% at round 20 and 87.61% at round 30 — a 3.5 point
swing with no change in configuration. Quoting the final round alone is
therefore fragile in either direction. Prefer the mean over the last several
rounds, or report a seed sweep; `scripts/seed_variance.py` does the latter.

The mechanism is worth understanding. At the default settings, step 1 injects a
noise variance of about 183,000 (std ≈ 427) into prototypes whose components are
of order 10²–10³. The model starts essentially buried, and rounds 2–30 dig it
back out through retraining while the incremental noise falls to ~5 per step.
Where a run lands depends on how favourable that initial draw was.

**Recommendation:** report mean ± sd over at least 5 seeds.

```bash
python scripts/seed_variance.py --seeds 10
```

## Two deviations from the released notebook, both optional

### 1. Client shard sizes

The released non-IID split draws from each class pool *without replacement*. With
100 clients sharing 6 classes, the pools run dry: 89 clients receive 72 samples,
9 receive 36, and 6,847 of the 7,352 training samples are used. On some seeds a
client is starved entirely, which crashes the original code.

This matters for the accounting. The derivation of `δ = δ₀/(tN)` assumes a common
`N`; with uneven shards, `N` is only well defined for the first client, and it
enters the formulas at `t = 1` only. The guarantee is therefore looser than
stated, though not by much.

The default here is `balance_clients: true`, which gives every client the same
`N` by reusing samples from over-subscribed classes. To reproduce the released
behaviour exactly:

```bash
privatedfl --config configs/ucihar_noniid.yaml --unbalanced-clients
```

Both are supported and both are tested; as the table above shows, the choice is
worth well under one standard deviation of seed noise. When a client would be
starved, this implementation gives it reused samples and logs a warning rather
than crashing.

### 2. Similarity measure

Eq. (3) of the paper specifies cosine similarity. The released implementation
scores with a raw dot product, `(H @ chv.T).argmax(dim=1)`. These are not
equivalent: prototypes accumulate different norms depending on how many samples
and corrections each class received, and the dot product rewards large norms.

`--similarity dot` (the default) reproduces the released numbers;
`--similarity cosine` follows the paper literally. Worth checking which one you
intend before the next submission.

## Expected behaviour

- **Round 1 is near chance.** Theorem 2's one-off perturbation dominates. Round 2
  typically jumps 30–40 points.
- **The curve is not monotone.** Noise is redrawn every step; individual rounds
  fluctuate. Judge the trend, not any single round.
- **Larger `D` saturates.** Capacity grows linearly in `D` and so does the noise
  variance. The paper's Fig. 9 shows `D = 5000` matching `D = 3000`.
- **Small `D` actively degrades.** At `D = 100` the noise dominates and accuracy
  *falls* across rounds. Reproducible here with `--dimensions 100 --rounds 200`.
- **`δ₀` barely matters.** Fifteen orders of magnitude cost under 4× variance.

## Determinism

Runs are reproducible for a fixed `--seed` on the same device. The seed fixes the
partition, the encoding basis, the retraining order and every noise draw. Noise
is sampled on the generator's device and moved, so a seeded run gives identical
noise on CPU and GPU; CPU/GPU results can still differ marginally through
floating-point reduction order in the matmuls.
