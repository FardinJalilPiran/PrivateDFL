# Changelog

## 1.0.2

### Changed
- Default seed is now 55 (was 42), in every config and in `ExperimentConfig`.

### Added
- A test pinning `_l2_normalise` to `sklearn.preprocessing.Normalizer(norm="l2")`,
  the call used in the released notebook. scikit-learn stays an optional
  test-only import; the check skips when it is absent.

## 1.0.1

### Fixed
- The DP accountant calibrated `N` against the **smallest** client shard. Since
  `delta = delta_0 / N`, a larger `N` requires more noise, so this under-noised
  every client holding more data than the minimum. Calibration now uses the
  largest shard, matching the worst-case reasoning used throughout the proofs.

### Added
- `FederatedDataset.max_client_samples()` and a regression test pinning the
  calibration direction.
- A note in `docs/REPRODUCING.md` on round-to-round accuracy oscillation, which
  makes any single final-round figure fragile.

## 1.0.0

First packaged release. Replaces the original single-notebook implementation.

### Added
- Installable `privatedfl` package with a `privatedfl` command-line entry point.
- The preprocessed UCI-HAR split ships in `Dataset/`, so runs need no download.
  MNIST and ISOLET drop in under the same `.choir_dat` convention.
- YAML configurations with command-line overrides (`configs/`).
- Test suite covering the accountant against each theorem's closed form, the
  encoder definition, the `.choir_dat` reader, partitioning and the ring loop.
- Scripts for the noise-accounting comparison, the privacy sweep, and the
  across-seed spread.
- `docs/ALGORITHM.md` and `docs/REPRODUCING.md`.

### Changed
- The four noise theorems are implemented through a single global step counter
  `t = K(r-1) + k`, which is what they reduce to in a ring. Theorem 5 with
  `r = 1` gives Theorem 3, with `k = 1` gives Theorem 4, and `t = 1` gives
  Theorem 2 — as the paper itself notes.
- `.choir_dat` files are read in one pass and reinterpreted rather than unpacked
  value by value, avoiding ~4.1 million `struct.unpack` calls on UCI-HAR.
- Client data is encoded once up front instead of once per round. The basis is
  fixed, so the hypervectors never change between rounds.
- The code runs on CPU as well as GPU; device selection is automatic.
- Non-IID shards are equal-sized by default, which is what the accountant's
  `delta = delta0 / (t N)` assumes. `--unbalanced-clients` restores the released
  split's uneven sizes.

### Fixed
- The released non-IID split consumes class pools without replacement and can
  starve a client of all its data, which crashed the original code on some
  seeds. Such a client now receives reused samples and a warning is logged.
- A seeded CPU generator paired with a CUDA model raised
  `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`.
  Noise is now sampled on the generator's device and moved, which also makes a
  seeded run produce identical noise on CPU and GPU.
- The first `Noise_first_round` definition in the original notebook called a
  non-existent `.device()` method; it was shadowed by a later cell and never
  ran. Only the working formulation survives here.

### Notes
- Final accuracy varies by roughly ±3 points across seeds under a tight budget.
  See `docs/REPRODUCING.md` before quoting a single run.
- Inference uses the dot product by default, matching the released code. Eq. (3)
  of the paper specifies cosine similarity; `--similarity cosine` selects it.
