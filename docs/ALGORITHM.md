# Algorithm reference

Notation follows the paper.

| Symbol | Meaning | Code |
| --- | --- | --- |
| `D` | hypervector size | `config.dimensions` |
| `K` | ring size (clients) | `config.n_clients` |
| `N` | samples held by one client | `history.samples_per_client` |
| `R` | communication rounds | `config.rounds` |
| `S` | number of classes | `dataset.n_classes` |
| `ε` | privacy budget | `config.epsilon` |
| `δ₀` | privacy-loss coefficient (`δ = δ₀/(tN)`) | `config.delta0` |
| `t` | global update index `K(r−1)+k` | `privacy.step_index` |
| `ξ` | noise required at step `t` | `required_variance` |
| `Ψ` | noise already in the received model | `cumulative_variance` |
| `Γ` | noise actually injected | `incremental_variance` |
| `Ξ` | black-box cumulative noise | `blackbox_cumulative_variance` |

## Hyperdimensional computing

**Encoding (Eq. 1).** A feature vector is projected through a fixed Gaussian
basis and passed through a cosine: `H = cos(X @ basis)`, `basis ~ N(0,1)` of
shape `(n_features, D)`. Outputs are real-valued in `[−1, 1]`; they are *not*
binarised. This directly determines the sensitivity used by every proof.

**Training (Eq. 2).** A class prototype is the sum of the encoded samples
carrying that label. One pass, no gradients.

**Inference (Eqs. 3–4).** Score the query against each prototype and take the
argmax. The paper writes cosine similarity; the released implementation uses the
raw dot product. These differ whenever prototypes have unequal norms, which they
do here. `--similarity` selects between them; `dot` is the default because it
reproduces the released results.

**Retraining (Eq. 5).** On a misclassification, add the sample to its true
prototype and subtract it from the predicted one. One shuffled pass per round.
Updates are sequential by construction — each correction moves the boundary the
next sample is scored against.

## The accountant

The Gaussian mechanism satisfies `(ε, δ)`-DP when

```
σ² ≥ 2 · (Δg)² · ln(1.25/δ) / ε²
```

**Sensitivity.** Neighbouring datasets differ by one sample, so two prototypes
differ by one hypervector. Cosine outputs are bounded by 1 in absolute value, so
the ℓ2 norm of that difference is at most `√D`, giving `Δg = √D` throughout.

**δ.** Scaled inversely with the number of samples absorbed so far: after `t`
updates the model has seen `tN` samples, so `δ = δ₀/(tN)`.

### Collapsing four theorems into one counter

Because a ring has no aggregation step, the model's state depends only on how
many updates have touched it. With `t = K(r−1) + k`:

```
ξ(t) = (2D/ε²) · ln(1.25 t N / δ₀)          required after t updates

Γ(1) = ξ(1)                                  Theorem 2
Γ(t) = (2D/ε²) · ln(t/(t−1))    for t ≥ 2    Theorems 3, 4, 5
```

Theorem 3 is the `r = 1` slice, Theorem 4 the `k = 1` slice, Theorem 5 the
general case — the paper itself notes 3 and 4 are special cases of 5. Since the
injected noises are independent, variances add and the logarithms telescope:

```
Σ_{j=1..t} Γ(j) = ξ(t)          exactly (Eq. 28)
```

The accountant is therefore tight. `tests/test_privacy.py` asserts the
telescoping to nine significant figures and checks each theorem against its
closed form independently.

### Why tracking matters

A black-box client cannot see `Ψ`, so it injects the full `ξ` every step:

```
Ξ(t) = (2D/ε²) · [ t·ln(1.25N/δ₀) + ln(t!) ]
```

The `ln(t!)` term is the whole story. At `K=100`, `R=30`, `N=73`, `D=2000`,
`ε=0.5`:

| Updates `t` | Tracked `ξ(t)` | Black-box `Ξ(t)` | Ratio |
| --- | --- | --- | --- |
| 1 | 182,742 | 182,742 | 1× |
| 10 | 219,583 | 2,069,088 | 9× |
| 300 | 274,002 | 77,461,013 | 283× |
| 3,000 | 310,844 | 884,609,595 | **2,846×** |

`ln(t!)` is computed with `math.lgamma`, so this stays finite well past the
point where `t!` would overflow.

## One pass around the ring

```
round r = 1:
    for k = 1..K:
        local  <- sum of client k's encoded samples, per class
        noisy  <- local + N(0, Γ(k))
        model  <- model + noisy            # accumulate, do not average

round r >= 2:
    for k = 1..K:
        model <- retrain(model, client k's data)     # Eq. 5
        model <- model + N(0, Γ(K(r-1)+k))
        pass model to client k+1
```

Implemented in `decentralized.run_privatedfl`. Client data is encoded once up
front: the basis is fixed, so a client's hypervectors never change between
rounds and re-encoding them every round would be pure waste.

## Threat model

Every model that leaves a client is already perturbed, which covers the attacks
in Fig. 1b:

| Attack | Why it fails |
| --- | --- |
| Model inversion | intercepted prototypes carry `ξ(t)` noise |
| Membership inference | `(ε, δ)`-DP bounds the distinguishing advantage |
| Curious neighbour | receives the same perturbed model as anyone else |
| No central server | there is no aggregation point to compromise |

Out of scope, and named as future work in the paper: data poisoning, backdoor
insertion, collusion between clients, asynchronous or dynamic topologies, and
heterogeneous per-client budgets.

## Complexity

Per client per round, with `N` samples, `S` classes, `D` dimensions:

| Stage | Cost |
| --- | --- |
| Encoding (once, amortised) | `O(N · n_features · D)` |
| Prototype construction | `O(N · D)` |
| Retraining | `O(N · S · D)` |
| Perturbation | `O(S · D)` |
| Communication (per hop) | `O(S · D)` |

Everything is linear in `D`. Note that the injected noise variance is *also*
linear in `D`, so raising `D` buys capacity and noise in equal measure — which
is why the paper's accuracy-versus-`D` curves saturate rather than climbing.
