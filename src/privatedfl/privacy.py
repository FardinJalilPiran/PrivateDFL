"""The PrivateDFL noise accountant.

In a ring topology there is no aggregation step, so the state of the shared
model is fully described by how many client updates have touched it. Writing
that global step index as

    t = K (r - 1) + k          (round r, client k, both 1-indexed)

collapses the paper's four theorems into two cases.

Required noise after step ``t``
    Theorems 2-5 all calibrate against the number of samples the model has
    absorbed, which is ``t N`` with ``delta = delta0 / (t N)``::

        xi(t) = (2D / eps^2) * ln(1.25 t N / delta0)

Incremental noise injected at step ``t``
    The client already received ``xi(t-1)`` inside the model, so it only adds
    the difference. The logarithms telescope::

        Gamma(1) = xi(1) = (2D / eps^2) * ln(1.25 N / delta0)
        Gamma(t) = (2D / eps^2) * ln(t / (t - 1))            for t >= 2

    which is Theorem 2 at ``t = 1``, Theorem 3 when ``r = 1``, Theorem 4 when
    ``k = 1`` and Theorem 5 in general.

Because the injected noises are independent, their variances add, and the
cumulative noise after step ``t`` equals ``xi(t)`` exactly (Eq. 28). The
accountant is therefore tight: never more noise than the budget demands, never
less.

Contrast with a black-box scheme, which cannot see the noise already present
and so must inject the full ``xi(t)`` every step::

    Xi(t) = (2D / eps^2) * [ t ln(1.25 N / delta0) + ln(t!) ]

The ``ln(t!)`` term is why untracked DP-DFL collapses over long runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "NoiseAccountant",
    "blackbox_cumulative_variance",
    "cumulative_variance",
    "incremental_variance",
    "required_variance",
    "step_index",
]

_GAUSSIAN_CONST = 1.25


def step_index(round_index: int, client_index: int, n_clients: int) -> int:
    """Map ``(r, k)`` to the global update counter ``t = K(r-1) + k``."""
    if round_index < 1:
        raise ValueError(f"round_index is 1-indexed, got {round_index}")
    if client_index < 1:
        raise ValueError(f"client_index is 1-indexed, got {client_index}")
    if client_index > n_clients:
        raise ValueError(f"client_index {client_index} exceeds n_clients {n_clients}")
    return n_clients * (round_index - 1) + client_index


def _check(dimensions: int, epsilon: float, delta0: float, samples_per_client: int) -> None:
    if dimensions <= 0:
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not 0 < delta0 < 1:
        raise ValueError(f"delta0 must lie in (0, 1), got {delta0}")
    if samples_per_client < 1:
        raise ValueError(f"samples_per_client must be >= 1, got {samples_per_client}")


def required_variance(
    dimensions: int,
    epsilon: float,
    delta0: float,
    samples_per_client: int,
    step: int,
) -> float:
    """Noise variance the privacy budget demands once ``step`` updates have run."""
    _check(dimensions, epsilon, delta0, samples_per_client)
    if step < 1:
        raise ValueError(f"step is 1-indexed, got {step}")
    coefficient = (2.0 * dimensions) / (epsilon**2)
    return coefficient * math.log(_GAUSSIAN_CONST * step * samples_per_client / delta0)


def cumulative_variance(
    dimensions: int,
    epsilon: float,
    delta0: float,
    samples_per_client: int,
    step: int,
) -> float:
    """Noise variance actually present in the model after ``step`` updates.

    Equal to :func:`required_variance` by construction (Eq. 28); exposed
    separately so the two can be asserted against each other in tests.
    """
    if step < 1:
        return 0.0
    return required_variance(dimensions, epsilon, delta0, samples_per_client, step)


def incremental_variance(
    dimensions: int,
    epsilon: float,
    delta0: float,
    samples_per_client: int,
    step: int,
) -> float:
    """Variance the client at ``step`` actually injects."""
    _check(dimensions, epsilon, delta0, samples_per_client)
    if step < 1:
        raise ValueError(f"step is 1-indexed, got {step}")
    coefficient = (2.0 * dimensions) / (epsilon**2)
    if step == 1:
        return coefficient * math.log(_GAUSSIAN_CONST * samples_per_client / delta0)
    return coefficient * math.log(step / (step - 1))


def blackbox_cumulative_variance(
    dimensions: int,
    epsilon: float,
    delta0: float,
    samples_per_client: int,
    step: int,
) -> float:
    """Cumulative variance without an accountant, for comparison (Eq. 33).

    Every client injects the full requirement, so the variances sum to
    ``(2D/eps^2) [ t ln(1.25 N / delta0) + ln(t!) ]``. ``lgamma`` keeps the
    factorial finite for large ``t``.
    """
    _check(dimensions, epsilon, delta0, samples_per_client)
    if step < 1:
        raise ValueError(f"step is 1-indexed, got {step}")
    coefficient = (2.0 * dimensions) / (epsilon**2)
    base = step * math.log(_GAUSSIAN_CONST * samples_per_client / delta0)
    return coefficient * (base + math.lgamma(step + 1))


@dataclass
class NoiseAccountant:
    """Tracks cumulative noise and perturbs class hypervectors in place.

    Parameters
    ----------
    dimensions:
        Hypervector size ``D``.
    epsilon:
        Privacy budget. Smaller means more privacy and more noise.
    delta0:
        Privacy-loss coefficient; the effective ``delta`` is ``delta0 / (t N)``.
    samples_per_client:
        ``N``, the number of samples one client holds.
    n_clients:
        ``K``, the number of clients in the ring.
    enabled:
        When ``False`` the accountant is a no-op, giving the non-private
        accuracy ceiling.
    """

    dimensions: int
    epsilon: float
    delta0: float
    samples_per_client: int
    n_clients: int
    enabled: bool = True

    def report(self, round_index: int, client_index: int) -> dict[str, float]:
        """Variance bookkeeping for one client update."""
        step = step_index(round_index, client_index, self.n_clients)
        if not self.enabled:
            return {"step": step, "required": 0.0, "added": 0.0, "blackbox": 0.0}
        args = (self.dimensions, self.epsilon, self.delta0, self.samples_per_client)
        return {
            "step": step,
            "required": required_variance(*args, step),
            "added": incremental_variance(*args, step),
            "blackbox": blackbox_cumulative_variance(*args, step),
        }

    def perturb(
        self,
        class_hypervectors: torch.Tensor,
        round_index: int,
        client_index: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``class_hypervectors`` plus this step's incremental noise."""
        if not self.enabled:
            return class_hypervectors

        step = step_index(round_index, client_index, self.n_clients)
        variance = incremental_variance(
            self.dimensions, self.epsilon, self.delta0, self.samples_per_client, step
        )
        if variance <= 0.0:
            return class_hypervectors

        target_device = class_hypervectors.device
        # A generator must live on the same device as the tensor being filled.
        # Sampling on the generator's own device and moving the result keeps a
        # seeded run identical on CPU and GPU.
        sample_device = generator.device if generator is not None else target_device
        noise = torch.randn(
            class_hypervectors.shape,
            device=sample_device,
            dtype=class_hypervectors.dtype,
            generator=generator,
        )
        return class_hypervectors + math.sqrt(variance) * noise.to(target_device)
