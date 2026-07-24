"""Hyperdimensional computing primitives for PrivateDFL.

The encoder is deliberately unchanged from the released implementation and from
Eq. (1) of the paper::

    basis ~ N(0, 1)  of shape (n_features, D)
    h_d   = cos(F . B_d)      ->      H = cos(X @ basis)

Note that the hypervectors are **real-valued in [-1, 1]**, not binarised. That
matters for the privacy analysis: the sensitivity bound ``||.|| = sqrt(D)`` in
Proofs 1-4 comes precisely from cosine outputs being bounded by 1 in absolute
value, so binarising or rescaling here would silently invalidate the noise
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Encoder", "HDModel", "SIMILARITIES"]

SIMILARITIES = ("dot", "cosine")


class Encoder:
    """Random cosine projection from feature space to hyperspace.

    Parameters
    ----------
    n_features:
        Input dimensionality.
    dimensions:
        Hypervector size ``D``.
    device, dtype:
        Where the basis and the encoded hypervectors live.
    seed:
        Fixes the basis so runs are reproducible.
    """

    def __init__(
        self,
        n_features: int,
        dimensions: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
        self.n_features = n_features
        self.dimensions = dimensions
        self.device = torch.device(device)
        self.dtype = dtype

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        # Drawn on the host then moved, so the basis is identical on CPU and GPU.
        self.basis = torch.randn(
            (n_features, dimensions), generator=generator, dtype=torch.float32
        ).to(device=self.device, dtype=self.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(n_samples, n_features)`` to ``(n_samples, D)``."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            return torch.cos(x @ self.basis)

    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        """Encode a large tensor in chunks, to bound peak memory."""
        if len(x) <= batch_size:
            return self(x)
        return torch.cat([self(x[i : i + batch_size]) for i in range(0, len(x), batch_size)], dim=0)


@dataclass
class HDModel:
    """A bundle of class hypervectors passed around the ring."""

    class_hypervectors: torch.Tensor  # (n_classes, D)

    @classmethod
    def zeros(
        cls,
        n_classes: int,
        dimensions: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> HDModel:
        return cls(torch.zeros((n_classes, dimensions), device=device, dtype=dtype))

    @property
    def n_classes(self) -> int:
        return self.class_hypervectors.shape[0]

    @property
    def dimensions(self) -> int:
        return self.class_hypervectors.shape[1]

    def clone(self) -> HDModel:
        return HDModel(self.class_hypervectors.clone())

    # ------------------------------------------------------------------ train

    @classmethod
    def from_samples(
        cls,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        n_classes: int,
    ) -> HDModel:
        """Build class prototypes by summing hypervectors per label (Eq. 2)."""
        model = cls.zeros(n_classes, encoded.shape[1], device=encoded.device, dtype=encoded.dtype)
        model.class_hypervectors.index_add_(0, labels.to(encoded.device).long(), encoded)
        return model

    def retrain(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 1,
        similarity: str = "dot",
        generator: torch.Generator | None = None,
    ) -> HDModel:
        """Error-driven prototype correction (Eq. 5).

        A misclassified sample is added to its true class prototype and
        subtracted from the predicted one. The released implementation makes a
        single shuffled pass, so ``n_epochs`` defaults to 1.

        The updates are inherently sequential: correcting one sample moves the
        boundary the next sample is scored against.
        """
        device = self.class_hypervectors.device
        encoded = encoded.to(device, dtype=self.class_hypervectors.dtype)
        labels = labels.to(device).long()
        n_samples = len(encoded)

        for _ in range(n_epochs):
            sample_device = generator.device if generator is not None else device
            # Indices are pulled to the host: the loop touches one row at a
            # time, and a device-resident index would transfer on every access.
            order = torch.randperm(n_samples, generator=generator, device=sample_device).tolist()
            mistakes = 0
            for index in order:
                sample = encoded[index]
                prediction = self._score(sample.unsqueeze(0), similarity).argmax()
                target = labels[index]
                if prediction != target:
                    self.class_hypervectors[target] += sample
                    self.class_hypervectors[prediction] -= sample
                    mistakes += 1
            if mistakes == 0:
                break
        return self

    # ---------------------------------------------------------------- predict

    def _score(self, encoded: torch.Tensor, similarity: str) -> torch.Tensor:
        prototypes = self.class_hypervectors
        if similarity == "cosine":
            norms = prototypes.norm(dim=1).clamp_min(1e-12)
            return (encoded @ prototypes.T) / norms
        if similarity == "dot":
            return encoded @ prototypes.T
        raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")

    def predict(
        self, encoded: torch.Tensor, similarity: str = "dot", batch_size: int = 8192
    ) -> torch.Tensor:
        """Return predicted class indices (Eq. 4)."""
        outputs = [
            self._score(
                encoded[i : i + batch_size].to(
                    self.class_hypervectors.device, dtype=self.class_hypervectors.dtype
                ),
                similarity,
            ).argmax(dim=1)
            for i in range(0, len(encoded), batch_size)
        ]
        if not outputs:
            return torch.empty(0, dtype=torch.long, device=self.class_hypervectors.device)
        return torch.cat(outputs)

    def accuracy(
        self, encoded: torch.Tensor, labels: torch.Tensor, similarity: str = "dot"
    ) -> float:
        predictions = self.predict(encoded, similarity)
        return (predictions == labels.to(predictions.device).long()).float().mean().item()
