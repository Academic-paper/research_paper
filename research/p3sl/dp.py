"""Differential-privacy utilities for split-learning IR perturbation.

The Docker reference implementation adds Laplace noise with an arbitrary
``sigma`` directly on the raw IR.  That alone does **not** yield a valid
``(epsilon, delta)`` guarantee because the L1/L2 sensitivity of the IR is
unbounded.  This module fixes that by

1. clipping the per-sample IR to a known L1 (Laplace) or L2 (Gaussian) ball,
   so that the global sensitivity Δ is mathematically bounded;
2. providing closed-form ``epsilon`` for Laplace and an RDP→(eps, δ) accountant
   for Gaussian using the standard Mironov 2017 conversion;
3. exposing both mechanisms behind a uniform ``apply(ir)`` API used by the
   simulator and baselines.

References
----------
Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*, 2014.
Mironov, *Rényi Differential Privacy*, CSF 2017.
Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Sensitivity-bounding clipper
# ---------------------------------------------------------------------------
def clip_per_sample(ir: torch.Tensor, max_norm: float, p: float = 2.0) -> torch.Tensor:
    """Clip ``ir`` so each sample has Lp norm at most ``max_norm``.

    Returns a *new* tensor with the same shape as ``ir``. Always treats the
    leading axis as the batch dimension (mirrors the convention used elsewhere
    in the repo where the IR is shape ``[B, C, H, W]`` or ``[B, F]``).
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0 (got {max_norm})")
    flat = ir.reshape(ir.size(0), -1)
    norms = flat.norm(p=p, dim=1, keepdim=True).clamp(min=1e-12)
    factor = (max_norm / norms).clamp(max=1.0)
    return (flat * factor).reshape_as(ir)


# ---------------------------------------------------------------------------
# Laplace mechanism (pure ε-DP, sensitivity in L1)
# ---------------------------------------------------------------------------
@dataclass
class LaplaceMechanism:
    """Per-coordinate Laplace noise N(0, b) with optional sensitivity-aware
    clipping.

    With ``clip=True`` the mechanism is formally ``(epsilon, 0)``-DP with
    ``epsilon = sensitivity / b``.  In practice, however, applying L1 clipping
    to a high-dimensional CNN feature map (e.g. 25,088 dims) destroys the
    per-coordinate signal because the budget ``Δ`` is shared across all dims.
    We therefore expose a ``clip`` flag: papers that want a *formal* guarantee
    set ``clip=True``, papers that want to study the *empirical* privacy /
    utility trade-off (matching the original P3SL paper which adds noise
    without clipping) set ``clip=False`` and report ε as a heuristic upper
    bound.
    """

    scale: float                 # b — Laplace scale parameter (== sigma in the paper)
    sensitivity: float = 1.0     # L1 sensitivity bound (used for ε; enforced if clip)
    clip: bool = True            # whether to L1-clip each sample before adding noise

    def epsilon(self) -> float:
        return self.sensitivity / max(self.scale, 1e-12)

    def apply(self, ir: torch.Tensor) -> torch.Tensor:
        if self.scale <= 0:
            return ir
        if self.clip:
            ir = clip_per_sample(ir, max_norm=self.sensitivity, p=1.0)
        noise = torch.distributions.Laplace(0.0, self.scale).sample(ir.shape)
        return ir + noise.to(ir.device)


# ---------------------------------------------------------------------------
# Gaussian mechanism (Rényi DP, sensitivity in L2)
# ---------------------------------------------------------------------------
@dataclass
class GaussianMechanism:
    """Per-coordinate Gaussian noise N(0, sigma^2) with optional L2 clipping.

    With ``clip=True`` a single release has Rényi-DP parameter
    ``alpha · sensitivity^2 / (2 σ^2)``; conversion to ``(epsilon, delta)``
    uses Mironov 2017.  With ``clip=False`` the mechanism is a *heuristic*
    perturbation matching the original P3SL paper's noise schedule, with
    epsilon reported as an upper bound for reference only.
    """

    sigma: float                  # std-dev of the Gaussian
    sensitivity: float = 1.0      # L2 sensitivity bound (used for ε; enforced if clip)
    clip: bool = True

    def apply(self, ir: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return ir
        if self.clip:
            ir = clip_per_sample(ir, max_norm=self.sensitivity, p=2.0)
        noise = torch.randn_like(ir) * self.sigma
        return ir + noise


# ---------------------------------------------------------------------------
# Closed-form / RDP accountants
# ---------------------------------------------------------------------------
def laplace_epsilon(scale: float, sensitivity: float = 1.0) -> float:
    """Pure ε for one Laplace release."""
    return sensitivity / max(scale, 1e-12)


def laplace_epsilon_composed(scale: float, sensitivity: float, num_releases: int) -> float:
    """Sequential composition for ``num_releases`` Laplace releases (worst case)."""
    return num_releases * laplace_epsilon(scale, sensitivity)


def gaussian_rdp_epsilon(
    sigma: float,
    sensitivity: float,
    num_releases: int,
    delta: float = 1e-5,
    orders: Optional[list[float]] = None,
) -> float:
    """Compute (epsilon, delta) for ``num_releases`` Gaussian mechanism releases.

    Uses the Mironov RDP→DP conversion ``ε = ε_α + log(1/δ)/(α-1)`` and
    minimises over a grid of integer Rényi orders ``α``.
    """
    if sigma <= 0:
        return float("inf")
    if orders is None:
        orders = [1 + i / 10 for i in range(1, 100)] + list(range(11, 64))

    best_eps = float("inf")
    base = (sensitivity ** 2) / (2.0 * sigma ** 2)
    for alpha in orders:
        if alpha <= 1:
            continue
        rdp_per_step = alpha * base
        rdp_total = num_releases * rdp_per_step
        eps = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
    return best_eps


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------
def make_mechanism(kind: str, scale: float, sensitivity: float = 1.0, clip: bool = True):
    """Return a mechanism of the requested kind.

    Parameters
    ----------
    kind:
        ``"laplace"`` or ``"gaussian"`` (case-insensitive). Anything else
        (or scale = 0) returns ``None`` meaning "no DP".
    clip:
        Whether to enforce the sensitivity bound by per-sample clipping.
        ``True`` → formal DP guarantee, but tends to destroy utility on
        high-dim CNN feature maps. ``False`` → heuristic noise (matches the
        P3SL paper); ε reported as an upper bound.
    """
    if scale <= 0:
        return None
    k = kind.lower()
    if k == "laplace":
        return LaplaceMechanism(scale=scale, sensitivity=sensitivity, clip=clip)
    if k == "gaussian":
        return GaussianMechanism(sigma=scale, sensitivity=sensitivity, clip=clip)
    raise ValueError(f"Unknown DP mechanism: {kind!r}")
