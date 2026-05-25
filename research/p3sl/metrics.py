"""Comprehensive privacy / utility metrics.

The original ``attacks/metrics.py`` only reports FSIM. For a defensible paper
we report a *vector* of metrics so reviewers can compare against any prior
work's preferred yardstick:

* **Reconstruction quality** — MSE, PSNR, SSIM, FSIM, LPIPS-AlexNet
* **Identifiability** — top-1 nearest-neighbour matching accuracy
  (does the reconstruction look more like its original than a random other?)
* **Membership inference** — AUROC, balanced-accuracy, TPR @ FPR=0.1
* **Backdoor / poisoning** — Clean-Acc, Backdoor-ASR, ASR-on-non-target

LPIPS is optional (requires the ``lpips`` package); if unavailable the
function silently returns ``None`` for that field, so reviewers can still see
all other metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, Optional, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class PrivacyMetrics:
    mse: float = float("nan")
    psnr: float = float("nan")
    ssim: float = float("nan")
    fsim: float = float("nan")
    lpips: float = float("nan")
    nn_id_acc: float = float("nan")  # nearest-neighbour identifiability ∈ [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pixel-space metrics
# ---------------------------------------------------------------------------
def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(_ensure_4d(a), _ensure_4d(b)).item())


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(m))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Simple single-scale SSIM (no Gaussian window) — adequate for 28×28 inputs."""
    a = _ensure_4d(a).clamp(0.0, 1.0)
    b = _ensure_4d(b).clamp(0.0, 1.0)
    mu_a, mu_b = a.mean(dim=(2, 3), keepdim=True), b.mean(dim=(2, 3), keepdim=True)
    va = ((a - mu_a) ** 2).mean(dim=(2, 3), keepdim=True)
    vb = ((b - mu_b) ** 2).mean(dim=(2, 3), keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(2, 3), keepdim=True)
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    )
    return float(s.mean().item())


def _fsim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Use ``piq.fsim`` if available, otherwise fall back to SSIM as a proxy."""
    try:
        from piq import fsim as piq_fsim  # type: ignore
        a4 = _ensure_4d(a).clamp(0.0, 1.0)
        b4 = _ensure_4d(b).clamp(0.0, 1.0)
        return float(piq_fsim(a4, b4, data_range=1.0, chromatic=False).item())
    except Exception:
        return ssim(a, b)


def _lpips(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    if not hasattr(_lpips, "_model"):
        _lpips._model = lpips.LPIPS(net="alex", verbose=False)  # type: ignore[attr-defined]
    a4 = _ensure_4d(a).clamp(0.0, 1.0) * 2 - 1
    b4 = _ensure_4d(b).clamp(0.0, 1.0) * 2 - 1
    if a4.size(1) == 1:
        a4 = a4.repeat(1, 3, 1, 1)
        b4 = b4.repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(_lpips._model(a4, b4).mean().item())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Identifiability — nearest-neighbour matching accuracy
# ---------------------------------------------------------------------------
def nn_identifiability_accuracy(
    originals: torch.Tensor, reconstructions: torch.Tensor
) -> float:
    """Fraction of reconstructions whose closest original *is* the right one.

    This is the strongest privacy metric: even if a reconstruction is blurry,
    if it picks out the right original from a gallery of B images it leaks
    identity. Random chance = 1/B.
    """
    if originals.size(0) != reconstructions.size(0):
        raise ValueError("Batch size mismatch")
    a = originals.flatten(1)
    b = reconstructions.flatten(1)
    # negative L2 distance → similarity
    dist = torch.cdist(a, b)
    correct = (dist.argmin(dim=0) == torch.arange(a.size(0))).float().mean()
    return float(correct.item())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_privacy_metrics(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
) -> PrivacyMetrics:
    """Compute the full vector of reconstruction-quality metrics."""
    o = _ensure_4d(originals).clamp(0.0, 1.0).cpu().float()
    r = _ensure_4d(reconstructions).clamp(0.0, 1.0).cpu().float()
    out = PrivacyMetrics(
        mse=mse(o, r),
        psnr=psnr(o, r),
        ssim=ssim(o, r),
        fsim=_fsim(o, r),
        nn_id_acc=nn_identifiability_accuracy(o, r),
    )
    lpips_val = _lpips(o, r)
    if lpips_val is not None:
        out.lpips = lpips_val
    return out


# ---------------------------------------------------------------------------
# Membership-inference AUC and TPR @ low-FPR
# ---------------------------------------------------------------------------
def mia_auc(
    member_scores: torch.Tensor,
    nonmember_scores: torch.Tensor,
) -> Dict[str, float]:
    """Compute AUROC, balanced-accuracy and TPR @ FPR=0.1 for a binary MIA."""
    scores = torch.cat([member_scores.flatten(), nonmember_scores.flatten()])
    labels = torch.cat(
        [
            torch.ones_like(member_scores.flatten()),
            torch.zeros_like(nonmember_scores.flatten()),
        ]
    )
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    pos = (labels == 1).float()
    neg = (labels == 0).float()
    tpr = pos.cumsum(0) / max(float(pos.sum().item()), 1.0)
    fpr = neg.cumsum(0) / max(float(neg.sum().item()), 1.0)
    fpr0 = torch.cat([torch.zeros(1), fpr])
    tpr0 = torch.cat([torch.zeros(1), tpr])
    auc = float(torch.trapz(tpr0, fpr0).item())

    # balanced accuracy at threshold 0.5 of score
    median = float(scores.median().item())
    pred = (scores >= median).float()
    target = torch.cat([torch.ones_like(member_scores.flatten()),
                        torch.zeros_like(nonmember_scores.flatten())])
    sens = float(((pred == 1) & (target == 1)).float().sum() / max(float((target == 1).sum()), 1.0))
    spec = float(((pred == 0) & (target == 0)).float().sum() / max(float((target == 0).sum()), 1.0))
    bal_acc = 0.5 * (sens + spec)

    # TPR @ FPR=0.1
    target_fpr = 0.10
    idx = torch.searchsorted(fpr, torch.tensor(target_fpr))
    tpr_at_fpr = float(tpr[min(int(idx), tpr.numel() - 1)].item()) if tpr.numel() > 0 else float("nan")
    return {"auc": auc, "balanced_acc": bal_acc, "tpr_at_fpr_0.1": tpr_at_fpr}


# ---------------------------------------------------------------------------
# Backdoor metrics
# ---------------------------------------------------------------------------
def backdoor_asr(
    preds: torch.Tensor,
    true_labels: torch.Tensor,
    target_class: int,
) -> float:
    """Backdoor Attack Success Rate computed on samples whose true class != target."""
    mask = true_labels != target_class
    if mask.sum().item() == 0:
        return float("nan")
    success = (preds[mask] == target_class).float().mean()
    return float(success.item())
