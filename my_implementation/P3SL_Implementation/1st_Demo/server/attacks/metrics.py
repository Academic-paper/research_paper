import math

import torch
import torch.nn.functional as F


def _ensure_4d(x):
    return x if x.ndim == 4 else x.unsqueeze(0)


def calculate_mse(original, reconstructed):
    a = _ensure_4d(original).clamp(0.0, 1.0)
    b = _ensure_4d(reconstructed).clamp(0.0, 1.0)
    return float(F.mse_loss(a, b).item())


def calculate_psnr(original, reconstructed, max_val=1.0):
    mse_val = calculate_mse(original, reconstructed)
    if mse_val <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse_val))


def calculate_ssim(original, reconstructed):
    a = _ensure_4d(original).clamp(0.0, 1.0)
    b = _ensure_4d(reconstructed).clamp(0.0, 1.0)
    mu_a = a.mean(dim=(2, 3), keepdim=True)
    mu_b = b.mean(dim=(2, 3), keepdim=True)
    va = ((a - mu_a) ** 2).mean(dim=(2, 3), keepdim=True)
    vb = ((b - mu_b) ** 2).mean(dim=(2, 3), keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(2, 3), keepdim=True)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    )
    return float(s.mean().item())


def calculate_fsim(original, reconstructed):
    """
    Grades the reconstruction. 1.0 = High Leakage, 0.0 = Perfect Privacy.
    """
    try:
        from piq import fsim
        orig = _ensure_4d(original).clamp(0.0, 1.0)
        recon = _ensure_4d(reconstructed).clamp(0.0, 1.0)
        return float(fsim(orig, recon, data_range=1.0, chromatic=False).item())
    except Exception:
        return calculate_ssim(original, reconstructed)


def calculate_nn_identifiability(original, reconstructed):
    a = _ensure_4d(original).flatten(1).cpu()
    b = _ensure_4d(reconstructed).flatten(1).cpu()
    if a.size(0) != b.size(0):
        raise ValueError("Batch size mismatch between original and reconstruction")
    dist = torch.cdist(a, b)
    return float((dist.argmin(dim=0) == torch.arange(a.size(0))).float().mean().item())


def calculate_all(original, reconstructed):
    return {
        "mse": calculate_mse(original, reconstructed),
        "psnr": calculate_psnr(original, reconstructed),
        "ssim": calculate_ssim(original, reconstructed),
        "fsim": calculate_fsim(original, reconstructed),
        "nn_id_acc": calculate_nn_identifiability(original, reconstructed),
    }
