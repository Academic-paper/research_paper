"""Honest-but-curious server-side attacks on the IR.

This is a clean rewrite of the attack code that lived in
``my_implementation/.../attacks/server_attacks.py``.  Improvements:

* All hyper-parameters (epochs, iterations, learning rates) are exposed as
  arguments instead of being hard-coded.
* The decoder and MIA classifier accept arbitrary IR shapes (not just 64-D
  flat vectors) — they pool to a 64-D summary internally so they work for
  any split layer.
* The MIA returns a *binary classification head* whose decision can be
  audited via AUC / TPR-at-FPR (see ``metrics.mia_auc``).
* The optimisation attack uses TV regularisation that scales with image size
  so the same code works for both 28×28 and 32×32 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import P3SLModel


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _flatten_pool(ir: torch.Tensor, target_dim: int = 64) -> torch.Tensor:
    """Pool an arbitrary-shaped IR down to a fixed-size vector for the attack heads."""
    flat = ir.reshape(ir.size(0), -1)
    if flat.size(1) == target_dim:
        return flat
    pooler = nn.AdaptiveAvgPool1d(target_dim).to(flat.device)
    return pooler(flat.unsqueeze(1)).squeeze(1)


class FrozenClientHead(nn.Module):
    """Snapshot of the architecturally-known client head used by the adversary.

    Wraps a :class:`P3SLModel` and a ``split_layer`` into a regular
    :class:`torch.nn.Module` so adversary helpers can call ``.eval()`` /
    ``.parameters()`` / ``.to()`` like on any other model.
    """

    def __init__(self, base_model: nn.Module, split_layer: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.split_layer = split_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.base_model.forward_upto(x, self.split_layer)


# ---------------------------------------------------------------------------
# Attack 1 — feed-forward inversion decoder
# ---------------------------------------------------------------------------
class InversionDecoder(nn.Module):
    """Maps a (pooled) IR back to an image of size ``image_shape``."""

    def __init__(
        self,
        ir_dim: int = 64,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        c, h, w = image_shape
        self.image_shape = image_shape
        self.decode = nn.Sequential(
            nn.Linear(ir_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c * h * w),
            nn.Sigmoid(),
        )

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        v = _flatten_pool(ir, self.decode[0].in_features)
        flat = self.decode(v)
        return flat.view(-1, *self.image_shape)


def train_inversion_decoder(
    decoder: InversionDecoder,
    client_head: nn.Module,
    public_loader: DataLoader,
    epochs: int = 5,
    lr: float = 2e-3,
    device: Optional[torch.device] = None,
) -> None:
    """Pre-train the decoder on a *public* dataset assumed available to the adversary.

    The ``client_head`` should be a snapshot of the (architecturally known)
    front-half of the network — the standard white-box setting.
    """
    device = device or next(client_head.parameters()).device
    decoder.to(device)
    client_head.eval()
    opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for images, _ in public_loader:
            images = images.to(device)
            with torch.no_grad():
                ir = client_head(images)
            opt.zero_grad()
            recon = decoder(ir)
            target_01 = (images * 0.5) + 0.5  # un-normalize to [0, 1]
            loss = criterion(recon, target_01)
            loss.backward()
            opt.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        print(f"   [decoder] epoch {epoch + 1}/{epochs} loss={running / max(n, 1):.4f}")


# ---------------------------------------------------------------------------
# Attack 2 — pure white-box optimization-based inversion
# ---------------------------------------------------------------------------
def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    tv_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    tv_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def optimization_attack(
    target_ir: torch.Tensor,
    client_head: nn.Module,
    image_shape: Tuple[int, int, int] = (1, 28, 28),
    iterations: int = 200,
    lr: float = 0.1,
    tv_weight: float = 5e-3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Recover the input image by gradient-descending image MSE in IR space."""
    device = device or target_ir.device
    client_head = client_head.to(device).eval()
    target_ir = target_ir.to(device).detach()
    dummy = torch.randn((target_ir.size(0), *image_shape), device=device, requires_grad=True)
    opt = optim.Adam([dummy], lr=lr)
    criterion = nn.MSELoss()
    for _ in range(iterations):
        opt.zero_grad()
        out = client_head(dummy)
        loss = criterion(out, target_ir) + tv_weight * total_variation_loss(dummy)
        loss.backward()
        opt.step()
    return torch.clamp(dummy.detach(), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Attack 3 — Membership Inference Attack (binary)
# ---------------------------------------------------------------------------
class MembershipInferenceClassifier(nn.Module):
    def __init__(self, ir_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ir_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.ir_dim = ir_dim

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        v = _flatten_pool(ir, self.ir_dim)
        return self.classifier(v).squeeze(-1)  # logit


def train_mia_classifier(
    mia: MembershipInferenceClassifier,
    client_head: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    epochs: int = 5,
    lr: float = 5e-3,
    device: Optional[torch.device] = None,
) -> None:
    """Train the MIA shadow classifier on labeled member / non-member IRs."""
    device = device or next(client_head.parameters()).device
    mia.to(device)
    client_head.eval()
    opt = optim.Adam(mia.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        running, n = 0.0, 0
        m_iter = iter(member_loader)
        nm_iter = iter(nonmember_loader)
        for _ in range(min(len(member_loader), len(nonmember_loader))):
            try:
                m_imgs, _ = next(m_iter)
                nm_imgs, _ = next(nm_iter)
            except StopIteration:
                break
            m_imgs = m_imgs.to(device)
            nm_imgs = nm_imgs.to(device)
            with torch.no_grad():
                m_ir = client_head(m_imgs)
                nm_ir = client_head(nm_imgs)
            X = torch.cat([m_ir, nm_ir], dim=0)
            y = torch.cat(
                [torch.ones(m_imgs.size(0)), torch.zeros(nm_imgs.size(0))], dim=0
            ).to(device)
            opt.zero_grad()
            logits = mia(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * X.size(0)
            n += X.size(0)
        print(f"   [mia] epoch {epoch + 1}/{epochs} loss={running / max(n, 1):.4f}")


# ---------------------------------------------------------------------------
# Backdoor / model-extraction helpers (client-side malicious behaviour)
# ---------------------------------------------------------------------------
def apply_backdoor(images: torch.Tensor, labels: torch.Tensor, target_class: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a 4×4 corner trigger and flip the label to ``target_class``."""
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    poisoned_images[:, :, :4, :4] = 1.0
    poisoned_labels[:] = target_class
    return poisoned_images, poisoned_labels
