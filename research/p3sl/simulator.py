"""In-process P3SL simulator and baseline trainers.

The Docker reference implementation requires 6 containers and a TCP
orchestration loop that is hard to script for a research paper sweep.
This module faithfully re-implements the *training math* of P3SL while
running entirely in a single Python process, which lets us:

* loop over many (sigma, alpha, split, seed) configurations cheaply,
* swap the training mode between
    ``"p3sl"`` — personalized split learning + DP + bi-level optimization,
    ``"vanilla_sl"`` — fixed-split, no DP,
    ``"fedavg"``    — Federated Averaging without splitting (model lives on each client),
    ``"central"``   — single-machine training (upper bound on accuracy).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

from .dp import LaplaceMechanism, GaussianMechanism, make_mechanism
from .model import P3SLModel, MAX_SPLIT_LAYER, DEFAULT_SPLIT_POINTS


# ---------------------------------------------------------------------------
# Profile — describes one client's preferences (Eq. 3 of the paper)
# ---------------------------------------------------------------------------
@dataclass
class ClientProfile:
    """Per-client privacy / energy preferences."""

    alpha: float                    # 0 → pure energy, 1 → pure privacy
    energy_table: Dict[int, float]  # split_layer → energy cost (relative)
    leakage_curve: Callable[[int, float], float] = field(repr=False)  # (split, sigma) → estimated leakage
    name: str = "client"


def default_energy_table(split_points: Sequence[int] = DEFAULT_SPLIT_POINTS) -> Dict[int, float]:
    """A monotone-decreasing energy table consistent with deeper splits doing more
    on-device work (FLOPs accumulate as we move the split deeper)."""
    return {sp: 1.0 + 0.10 * (idx + 1) for idx, sp in enumerate(split_points)}


def default_leakage_curve(split_layer: int, sigma: float) -> float:
    """Heuristic FSIM-like leakage: deeper split + more noise → less leakage."""
    base = max(0.05, 0.55 - 0.04 * split_layer)
    return max(0.05, base - 0.02 * sigma)


def select_optimal_split_point(
    profile: ClientProfile,
    noise_table: Dict[int, float],
) -> int:
    """Argmin of ``α·norm(leakage) + (1-α)·norm(energy)`` (P3SL Eq. 3)."""
    splits = list(noise_table.keys())
    leakages = {s: profile.leakage_curve(s, noise_table[s]) for s in splits}
    energies = {s: profile.energy_table[s] for s in splits}

    lmin, lmax = min(leakages.values()), max(leakages.values())
    emin, emax = min(energies.values()), max(energies.values())

    def norm(v, lo, hi):
        return 0.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo)

    best_split, best_score = splits[0], math.inf
    for s in splits:
        score = profile.alpha * norm(leakages[s], lmin, lmax) + (1.0 - profile.alpha) * norm(
            energies[s], emin, emax
        )
        if score < best_score:
            best_split, best_score = s, score
    return best_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class P3SLConfig:
    """Hyper-parameters for one full simulator run."""

    mode: str = "p3sl"          # p3sl | vanilla_sl | fedavg | central
    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"

    # DP
    dp_mechanism: str = "laplace"        # laplace | gaussian | none
    sigma_init: float = 1.0              # initial noise scale
    sensitivity: float = 1.0             # bound used for ε reporting (and clipping if enabled)
    dp_clip: bool = False                # enforce L1/L2 clipping (formal DP, destroys utility on high-dim IRs)
    sigma_decay: float = 0.95            # multiplicative decay per round
    target_accuracy: float = 80.0        # P3SL bi-level early-exit target (%)
    sigma_floor: float = 0.0             # clamp σ from below

    # Splitting
    split_points: Tuple[int, ...] = DEFAULT_SPLIT_POINTS
    fixed_split_layer: int = 6           # used in vanilla_sl / fedavg

    # Data partitioning
    iid: bool = True
    dirichlet_alpha: float = 0.5         # used when iid=False

    # Logging
    log_every: int = 1
    record_per_client: bool = True

    # Optional dataset sub-sampling — reduce train set to this many samples for
    # rapid smoke tests / unit-test parity. ``0`` = use full dataset.
    data_subset: int = 0


# ---------------------------------------------------------------------------
# Helpers — data partitioning
# ---------------------------------------------------------------------------
def partition_indices(
    labels: np.ndarray,
    num_clients: int,
    iid: bool,
    dirichlet_alpha: float,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Either uniform IID shards or a Dirichlet non-IID partition."""
    n = labels.shape[0]
    if iid:
        order = rng.permutation(n)
        return [list(s) for s in np.array_split(order, num_clients)]

    # Non-IID Dirichlet partition
    num_classes = int(labels.max() + 1)
    idx_by_class = [list(np.where(labels == c)[0]) for c in range(num_classes)]
    for lst in idx_by_class:
        rng.shuffle(lst)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
        splits = np.cumsum(proportions) * len(idx_by_class[c])
        prev = 0
        for cid in range(num_clients):
            end = int(splits[cid])
            client_indices[cid].extend(idx_by_class[c][prev:end])
            prev = end
    for lst in client_indices:
        rng.shuffle(lst)
    return client_indices


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class P3SLSimulator:
    """Runs one full training simulation and exposes per-round metrics."""

    def __init__(
        self,
        cfg: P3SLConfig,
        train_dataset: Dataset,
        test_dataset: Dataset,
        profiles: Optional[Sequence[ClientProfile]] = None,
        in_channels: int = 1,
        num_classes: int = 10,
    ) -> None:
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = torch.device(cfg.device)

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.backends.cudnn.deterministic = True

        rng = np.random.default_rng(cfg.seed)

        # Optional sub-sampling for fast smoke tests
        if cfg.data_subset > 0 and cfg.data_subset < len(train_dataset):
            keep = rng.permutation(len(train_dataset))[: cfg.data_subset]
            train_dataset = Subset(train_dataset, keep.tolist())
            self.train_dataset = train_dataset

        # Dataset partitioning
        labels = self._dataset_labels(train_dataset)
        self.client_indices = partition_indices(
            labels, cfg.num_clients, cfg.iid, cfg.dirichlet_alpha, rng
        )

        # Default profiles (alphas spread between 0.1 and 0.9)
        if profiles is None:
            alphas = np.linspace(0.1, 0.9, cfg.num_clients)
            profiles = [
                ClientProfile(
                    alpha=float(a),
                    energy_table=default_energy_table(cfg.split_points),
                    leakage_curve=default_leakage_curve,
                    name=f"client{i}",
                )
                for i, a in enumerate(alphas)
            ]
        if len(profiles) != cfg.num_clients:
            raise ValueError(
                f"Expected {cfg.num_clients} profiles, got {len(profiles)}"
            )
        self.profiles: List[ClientProfile] = list(profiles)

        # Global model
        self.global_model = P3SLModel(in_channels=in_channels, num_classes=num_classes).to(
            self.device
        )

        # Per-client local models (each client owns layers 0..split_layer for P3SL,
        # but for simplicity we store a full copy on each)
        self.client_models: List[P3SLModel] = [
            P3SLModel(in_channels=in_channels, num_classes=num_classes).to(self.device)
            for _ in range(cfg.num_clients)
        ]
        for m in self.client_models:
            m.load_state_dict(self.global_model.state_dict())

        # Per-round state
        self.noise_table: Dict[int, float] = {sp: cfg.sigma_init for sp in cfg.split_points}
        self.history: List[Dict[str, float]] = []

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _dataset_labels(ds: Dataset) -> np.ndarray:
        if isinstance(ds, Subset):
            inner = ds.dataset
            if hasattr(inner, "targets"):
                t = inner.targets  # type: ignore[attr-defined]
                return np.asarray(t)[ds.indices]
        if hasattr(ds, "targets"):
            t = ds.targets  # type: ignore[attr-defined]
            return np.asarray(t)
        labels = []
        for _, y in ds:
            labels.append(int(y))
        return np.asarray(labels)

    def _client_loader(self, cid: int) -> DataLoader:
        sub = Subset(self.train_dataset, self.client_indices[cid])
        g = torch.Generator().manual_seed(self.cfg.seed + cid)
        return DataLoader(sub, batch_size=self.cfg.batch_size, shuffle=True, generator=g)

    def _test_loader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=256, shuffle=False)

    # --------------------------------------------------------------- training
    def train(self) -> List[Dict[str, float]]:
        """Run ``cfg.rounds`` rounds, returning the per-round metric history."""
        if self.cfg.mode == "central":
            return self._train_central()
        for round_idx in range(self.cfg.rounds):
            client_split_layers, client_loss = self._run_round(round_idx)
            acc = self.evaluate_global_accuracy()
            metrics = {
                "round": round_idx + 1,
                "accuracy": acc,
                "mean_loss": float(np.mean(client_loss)) if client_loss else float("nan"),
                "mean_split": float(np.mean(client_split_layers)) if client_split_layers else float("nan"),
                "sigma_mean": float(np.mean(list(self.noise_table.values()))),
            }
            if self.cfg.record_per_client:
                for cid, sl in enumerate(client_split_layers):
                    metrics[f"client{cid}_split"] = float(sl)
            self.history.append(metrics)
            if (round_idx + 1) % self.cfg.log_every == 0:
                print(
                    f"[round {round_idx + 1}/{self.cfg.rounds}] "
                    f"acc={acc:.2f}% loss={metrics['mean_loss']:.4f} "
                    f"σ̄={metrics['sigma_mean']:.3f} mean_split={metrics['mean_split']:.1f}"
                )
            self._update_noise_table(acc)
            if acc >= self.cfg.target_accuracy and self.cfg.mode == "p3sl":
                print(f"[simulator] target accuracy {self.cfg.target_accuracy:.1f}% reached early")
        return self.history

    # ---------------------------------------------------------- one full round
    def _run_round(self, round_idx: int) -> Tuple[List[int], List[float]]:
        per_client_split: List[int] = []
        per_client_loss: List[float] = []
        per_client_state: List[Dict[str, torch.Tensor]] = []

        # Sync all clients with the latest global head before training
        for m in self.client_models:
            m.load_state_dict(self.global_model.state_dict())

        for cid in range(self.cfg.num_clients):
            split_layer = self._choose_split_layer(cid)
            sigma = self.noise_table[split_layer] if self.cfg.mode == "p3sl" else (
                self.cfg.sigma_init if self.cfg.mode == "vanilla_sl" else 0.0
            )
            mech = (
                make_mechanism(
                    self.cfg.dp_mechanism,
                    sigma,
                    sensitivity=self.cfg.sensitivity,
                    clip=self.cfg.dp_clip,
                )
                if sigma > 0
                else None
            )

            loss = self._local_train_one_client(cid, split_layer, mech)
            per_client_split.append(split_layer)
            per_client_loss.append(loss)
            per_client_state.append({k: v.detach().clone() for k, v in self.client_models[cid].state_dict().items()})

        # Aggregate
        if self.cfg.mode in ("p3sl", "vanilla_sl", "fedavg"):
            self._aggregate(per_client_state, per_client_split)

        return per_client_split, per_client_loss

    def _choose_split_layer(self, cid: int) -> int:
        if self.cfg.mode == "p3sl":
            return select_optimal_split_point(self.profiles[cid], self.noise_table)
        if self.cfg.mode == "vanilla_sl":
            return self.cfg.fixed_split_layer
        if self.cfg.mode == "fedavg":
            return self.global_model.num_layers - 1  # whole network on the client
        raise ValueError(self.cfg.mode)

    def _local_train_one_client(
        self,
        cid: int,
        split_layer: int,
        mech,
    ) -> float:
        """One pass of (split-)SGD on the client's local data."""
        cfg = self.cfg
        model = self.client_models[cid]
        model.train()
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        criterion = nn.CrossEntropyLoss()
        loader = self._client_loader(cid)
        running, n = 0.0, 0
        for _ in range(cfg.local_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()

                # ---- Client (head) forward ----
                ir = model.forward_upto(x, split_layer)
                ir_for_server = ir.detach().requires_grad_(True)
                noisy_ir = mech.apply(ir_for_server) if mech is not None else ir_for_server

                # ---- Server (tail) forward + loss + backprop into IR ----
                out = model.forward_from(noisy_ir, split_layer)
                loss = criterion(out, y)
                loss.backward()
                ir_grad = ir_for_server.grad

                # ---- Boundary backprop into client head ----
                ir.backward(ir_grad)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                opt.step()
                running += float(loss.item()) * x.size(0)
                n += x.size(0)
        return running / max(n, 1)

    # -------------------------------------------------------------- aggregate
    def _aggregate(
        self,
        per_client_state: List[Dict[str, torch.Tensor]],
        per_client_split: List[int],
    ) -> None:
        """P3SL aggregation: average client-owned head layers per index."""
        global_sd = self.global_model.state_dict()

        # Determine the union of head indices = max split across clients
        max_split = max(per_client_split) if per_client_split else 0
        head_keys = [
            k for k in global_sd.keys()
            if (mi := self._layer_index(k)) is not None and mi <= max_split
        ]

        for key in head_keys:
            stacked = torch.stack(
                [per_client_state[cid][key].float() for cid in range(self.cfg.num_clients)],
                dim=0,
            )
            global_sd[key] = stacked.mean(dim=0).to(global_sd[key].dtype)

        # Tail layers come from any one client — they are trained identically given identical
        # initialisation, but for safety we average them too.
        tail_keys = [k for k in global_sd.keys() if k not in head_keys]
        for key in tail_keys:
            stacked = torch.stack(
                [per_client_state[cid][key].float() for cid in range(self.cfg.num_clients)],
                dim=0,
            )
            global_sd[key] = stacked.mean(dim=0).to(global_sd[key].dtype)

        self.global_model.load_state_dict(global_sd)

    @staticmethod
    def _layer_index(key: str) -> Optional[int]:
        if not key.startswith("layers."):
            return None
        try:
            return int(key.split(".")[1])
        except (IndexError, ValueError):
            return None

    # --------------------------------------------------------- noise schedule
    def _update_noise_table(self, current_accuracy: float) -> None:
        """Bi-level optimisation: shrink noise when accuracy hasn't reached target."""
        cfg = self.cfg
        if cfg.mode != "p3sl":
            # For non-P3SL modes we just keep sigma constant so reviewers see the
            # raw privacy/utility trade-off without confounding from scheduling.
            return
        if current_accuracy >= cfg.target_accuracy:
            return
        for sp in self.noise_table:
            self.noise_table[sp] = max(cfg.sigma_floor, self.noise_table[sp] * cfg.sigma_decay)

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_global_accuracy(self) -> float:
        self.global_model.eval()
        loader = self._test_loader()
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(self.device); y = y.to(self.device)
            out = self.global_model(x)
            correct += int((out.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))
        return 100.0 * correct / max(total, 1)

    # ------------------------------------------------------ centralized mode
    def _train_central(self) -> List[Dict[str, float]]:
        cfg = self.cfg
        opt = optim.SGD(self.global_model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True)
        for round_idx in range(cfg.rounds):
            self.global_model.train()
            running, n = 0.0, 0
            for x, y in loader:
                x = x.to(self.device); y = y.to(self.device)
                opt.zero_grad()
                out = self.global_model(x)
                loss = criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=cfg.grad_clip)
                opt.step()
                running += float(loss.item()) * x.size(0)
                n += x.size(0)
            acc = self.evaluate_global_accuracy()
            self.history.append(
                {
                    "round": round_idx + 1,
                    "accuracy": acc,
                    "mean_loss": running / max(n, 1),
                    "mean_split": float("nan"),
                    "sigma_mean": 0.0,
                }
            )
            print(f"[central round {round_idx + 1}/{cfg.rounds}] acc={acc:.2f}%")
        return self.history
