"""Single-command experiment runner that produces every figure / table the
paper needs.

Examples
--------
Smoke test (≈ 1 min on a laptop CPU)::

    python research/run_experiments.py --config research/configs/smoke.yaml

Full benchmark (≈ 30 min on a CPU, < 5 min on a GPU)::

    python research/run_experiments.py --config research/configs/main.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Allow ``python research/run_experiments.py`` from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3sl.simulator import P3SLConfig, P3SLSimulator, ClientProfile, default_energy_table, default_leakage_curve
from p3sl.attacks import (
    FrozenClientHead,
    InversionDecoder,
    MembershipInferenceClassifier,
    apply_backdoor,
    optimization_attack,
    train_inversion_decoder,
    train_mia_classifier,
)
from p3sl.metrics import compute_privacy_metrics, mia_auc, backdoor_asr
from p3sl.dp import laplace_epsilon_composed, gaussian_rdp_epsilon
from p3sl.model import DEFAULT_SPLIT_POINTS, P3SLModel


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("P3SL_DATA_ROOT", str(Path(__file__).parent / "data"))


def load_dataset(name: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if name in ("mnist", "fashionmnist") else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if name == "mnist":
        train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
        in_ch, n_cls, image_shape = 1, 10, (1, 28, 28)
    elif name == "fashionmnist":
        train = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
        in_ch, n_cls, image_shape = 1, 10, (1, 28, 28)
    elif name == "cifar10":
        train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        in_ch, n_cls, image_shape = 3, 10, (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, in_ch, n_cls, image_shape


# ---------------------------------------------------------------------------
# Per-config training + evaluation
# ---------------------------------------------------------------------------
def make_config_from_dict(d: Dict[str, Any]) -> P3SLConfig:
    """Build a :class:`P3SLConfig` from the YAML dict, preserving defaults."""
    fields = {f for f in P3SLConfig.__dataclass_fields__.keys()}
    payload = {k: v for k, v in d.items() if k in fields}
    if "split_points" in payload and isinstance(payload["split_points"], list):
        payload["split_points"] = tuple(payload["split_points"])
    return P3SLConfig(**payload)


def make_default_profiles(cfg: P3SLConfig, alphas: Sequence[float]) -> List[ClientProfile]:
    if len(alphas) != cfg.num_clients:
        raise ValueError(
            f"Need {cfg.num_clients} alphas, got {len(alphas)}"
        )
    energy_table = default_energy_table(cfg.split_points)
    return [
        ClientProfile(
            alpha=float(a),
            energy_table=energy_table,
            leakage_curve=default_leakage_curve,
            name=f"client{i}",
        )
        for i, a in enumerate(alphas)
    ]


def run_single(
    name: str,
    cfg: P3SLConfig,
    dataset_name: str,
    alphas: Sequence[float],
    out_dir: Path,
    *,
    image_shape=(1, 28, 28),
    in_channels: int = 1,
    num_classes: int = 10,
    run_attacks: bool = True,
    attack_epochs: int = 5,
    opt_attack_iters: int = 200,
) -> Dict[str, Any]:
    """Run one full experiment and persist its CSV + summary JSON."""
    print(f"\n=================  {name}  =================")
    train_ds, test_ds, in_ch, n_cls, img_shape = load_dataset(dataset_name)
    profiles = make_default_profiles(cfg, alphas)

    sim = P3SLSimulator(
        cfg=cfg,
        train_dataset=train_ds,
        test_dataset=test_ds,
        profiles=profiles,
        in_channels=in_ch,
        num_classes=n_cls,
    )
    t0 = time.time()
    history = sim.train()
    train_secs = time.time() - t0

    summary: Dict[str, Any] = {
        "name": name,
        "dataset": dataset_name,
        "config": asdict(cfg),
        "alphas": list(alphas),
        "final_accuracy": history[-1]["accuracy"] if history else float("nan"),
        "rounds_run": len(history),
        "train_seconds": train_secs,
    }

    # ---- DP accounting ----
    if cfg.dp_mechanism.lower() == "laplace" and cfg.sigma_init > 0:
        eps = laplace_epsilon_composed(
            scale=cfg.sigma_init,
            sensitivity=cfg.sensitivity,
            num_releases=cfg.rounds,
        )
        summary["epsilon_total"] = eps
        summary["delta"] = 0.0
    elif cfg.dp_mechanism.lower() == "gaussian" and cfg.sigma_init > 0:
        eps = gaussian_rdp_epsilon(
            sigma=cfg.sigma_init,
            sensitivity=cfg.sensitivity,
            num_releases=cfg.rounds,
            delta=1e-5,
        )
        summary["epsilon_total"] = eps
        summary["delta"] = 1e-5
    else:
        summary["epsilon_total"] = float("inf")
        summary["delta"] = 0.0

    # ---- Attacks (run on the trained model + a held-out probe batch) ----
    if run_attacks and cfg.mode != "central":
        summary.update(_run_attacks(
            sim=sim,
            train_ds=train_ds,
            test_ds=test_ds,
            in_ch=in_ch,
            image_shape=img_shape,
            num_classes=n_cls,
            attack_epochs=attack_epochs,
            opt_attack_iters=opt_attack_iters,
        ))

    # ---- Persist ----
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "history.csv", "w", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            for row in history:
                writer.writerow(row)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(
        f"[done] {name}: acc={summary['final_accuracy']:.2f}% ε={summary['epsilon_total']:.3f} "
        f"({train_secs:.1f}s)"
    )
    return summary


# ---------------------------------------------------------------------------
# Attack pipeline
# ---------------------------------------------------------------------------
def _run_attacks(
    sim: P3SLSimulator,
    train_ds,
    test_ds,
    in_ch: int,
    image_shape,
    num_classes: int,
    attack_epochs: int,
    opt_attack_iters: int,
) -> Dict[str, Any]:
    print("[attacks] training adversary models on a public split…")

    # Use a *disjoint* half of train_ds to mimic the adversary's "public" data
    n = len(train_ds)
    pub_idx = list(range(n // 2))
    priv_idx = list(range(n // 2, n))
    pub_loader = DataLoader(Subset(train_ds, pub_idx), batch_size=128, shuffle=True)

    # Adversary uses an architecturally-known head at a middle split point —
    # this is the threat model standard in the literature (white-box on f_head).
    eval_split = max(2, min(5, sim.cfg.split_points[len(sim.cfg.split_points) // 2]))
    snapshot_full = P3SLModel(in_channels=in_ch, num_classes=num_classes).to(sim.device)
    snapshot_full.load_state_dict(sim.global_model.state_dict())
    head_snapshot = FrozenClientHead(snapshot_full, split_layer=eval_split).to(sim.device)
    head_snapshot.eval()

    # Probe IR shape to size the decoder & MIA correctly
    with torch.no_grad():
        probe_x = next(iter(pub_loader))[0][:1].to(sim.device)
        probe_ir = head_snapshot(probe_x)
    ir_dim = 64  # we always pool to 64 (see attacks._flatten_pool)
    print(f"[attacks] eval_split={eval_split}, IR shape per sample={tuple(probe_ir.shape[1:])}")

    decoder = InversionDecoder(ir_dim=ir_dim, image_shape=image_shape).to(sim.device)
    train_inversion_decoder(
        decoder=decoder,
        client_head=head_snapshot,
        public_loader=pub_loader,
        epochs=attack_epochs,
    )

    mia = MembershipInferenceClassifier(ir_dim=ir_dim).to(sim.device)
    member_loader = DataLoader(Subset(train_ds, pub_idx[: len(pub_idx) // 2]), batch_size=128, shuffle=True)
    nonmember_loader = DataLoader(Subset(train_ds, pub_idx[len(pub_idx) // 2 :]), batch_size=128, shuffle=True)
    train_mia_classifier(
        mia=mia,
        client_head=head_snapshot,
        member_loader=member_loader,
        nonmember_loader=nonmember_loader,
        epochs=attack_epochs,
    )

    # ---- Probe a held-out batch to compute reconstruction quality ----
    probe_loader = DataLoader(Subset(train_ds, priv_idx[:128]), batch_size=128, shuffle=False)
    images, labels = next(iter(probe_loader))
    images = images.to(sim.device); labels = labels.to(sim.device)
    images_01 = (images * 0.5) + 0.5  # un-normalise
    with torch.no_grad():
        ir = head_snapshot(images)
    decoder_recon = decoder(ir).detach()
    metrics_decoder = compute_privacy_metrics(images_01, decoder_recon)

    print("[attacks] running optimisation-based reconstruction…")
    opt_recon = optimization_attack(
        target_ir=ir,
        client_head=head_snapshot,
        image_shape=image_shape,
        iterations=opt_attack_iters,
    )
    metrics_opt = compute_privacy_metrics(images_01, opt_recon)

    # ---- MIA AUC on members vs non-members from the held-out half ----
    held_member_loader = DataLoader(Subset(train_ds, priv_idx[:512]), batch_size=128, shuffle=False)
    held_nonmember_loader = DataLoader(Subset(test_ds, list(range(512))), batch_size=128, shuffle=False)
    member_logits, nonmember_logits = [], []
    with torch.no_grad():
        for x, _ in held_member_loader:
            x = x.to(sim.device)
            member_logits.append(mia(head_snapshot(x)).cpu())
        for x, _ in held_nonmember_loader:
            x = x.to(sim.device)
            nonmember_logits.append(mia(head_snapshot(x)).cpu())
    mia_summary = mia_auc(torch.cat(member_logits), torch.cat(nonmember_logits))

    # ---- Backdoor ASR on the trained global model ----
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    asr_correct, asr_total = 0, 0
    clean_correct, clean_total = 0, 0
    sim.global_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(sim.device); y = y.to(sim.device)
            preds = sim.global_model(x).argmax(dim=1)
            clean_correct += int((preds == y).sum().item())
            clean_total += int(y.size(0))
            xb, _ = apply_backdoor(x, y, target_class=9)
            preds_b = sim.global_model(xb).argmax(dim=1)
            mask = y != 9
            asr_correct += int((preds_b[mask] == 9).sum().item())
            asr_total += int(mask.sum().item())
    clean_acc = 100.0 * clean_correct / max(clean_total, 1)
    asr = 100.0 * asr_correct / max(asr_total, 1)

    return {
        "decoder_metrics": metrics_decoder.to_dict(),
        "optimization_metrics": metrics_opt.to_dict(),
        "mia": mia_summary,
        "clean_accuracy": clean_acc,
        "backdoor_asr": asr,
        "eval_split_layer": eval_split,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    with open(args.config) as f:
        plan = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []
    for entry in plan["runs"]:
        name = entry["name"]
        dataset_name = entry.get("dataset", plan.get("dataset", "fashionmnist"))
        alphas = entry.get("alphas", plan.get("alphas", [0.1, 0.3, 0.5, 0.7, 0.9]))
        seeds = entry.get("seeds", plan.get("seeds", [42]))
        run_attacks = bool(entry.get("run_attacks", plan.get("run_attacks", True)))
        attack_epochs = int(entry.get("attack_epochs", plan.get("attack_epochs", 5)))
        opt_attack_iters = int(entry.get("opt_attack_iters", plan.get("opt_attack_iters", 200)))

        cfg_dict = {**plan.get("defaults", {}), **entry.get("config", {})}
        for seed in seeds:
            sname = f"{name}_seed{seed}"
            cfg_dict_s = {**cfg_dict, "seed": int(seed)}
            cfg = make_config_from_dict(cfg_dict_s)
            summary = run_single(
                name=sname,
                cfg=cfg,
                dataset_name=dataset_name,
                alphas=alphas,
                out_dir=out_dir,
                run_attacks=run_attacks,
                attack_epochs=attack_epochs,
                opt_attack_iters=opt_attack_iters,
            )
            aggregate.append(summary)

    # Write aggregated summary across runs
    with open(out_dir / "all_runs.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=float)
    _write_summary_csv(aggregate, out_dir / "all_runs.csv")
    print(f"\n[runner] saved {len(aggregate)} runs to {out_dir}")


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    flat: List[Dict[str, Any]] = []
    for r in rows:
        flat.append(_flatten(r))
    keys = sorted({k for f in flat for k in f.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat:
            w.writerow(r)


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
