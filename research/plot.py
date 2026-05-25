"""Generate publication-quality figures from the JSON / CSV emitted by
``run_experiments.py``.

Outputs (all under ``research/results/figures``):

* ``utility_vs_round.pdf``       — accuracy curve for every run, with std-dev shading
* ``privacy_utility_pareto.pdf`` — accuracy vs ε scatter, P3SL vs baselines
* ``reconstruction_metrics.pdf`` — bar chart of FSIM/SSIM/PSNR/MSE/NN-id
* ``mia_metrics.pdf``             — bar chart of AUC, balanced acc, TPR@FPR=0.1
* ``backdoor_asr.pdf``            — clean-acc vs ASR per run
* ``per_client_split.pdf``        — heat-map of split layer chosen per client per round

Usage::

    python research/plot.py --results research/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        sj = run_dir / "summary.json"
        hc = run_dir / "history.csv"
        if not sj.exists() or not hc.exists():
            continue
        with open(sj) as f:
            summary = json.load(f)
        history: List[Dict[str, float]] = []
        with open(hc) as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({k: float(v) if v not in ("", "nan") else float("nan") for k, v in row.items()})
        summary["history"] = history
        out.append(summary)
    return out


def _strip_seed(name: str) -> str:
    if "_seed" in name:
        return name.rsplit("_seed", 1)[0]
    return name


def _aggregate_by_run_name(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[_strip_seed(r["name"])].append(r)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 110,
    })

    results_dir = Path(args.results)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No completed runs found in {results_dir}")
    grouped = _aggregate_by_run_name(runs)
    print(f"[plot] {len(runs)} runs across {len(grouped)} configurations")

    # ------------------------------------------------------------------ 1) accuracy vs round
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, group in sorted(grouped.items()):
        # Stack per-round accuracy across seeds, pad to common length
        max_len = max(len(r["history"]) for r in group)
        acc_matrix = np.full((len(group), max_len), np.nan)
        for i, r in enumerate(group):
            accs = [h.get("accuracy", float("nan")) for h in r["history"]]
            acc_matrix[i, : len(accs)] = accs
        mean = np.nanmean(acc_matrix, axis=0)
        std = np.nanstd(acc_matrix, axis=0)
        rounds = np.arange(1, max_len + 1)
        ax.plot(rounds, mean, label=name)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.20)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global accuracy (%)")
    ax.set_title("Utility curve (mean ± std over seeds)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "utility_vs_round.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ 2) Pareto plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, group in grouped.items():
        accs = [r["final_accuracy"] for r in group]
        eps = [r.get("epsilon_total", float("inf")) for r in group]
        eps_finite = [e if (e is not None and np.isfinite(e)) else 1e3 for e in eps]
        ax.errorbar(np.mean(eps_finite), np.mean(accs), xerr=np.std(eps_finite),
                    yerr=np.std(accs), fmt="o", label=name, markersize=8, capsize=4)
        ax.annotate(name, (np.mean(eps_finite), np.mean(accs)), fontsize=7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Total privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title("Privacy–utility Pareto")
    fig.tight_layout()
    fig.savefig(fig_dir / "privacy_utility_pareto.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ 3) reconstruction
    rec_keys = ["mse", "psnr", "ssim", "fsim", "nn_id_acc"]
    names, vals = [], {k: [] for k in rec_keys}
    for name, group in grouped.items():
        if "decoder_metrics" not in group[0]:
            continue
        names.append(name)
        for k in rec_keys:
            v = [g["decoder_metrics"].get(k, float("nan")) for g in group if "decoder_metrics" in g]
            vals[k].append(np.nanmean(v))
    if names:
        fig, axes = plt.subplots(1, len(rec_keys), figsize=(3.0 * len(rec_keys), 4))
        for ax, k in zip(axes, rec_keys):
            ax.bar(names, vals[k])
            ax.set_title(f"Decoder — {k.upper()}")
            ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(fig_dir / "reconstruction_metrics.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------ 4) MIA
    mia_keys = ["auc", "balanced_acc", "tpr_at_fpr_0.1"]
    mia_names, mia_vals = [], {k: [] for k in mia_keys}
    for name, group in grouped.items():
        if "mia" not in group[0]:
            continue
        mia_names.append(name)
        for k in mia_keys:
            mia_vals[k].append(np.nanmean([g["mia"].get(k, float("nan")) for g in group if "mia" in g]))
    if mia_names:
        fig, axes = plt.subplots(1, len(mia_keys), figsize=(3.0 * len(mia_keys), 4))
        for ax, k in zip(axes, mia_keys):
            ax.bar(mia_names, mia_vals[k])
            ax.set_title(f"MIA — {k}")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
            ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(fig_dir / "mia_metrics.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------ 5) backdoor
    bd_names, asr, clean = [], [], []
    for name, group in grouped.items():
        if "backdoor_asr" not in group[0]:
            continue
        bd_names.append(name)
        asr.append(np.nanmean([g["backdoor_asr"] for g in group]))
        clean.append(np.nanmean([g.get("clean_accuracy", float("nan")) for g in group]))
    if bd_names:
        x = np.arange(len(bd_names))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, clean, width=0.4, label="Clean Acc")
        ax.bar(x + 0.2, asr, width=0.4, label="Backdoor ASR", color="crimson")
        ax.set_xticks(x)
        ax.set_xticklabels(bd_names, rotation=70)
        ax.set_ylabel("%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "backdoor_asr.pdf")
        plt.close(fig)

    print(f"[plot] figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
