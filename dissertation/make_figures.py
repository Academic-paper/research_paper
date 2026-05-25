"""Generate every figure used in thesis.tex from the real benchmark
summaries in research/results/.

The pipeline reads each per-run summary.json and history.csv. Plots that
depend on data we have are produced from that data. Plots that depend on
data we do not have (e.g. the Gaussian-mechanism configurations, which
were not finished within the project deadline) are not produced; the
thesis itself does not reference them.

Run: python3 dissertation/make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "research", "results"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@dataclass
class RunGroup:
    label: str
    prefix: str
    color: str
    final_accs: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    decoder: List[Dict[str, float]] = field(default_factory=list)
    optim: List[Dict[str, float]] = field(default_factory=list)
    mia: List[Dict[str, float]] = field(default_factory=list)
    asr: List[float] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)
    histories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    splits_per_client: Optional[List[int]] = None
    sigma_init: Optional[float] = None
    sigma_decay: Optional[float] = None
    target_accuracy: Optional[float] = None
    dp_mechanism: Optional[str] = None

    def n(self) -> int:
        return len(self.final_accs)

    def acc(self) -> Tuple[float, float]:
        if not self.final_accs:
            return (float("nan"), 0.0)
        m = st.mean(self.final_accs)
        s = st.pstdev(self.final_accs) if len(self.final_accs) > 1 else 0.0
        return m, s

    def metric(self, attr: str, sub: Optional[str] = None) -> Tuple[float, float]:
        vals: List[float] = []
        src = getattr(self, attr)
        for entry in src:
            if isinstance(entry, dict) and sub is not None:
                v = entry.get(sub)
            else:
                v = entry
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(vf) or math.isinf(vf):
                continue
            vals.append(vf)
        if not vals:
            return (float("nan"), 0.0)
        m = st.mean(vals)
        s = st.pstdev(vals) if len(vals) > 1 else 0.0
        return m, s


def load_history(run_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    p = os.path.join(run_dir, "history.csv")
    if not os.path.isfile(p):
        return None
    rounds, accs, sigmas = [], [], []
    with open(p) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                rounds.append(int(row["round"]))
                accs.append(float(row["accuracy"]))
                sigmas.append(float(row.get("sigma_mean", 0.0)))
            except (KeyError, ValueError):
                continue
    if not rounds:
        return None
    return np.asarray(rounds), np.asarray(accs), np.asarray(sigmas)


def load_groups() -> Dict[str, RunGroup]:
    groups = {
        "central":     RunGroup("Centralised (upper bound)",        "central_upper_bound_seed",  "#1f4e8a"),
        "fedavg":      RunGroup("FedAvg (no DP)",                    "fedavg_no_dp_seed",         "#2e7d32"),
        "vanilla":     RunGroup("Vanilla split learning (no DP)",    "vanilla_sl_no_dp_seed",     "#b35900"),
        "vanilla_dp":  RunGroup(r"Vanilla SL + Laplace ($\sigma$=0.5)", "vanilla_sl_laplace_seed", "#5e35b1"),
        "p3sl_low":    RunGroup(r"P3SL + Laplace ($\sigma_0$=0.3)",  "p3sl_laplace_low_seed",     "#a4133c"),
        "p3sl_med":    RunGroup(r"P3SL + Laplace ($\sigma_0$=1.0)",  "p3sl_laplace_med_seed",     "#bf360c"),
    }
    for run_dir in sorted(glob(os.path.join(RESULTS_DIR, "*"))):
        name = os.path.basename(run_dir)
        spath = os.path.join(run_dir, "summary.json")
        if not os.path.isfile(spath):
            continue
        try:
            data = json.load(open(spath))
        except Exception:
            continue
        for g in groups.values():
            if name.startswith(g.prefix):
                g.final_accs.append(float(data["final_accuracy"]))
                eps = data.get("epsilon_total")
                if eps is not None and not (isinstance(eps, float) and (math.isinf(eps) or math.isnan(eps))):
                    g.epsilons.append(float(eps))
                if data.get("decoder_metrics"):
                    g.decoder.append(data["decoder_metrics"])
                if data.get("optimization_metrics"):
                    g.optim.append(data["optimization_metrics"])
                if data.get("mia"):
                    g.mia.append(data["mia"])
                if "backdoor_asr" in data:
                    g.asr.append(float(data["backdoor_asr"]))
                if "train_seconds" in data:
                    g.seconds.append(float(data["train_seconds"]))
                hist = load_history(run_dir)
                if hist is not None:
                    g.histories.append(hist)
                cfg = data.get("config", {})
                if g.sigma_init is None:
                    g.sigma_init = cfg.get("sigma_init")
                    g.sigma_decay = cfg.get("sigma_decay")
                    g.target_accuracy = cfg.get("target_accuracy")
                    g.dp_mechanism = cfg.get("dp_mechanism")
                if g.splits_per_client is None and hist is not None:
                    try:
                        with open(os.path.join(run_dir, "history.csv")) as fh:
                            rdr = csv.DictReader(fh)
                            row = next(rdr)
                            sp = []
                            for k in sorted(row):
                                if k.startswith("client") and k.endswith("_split"):
                                    sp.append(int(float(row[k])))
                            if sp:
                                g.splits_per_client = sp
                    except Exception:
                        pass
                break
    return groups


# ---------------------------------------------------------------------------
# Diagram primitives (used by architectural figures)
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, text, fc="#e8f0fe", ec="#1f4e8a", lw=1.4, fontsize=9, weight="normal"):
    p = FancyBboxPatch(xy, w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=lw, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.2, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=12, color=color, linewidth=lw)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Architectural diagrams (data-independent)
# ---------------------------------------------------------------------------
def fig_topologies():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    titles = ["Centralised learning", "Federated learning", "Split learning"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
        ax.set_title(t, fontweight="bold")
    client_xs = [0.25, 1.65, 3.05, 4.45]
    client_w = 1.2
    arrow_offsets = [x + client_w / 2 for x in client_xs]
    ax = axes[0]
    box(ax, (2.3, 3.5), 1.4, 1.0, "Server\n(full model)", fc="#fce4ec", ec="#a4133c")
    for x, ax_x in zip(client_xs, arrow_offsets):
        box(ax, (x, 0.7), client_w, 0.7, "Client\n(raw data)", fc="#fff3e0", ec="#b35900", fontsize=7.5)
        arrow(ax, ax_x, 1.4, 3.0, 3.5, color="#888")
    ax.text(3.0, 0.1, "raw inputs leave the device", ha="center", fontsize=8, style="italic")
    ax = axes[1]
    box(ax, (2.3, 3.5), 1.4, 1.0, "Server\n(aggregator)", fc="#fce4ec", ec="#a4133c")
    for x, ax_x in zip(client_xs, arrow_offsets):
        box(ax, (x, 0.7), client_w, 0.7, "Client\n(full model)", fc="#fff3e0", ec="#b35900", fontsize=7.5)
        arrow(ax, ax_x, 1.4, 3.0, 3.5, color="#888")
        arrow(ax, 3.0, 3.5, ax_x, 1.4, color="#bbb", style="-")
    ax.text(3.0, 0.1, "weights / gradients exchanged", ha="center", fontsize=8, style="italic")
    ax = axes[2]
    for x in [0.3, 1.7, 3.1, 4.5]:
        box(ax, (x, 4.7), 1.0, 0.7, "Front\n(client)", fc="#e8f5e9", ec="#2e7d32", fontsize=8)
    box(ax, (1.5, 2.3), 3.0, 1.0, "Server (back layers)", fc="#fce4ec", ec="#a4133c")
    for x in [0.8, 2.2, 3.6, 5.0]:
        arrow(ax, x, 4.7, 3.0, 3.3, color="#888")
        arrow(ax, 3.0, 2.3, x, 4.7, color="#bbb", style="-")
    box(ax, (1.5, 0.4), 3.0, 0.8, "intermediate representation", fc="#ede7f6", ec="#5e35b1", fontsize=8)
    ax.text(3.0, 0.05, "only activations cross the boundary", ha="center", fontsize=8, style="italic")
    fig.suptitle("Three families of collaborative learning", fontsize=12, fontweight="bold")
    save(fig, "fig_topologies")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    box(ax, (0.2, 5.5), 2.6, 1.4,
        r"Client $i$" + "\n" + r"front network $f_i$" + "\n" + r"local data $D_i$",
        fc="#e8f5e9", ec="#2e7d32")
    box(ax, (0.2, 2.3), 2.6, 1.4,
        "Local DP layer\n(Laplace noise on the IR)",
        fc="#ede7f6", ec="#5e35b1")
    arrow(ax, 1.5, 5.5, 1.5, 3.7)
    box(ax, (4.0, 3.5), 3.0, 2.0,
        "Intermediate\nrepresentation\n" + r"$\tilde{a}_i = a_i + \xi$",
        fc="#fff3e0", ec="#b35900", fontsize=10, weight="bold")
    arrow(ax, 2.8, 3.0, 4.0, 4.5)
    box(ax, (8.2, 5.5), 3.0, 1.4,
        "Server back network\nshared across clients",
        fc="#fce4ec", ec="#a4133c")
    box(ax, (8.2, 3.5), 3.0, 1.4,
        "Task loss " + r"$\mathcal{L}_i$",
        fc="#fce4ec", ec="#a4133c")
    arrow(ax, 7.0, 4.5, 8.2, 6.2)
    arrow(ax, 9.7, 5.5, 9.7, 4.9)
    box(ax, (8.2, 1.0), 3.0, 1.6,
        "Bi-level outer step\nadjust " + r"$\sigma$ when round-end" + "\nacc reaches target",
        fc="#fff8e1", ec="#f57f17")
    arrow(ax, 9.7, 3.5, 9.7, 2.6)
    arrow(ax, 8.2, 1.8, 1.5, 2.3, color="#888")
    ax.text(4.8, 1.4, r"$\sigma$ feedback", fontsize=9, style="italic", color="#555")
    box(ax, (12.2, 3.5), 1.6, 2.0,
        "Backward\ngradients",
        fc="#fbe9e7", ec="#bf360c", fontsize=9)
    arrow(ax, 11.2, 4.5, 12.2, 4.5)
    arrow(ax, 12.2, 3.6, 7.0, 3.6, color="#888")
    ax.text(7.0, 7.5,
            "P3SL: per-client split + bi-level utility loop on " + r"$\sigma$",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(1.5, 7.3, "client side", ha="center", fontsize=9, style="italic", color="#2e7d32")
    ax.text(9.7, 7.3, "server side", ha="center", fontsize=9, style="italic", color="#a4133c")
    save(fig, "fig_architecture")


def fig_cnn_splits():
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")
    layers = [
        ("input\n28x28x1", "#fafafa", "#666"),
        ("conv 3x3\n+ReLU", "#e8f5e9", "#2e7d32"),
        ("pool 2x2", "#e8f5e9", "#2e7d32"),
        ("conv 3x3\n+ReLU", "#e8f5e9", "#2e7d32"),
        ("pool 2x2", "#e8f5e9", "#2e7d32"),
        ("flatten", "#fff3e0", "#b35900"),
        ("FC 256", "#fce4ec", "#a4133c"),
        ("ReLU", "#fce4ec", "#a4133c"),
        ("dropout", "#fce4ec", "#a4133c"),
        ("FC 128", "#fce4ec", "#a4133c"),
        ("ReLU", "#fce4ec", "#a4133c"),
        ("FC 10", "#fbe9e7", "#bf360c"),
    ]
    w = 0.95
    splits = {2, 5, 6, 9, 10}
    for i, (name, fc, ec) in enumerate(layers):
        x = 0.6 + i * 1.05
        box(ax, (x, 1.6), w, 1.2, name, fc=fc, ec=ec, fontsize=8)
        if i + 1 in splits:
            ax.plot([x + w + 0.05, x + w + 0.05], [1.3, 3.1], "--", color="#a4133c", linewidth=1.4)
            ax.text(x + w + 0.05, 3.25, f"S={i+1}", ha="center", color="#a4133c",
                    fontsize=9, fontweight="bold")
        if i < len(layers) - 1:
            arrow(ax, x + w, 2.2, x + 1.05, 2.2)
    ax.text(7.0, 0.6,
            "Each client owns layers 1..S (front network); server owns layers S+1..L (back network).",
            ha="center", fontsize=9, style="italic")
    ax.text(7.0, 3.9, r"Allowed split points $S \in \{2,5,6,9,10\}$",
            ha="center", fontsize=10, fontweight="bold", color="#a4133c")
    save(fig, "fig_cnn_splits")


def fig_round():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    steps = [
        ("(1) sample\nmini-batch", "#fff3e0", "#b35900"),
        ("(2) forward\nfront network", "#e8f5e9", "#2e7d32"),
        ("(3) add\nDP noise", "#ede7f6", "#5e35b1"),
        ("(4) send IR\nto server", "#fff8e1", "#f57f17"),
        ("(5) forward\nback network", "#fce4ec", "#a4133c"),
        ("(6) loss + grad\non server", "#fce4ec", "#a4133c"),
        ("(7) return\nIR-grad", "#fff8e1", "#f57f17"),
        ("(8) update\nfront network", "#e8f5e9", "#2e7d32"),
    ]
    for i, (label, fc, ec) in enumerate(steps):
        x = 0.2 + i * 1.45
        box(ax, (x, 1.7), 1.25, 1.6, label, fc=fc, ec=ec, fontsize=8)
        if i < len(steps) - 1:
            arrow(ax, x + 1.25, 2.5, x + 1.45, 2.5)
    ax.axhline(0.9, color="#bbb", linestyle=":")
    ax.text(6.0, 0.5,
            "client-side                                        server-side                                    client-side",
            ha="center", fontsize=9, color="#555")
    ax.text(6.0, 4.2, "Single P3SL training round", ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig_round")


def fig_bilevel():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, (4.2, 4.6), 2.6, 1.0, r"Init $\sigma^{(0)}$, per-client $S_i$",
        fc="#e3f2fd", ec="#1f4e8a", fontsize=10)
    box(ax, (4.2, 3.0), 2.6, 1.2,
        "Inner loop:\nstandard SGD on " + r"$f_i, g$",
        fc="#e8f5e9", ec="#2e7d32", fontsize=10)
    box(ax, (4.2, 1.4), 2.6, 1.2,
        "Measure round-end\naccuracy " + r"$\hat{\tau}^{(t)}$",
        fc="#fff3e0", ec="#b35900", fontsize=10)
    box(ax, (8.2, 1.4), 2.4, 1.2,
        "Outer step:\nshrink " + r"$\sigma$ once" + "\n" + r"$\hat{\tau} \geq \tau$",
        fc="#fff8e1", ec="#f57f17", fontsize=10)
    box(ax, (0.4, 1.4), 2.6, 1.2,
        "Stop if T rounds\nreached or " + r"$\sigma \leq \sigma_{\mathrm{floor}}$",
        fc="#fce4ec", ec="#a4133c", fontsize=10)
    arrow(ax, 5.5, 4.6, 5.5, 4.2)
    arrow(ax, 5.5, 3.0, 5.5, 2.6)
    arrow(ax, 6.8, 2.0, 8.2, 2.0)
    arrow(ax, 8.2, 1.6, 3.0, 1.6, color="#888")
    arrow(ax, 1.7, 2.6, 1.7, 4.2, color="#888")
    arrow(ax, 1.7, 4.5, 4.2, 5.0, color="#888")
    ax.text(5.5, 5.7, "Bi-level utility-bounded DP loop",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.5, 0.6,
            r"$\sigma^{(t+1)} = \max(\sigma_{\mathrm{floor}}, \rho \cdot \sigma^{(t)})$"
            + r" once $\hat{\tau}^{(t)} \geq \tau$",
            ha="center", fontsize=10, style="italic")
    save(fig, "fig_bilevel")


def fig_threat():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.0); ax.axis("off")
    box(ax, (0.4, 1.7), 2.4, 1.4, "Honest client\nowns " + r"$f_i, D_i$",
        fc="#e8f5e9", ec="#2e7d32")
    box(ax, (4.0, 1.7), 3.0, 1.4,
        "Honest-but-curious\nserver " + r"$g$" + "\n(observes IR)",
        fc="#fff3e0", ec="#b35900")
    box(ax, (8.2, 1.7), 2.4, 1.4,
        "External\nmembership / poison\nadversary",
        fc="#fce4ec", ec="#a4133c")
    arrow(ax, 2.8, 2.4, 4.0, 2.4)
    arrow(ax, 8.2, 2.4, 7.0, 2.4, color="#bbb")
    ax.text(5.5, 4.4, "Threat model: who sees what", ha="center",
            fontsize=12, fontweight="bold")
    ax.text(1.6, 0.9, "private", ha="center", fontsize=9, style="italic", color="#2e7d32")
    ax.text(5.5, 0.9, r"sees only $\tilde{a}_i$ and gradients",
            ha="center", fontsize=9, style="italic", color="#b35900")
    ax.text(9.4, 0.9, "queries trained model", ha="center", fontsize=9,
            style="italic", color="#a4133c")
    box(ax, (0.4, 3.6), 2.4, 0.8, "Reconstruction\nattack", fc="#fafafa", ec="#666", fontsize=8)
    box(ax, (4.0, 3.6), 3.0, 0.8, "Inversion attacks\n(decoder, optimisation)", fc="#fafafa", ec="#666", fontsize=8)
    box(ax, (8.2, 3.6), 2.4, 0.8, "MIA, backdoor", fc="#fafafa", ec="#666", fontsize=8)
    save(fig, "fig_threat")


def fig_gantt():
    tasks = [
        ("Literature review",                "Aug 2025", 6),
        ("Reference repo study",             "Sep 2025", 4),
        ("Single-process simulator",         "Sep 2025", 6),
        ("DP mechanism + accountant",        "Oct 2025", 5),
        ("Bi-level optimisation loop",       "Oct 2025", 5),
        ("Attack suite",                     "Nov 2025", 7),
        ("Benchmark grid (3 seeds)",         "Dec 2025", 6),
        ("Figure pipeline",                  "Jan 2026", 3),
        ("Dissertation writing",             "Feb 2026", 8),
        ("Internal review",                  "Apr 2026", 4),
        ("Submission",                       "May 2026", 1),
    ]
    week_starts = [0, 4, 6, 12, 16, 21, 28, 33, 36, 44, 48]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tasks)))
    for i, ((name, _, dur), s, c) in enumerate(zip(tasks, week_starts, colors)):
        ax.barh(i, dur, left=s, color=c, edgecolor="black")
        ax.text(s + dur + 0.2, i, f"{dur} wk", va="center", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Weeks since project start (Aug 2025)")
    ax.set_title("Project work plan and timeline")
    ax.grid(alpha=0.3, axis="x")
    save(fig, "fig_gantt")


# ---------------------------------------------------------------------------
# Data-driven figures (read real summaries)
# ---------------------------------------------------------------------------
def fig_results_bar(groups: Dict[str, RunGroup]):
    order = ["central", "fedavg", "vanilla", "vanilla_dp", "p3sl_low", "p3sl_med"]
    short = {
        "central":    "Centralised\n(upper bound)",
        "fedavg":     "FedAvg\n(no DP)",
        "vanilla":    "Vanilla SL\n(no DP)",
        "vanilla_dp": r"Vanilla SL" + "\n" + r"+ Lap. ($\sigma$=0.5)",
        "p3sl_low":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=0.3)",
        "p3sl_med":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=1.0)",
    }
    labels, means, stds, colors, ns = [], [], [], [], []
    for k in order:
        g = groups[k]
        if g.n() == 0:
            continue
        m, s = g.acc()
        labels.append(short[k]); means.append(m); stds.append(s)
        colors.append(g.color); ns.append(g.n())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Final accuracy across configurations (FashionMNIST, 30K subset, 25 rounds)")
    lo = max(50, min(means) - max(stds) - 4)
    hi = max(means) + max(stds) + 3
    ax.set_ylim(lo, hi); ax.grid(alpha=0.3, axis="y")
    for b, m, s, n in zip(bars, means, stds, ns):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.25,
                f"{m:.2f}$\\pm${s:.2f}\nn={n}", ha="center", fontsize=8)
    save(fig, "fig_results_bar")


def fig_recon_bars(groups: Dict[str, RunGroup]):
    short = {
        "vanilla":    "Vanilla SL\n(no DP)",
        "vanilla_dp": r"Vanilla SL" + "\n" + r"+ Lap. ($\sigma$=0.5)",
        "p3sl_low":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=0.3)",
        "p3sl_med":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=1.0)",
    }
    keys = [k for k in ["vanilla", "vanilla_dp", "p3sl_low", "p3sl_med"]
            if groups[k].n() > 0]
    if not keys:
        return
    cfgs = [groups[k] for k in keys]
    labels = [short[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(cfgs))
    width = 0.36

    psnr_dec = [g.metric("decoder", "psnr")[0] for g in cfgs]
    psnr_dec_e = [g.metric("decoder", "psnr")[1] for g in cfgs]
    psnr_opt = [g.metric("optim", "psnr")[0] for g in cfgs]
    psnr_opt_e = [g.metric("optim", "psnr")[1] for g in cfgs]
    axes[0].bar(x - width/2, psnr_dec, width, yerr=psnr_dec_e, color="#1f4e8a",
                edgecolor="black", label="Decoder attack", capsize=4)
    axes[0].bar(x + width/2, psnr_opt, width, yerr=psnr_opt_e, color="#a4133c",
                edgecolor="black", label="Optimisation attack", capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("PSNR (dB)")
    pmin = min(psnr_opt) - 1.0
    pmax = max(psnr_dec) + 1.0
    axes[0].set_ylim(pmin, pmax)
    axes[0].set_title("Reconstruction PSNR (higher is worse for the defender)")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].legend(loc="lower right", fontsize=8)

    ssim_dec = [g.metric("decoder", "ssim")[0] for g in cfgs]
    ssim_dec_e = [g.metric("decoder", "ssim")[1] for g in cfgs]
    ssim_opt = [g.metric("optim", "ssim")[0] for g in cfgs]
    ssim_opt_e = [g.metric("optim", "ssim")[1] for g in cfgs]
    axes[1].bar(x - width/2, ssim_dec, width, yerr=ssim_dec_e, color="#1f4e8a",
                edgecolor="black", label="Decoder attack", capsize=4)
    axes[1].bar(x + width/2, ssim_opt, width, yerr=ssim_opt_e, color="#a4133c",
                edgecolor="black", label="Optimisation attack", capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel("SSIM")
    smin = min(ssim_dec + ssim_opt) - 0.02
    smax = max(ssim_dec + ssim_opt) + 0.02
    axes[1].set_ylim(smin, smax)
    axes[1].set_title("Reconstruction SSIM (higher is worse for the defender)")
    axes[1].grid(alpha=0.3, axis="y"); axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Reconstruction quality across configurations (mean over seeds)",
                 fontsize=11, fontweight="bold")
    save(fig, "fig_recon")


_RECON_SHORT = {
    "vanilla":    "Vanilla SL\n(no DP)",
    "vanilla_dp": r"Vanilla SL" + "\n" + r"+ Lap. ($\sigma$=0.5)",
    "p3sl_low":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=0.3)",
    "p3sl_med":   r"P3SL" + "\n" + r"+ Lap. ($\sigma_0$=1.0)",
}


def fig_mia_bars(groups: Dict[str, RunGroup]):
    keys = [k for k in ["vanilla", "vanilla_dp", "p3sl_low", "p3sl_med"]
            if groups[k].n() > 0]
    if not keys:
        return
    cfgs = [groups[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [_RECON_SHORT[k] for k in keys]
    x = np.arange(len(cfgs))
    auc = [g.metric("mia", "auc")[0] for g in cfgs]
    auc_e = [g.metric("mia", "auc")[1] for g in cfgs]
    bars = ax.bar(x, auc, yerr=auc_e, color="#5e35b1", edgecolor="black", capsize=4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MIA AUROC")
    ax.set_ylim(0.4, 0.65)
    ax.set_title("Membership-inference AUROC across configurations")
    ax.grid(alpha=0.3, axis="y"); ax.legend(loc="upper right", fontsize=8)
    for b, m in zip(bars, auc):
        ax.text(b.get_x() + b.get_width()/2, m + 0.005, f"{m:.3f}",
                ha="center", fontsize=8)
    save(fig, "fig_mia")


def fig_backdoor_bars(groups: Dict[str, RunGroup]):
    keys = [k for k in ["vanilla", "vanilla_dp", "p3sl_low", "p3sl_med"]
            if groups[k].n() > 0]
    if not keys:
        return
    cfgs = [groups[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [_RECON_SHORT[k] for k in keys]
    x = np.arange(len(cfgs))
    asr = [g.metric("asr")[0] * 100 for g in cfgs]
    asr_e = [g.metric("asr")[1] * 100 for g in cfgs]
    bars = ax.bar(x, asr, yerr=asr_e, color="#a4133c", edgecolor="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Backdoor attack-success rate (%)")
    ax.set_ylim(0, max(asr_e) * 1.1 + max(asr) + 12)
    ax.set_title("Backdoor ASR across configurations (mean $\\pm$ std over seeds)")
    ax.grid(alpha=0.3, axis="y")
    for b, m, s in zip(bars, asr, asr_e):
        ax.text(b.get_x() + b.get_width()/2, m + s + 1.0,
                f"{m:.1f}%", ha="center", fontsize=8)
    save(fig, "fig_backdoor")


def fig_eps_acc(groups: Dict[str, RunGroup]):
    pts = []
    for k in ["vanilla_dp", "p3sl_low", "p3sl_med"]:
        g = groups[k]
        if g.n() == 0:
            continue
        eps_m, eps_s = g.metric("epsilons")
        acc_m, acc_s = g.acc()
        if math.isnan(eps_m) or math.isnan(acc_m):
            continue
        pts.append((g.label, eps_m, eps_s, acc_m, acc_s, g.color))
    g0 = groups["vanilla"]
    if g0.n() > 0:
        m, s = g0.acc()
        pts.append((g0.label, math.nan, 0.0, m, s, g0.color))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    no_dp_acc = None
    for label, eps, _, acc, accs, color in pts:
        if math.isnan(eps):
            no_dp_acc = acc
            ax.axhline(acc, color=color, linestyle=":", linewidth=1)
        else:
            ax.errorbar(eps, acc, yerr=accs, fmt="o", color=color,
                        markersize=10, label=label, capsize=4)
            ax.annotate(rf" $\varepsilon$={eps:.0f}, acc={acc:.2f}$\pm${accs:.2f}",
                        (eps, acc), textcoords="offset points",
                        xytext=(8, 4), fontsize=8)
    ax.axvline(10, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Reported $\varepsilon$ (Laplace closed form, sensitivity = 1, log scale)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_xlim(1, 200)
    ymin = min(p[3] - p[4] for p in pts) - 0.6
    ymax = (no_dp_acc if no_dp_acc is not None else max(p[3] for p in pts)) + 0.6
    ax.set_ylim(ymin, ymax)
    if no_dp_acc is not None:
        ax.text(180, no_dp_acc - 0.15,
                r"vanilla SL (no DP, $\varepsilon = \infty$)",
                color="#b35900", fontsize=8, ha="right", va="top")
    ax.text(9.3, ymax - 0.25,
            r"$\varepsilon = 10$ (commonly cited threshold)",
            color="grey", fontsize=8, ha="right", va="top", rotation=90)
    ax.set_title(r"Privacy / utility scatter: $\varepsilon$ vs accuracy across configurations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig_eps_acc")


def fig_pareto(groups: Dict[str, RunGroup]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    no_dp_acc = None
    no_dp_modes: List[str] = []
    dp_pts = []
    for k in ["fedavg", "vanilla", "vanilla_dp", "p3sl_low", "p3sl_med"]:
        g = groups[k]
        if g.n() == 0:
            continue
        eps_m, _ = g.metric("epsilons")
        acc_m, acc_s = g.acc()
        if math.isnan(eps_m):
            no_dp_acc = acc_m
            no_dp_modes.append(g.label.split(" (")[0])
        else:
            dp_pts.append((g.label, eps_m, acc_m, acc_s, g.color))
    dp_pts.sort(key=lambda r: r[1])

    if no_dp_acc is not None:
        ax.axhline(no_dp_acc, color="#888", linestyle=":", linewidth=1.2,
                   label=f"no-DP baselines @ {no_dp_acc:.2f}% ($\\varepsilon = \\infty$)")
    for label, eps, acc, accs, color in dp_pts:
        ax.errorbar(eps, acc, yerr=accs, fmt="o", color=color, markersize=11,
                    capsize=4, label=label)
    for label, eps, acc, accs, color in dp_pts:
        ax.annotate(rf" $\varepsilon$={eps:.0f}", (eps, acc),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlabel(r"Reported $\varepsilon$ (Laplace closed form, sensitivity = 1, log scale)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_xlim(10, 200)
    if no_dp_acc is not None and dp_pts:
        ymin = min(p[2] - p[3] for p in dp_pts) - 0.6
        ymax = no_dp_acc + 0.6
        ax.set_ylim(ymin, ymax)
    ax.set_title("Privacy / utility frontier across DP configurations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_pareto")


def fig_utility(groups: Dict[str, RunGroup]):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = 0
    plot_order = [
        ("central",    "-",  2.2),
        ("fedavg",     "-",  2.0),
        ("vanilla",    "--", 2.0),
        ("vanilla_dp", "-",  1.8),
        ("p3sl_low",   "-",  1.8),
        ("p3sl_med",   "--", 1.8),
    ]
    for k, ls, lw in plot_order:
        g = groups[k]
        if not g.histories:
            continue
        rounds, accs, _ = g.histories[0]
        ax.plot(rounds, accs, label=g.label, color=g.color, linestyle=ls, linewidth=lw)
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No history.csv files found",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy across communication rounds (seed 42)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_utility")


def fig_sigma_traj(groups: Dict[str, RunGroup]):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    plotted = 0
    for k, color, label in [
        ("p3sl_low", "#5e35b1", r"P3SL Laplace, $\sigma_0=0.3$"),
        ("p3sl_med", "#a4133c", r"P3SL Laplace, $\sigma_0=1.0$"),
        ("vanilla_dp", "#1f4e8a", r"Vanilla SL + Laplace, $\sigma=0.5$"),
    ]:
        g = groups[k]
        if not g.histories:
            continue
        for j, (rounds, accs, sigmas) in enumerate(g.histories):
            ls = "-" if j == 0 else (":" if j == 1 else "--")
            ax.plot(rounds, sigmas, color=color, linestyle=ls,
                    linewidth=1.8, alpha=0.85,
                    label=label if j == 0 else None)
            plotted += 1
    if g := groups["p3sl_low"]:
        if g.target_accuracy is not None:
            ax.axhline(0, color="#888", linestyle=":", linewidth=0.8)
    if plotted == 0:
        ax.text(0.5, 0.5, "No history.csv files found",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"Mean noise scale $\bar{\sigma}^{(t)}$")
    ax.set_title("Noise scale trajectory under the bi-level loop (one line per seed)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    save(fig, "fig_sigma_traj")


def fig_heterogeneity(groups: Dict[str, RunGroup]):
    g = groups.get("p3sl_low") or groups.get("p3sl_med")
    if g is None or g.splits_per_client is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.0))
    splits = g.splits_per_client
    labels = [f"Client {chr(ord('A')+i)}" for i in range(len(splits))]
    x = np.arange(len(splits))
    bars = ax.bar(x, splits, color="#5e35b1", edgecolor="black", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Front-network depth (split index $S_i$)")
    ax.set_yticks([2, 5, 6, 9, 10])
    ax.set_title("Per-client split assignments under P3SL")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, splits):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"S={v}",
                ha="center", fontsize=9, fontweight="bold")
    save(fig, "fig_heterogeneity")


# ---------------------------------------------------------------------------
def main():
    groups = load_groups()
    n_runs = sum(g.n() for g in groups.values())
    print(f"Loaded {n_runs} run summaries from {RESULTS_DIR}")
    for k, g in groups.items():
        m, s = g.acc()
        print(f"  {k:<10} n={g.n():>2} acc={m:.2f}±{s:.2f}")

    fig_topologies()
    fig_architecture()
    fig_cnn_splits()
    fig_round()
    fig_bilevel()
    fig_threat()
    fig_gantt()

    fig_results_bar(groups)
    fig_recon_bars(groups)
    fig_mia_bars(groups)
    fig_backdoor_bars(groups)
    fig_eps_acc(groups)
    fig_pareto(groups)
    fig_utility(groups)
    fig_sigma_traj(groups)
    fig_heterogeneity(groups)

    print(f"All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
