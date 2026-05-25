# P3SL Research Harness

A clean, single-process re-implementation of the Personalized Privacy-Preserving
Split Learning protocol used to generate **publication-grade** numbers (sweeps,
baselines, multi-seed averages, Pareto curves, attack benchmarks).

The Docker code in `my_implementation/P3SL_Implementation` is the *production*
reference. This `research/` directory is what you run for the paper.

## Why a separate harness?

Issues that made the Docker code unsuitable for a paper:

| # | Issue | Fix in `research/` |
|---|---|---|
| 1 | `TOTAL_ROUNDS = 1` — model never converges | `rounds: 30` (configurable) |
| 2 | Pre-train epochs = 1, opt-attack iters = 10 | `attack_epochs: 5`, `opt_attack_iters: 200` |
| 3 | Bi-level optimization is commented out | Re-enabled in `simulator._update_noise_table` |
| 4 | Energy + FSIM tables are fabricated | Profile object exposed; user can plug in measured values |
| 5 | Laplace noise has no formal DP guarantee (no clipping) | `p3sl/dp.py` clips IR to L1 ball, returns ε = Δ/b |
| 6 | Only FSIM reported | MSE / PSNR / SSIM / FSIM / LPIPS / NN-identifiability |
| 7 | MIA reports only mean confidence | AUROC, balanced acc, TPR @ FPR=0.1 |
| 8 | Single seed | Multi-seed sweeps with std-dev shading |
| 9 | One dataset | MNIST / FashionMNIST / CIFAR-10 selectable |
| 10 | No baselines | central / vanilla-SL / FedAvg / no-DP |
| 11 | Six containers, slow iteration | Single Python process |

## Layout

```
research/
├── p3sl/
│   ├── model.py        # P3SLModel (identical to Docker version)
│   ├── dp.py           # Laplace + Gaussian mechanisms with clipping & RDP
│   ├── attacks.py      # InversionDecoder, MIA, optimisation attack, backdoor
│   ├── metrics.py      # MSE/PSNR/SSIM/FSIM/LPIPS/NN-id, MIA AUC, ASR
│   └── simulator.py    # Trainer for p3sl / vanilla_sl / fedavg / central
├── configs/
│   ├── smoke.yaml      # ≤ 1 min smoke test
│   └── main.yaml       # full benchmark (≈ 30 min CPU, < 5 min GPU)
├── run_experiments.py  # main entry point
├── plot.py             # produces every paper figure
└── requirements.txt
```

## Quick start

```bash
pip install -r research/requirements.txt

# 1) Smoke test — ≈ 1 minute on a laptop CPU
python research/run_experiments.py --config research/configs/smoke.yaml

# 2) Full benchmark — produces every figure / table
python research/run_experiments.py --config research/configs/main.yaml

# 3) Generate publication figures
python research/plot.py --results research/results
```

Results are written under `research/results/<run_name>/`:

* `history.csv`     — per-round accuracy, loss, mean σ, mean split layer, per-client splits
* `summary.json`    — final accuracy, ε, attack-success metrics
* `figures/*.pdf`   — utility curves, Pareto plot, reconstruction bars, MIA bars, ASR bars

## Reproducibility

Every run is fully seeded (`torch`, `numpy`, `random`, `cuDNN deterministic`).
The default plan runs each configuration with three seeds (`42, 1, 7`) so you
can report mean ± std-dev. To add more seeds, edit `seeds:` in the YAML.

## Adding a new experiment

1. Open `configs/main.yaml` and append a new `runs:` entry, e.g.

   ```yaml
   - name: p3sl_dirichlet_extreme
     config:
       mode: p3sl
       dp_mechanism: laplace
       sigma_init: 1.5
       iid: false
       dirichlet_alpha: 0.1
   ```

2. Re-run `run_experiments.py`; only the new entry is added (existing CSVs are
   not deleted).
3. Re-run `plot.py` to refresh figures.

## What numbers does the paper need?

The harness emits everything required for the canonical Tables 1–3 and
Figures 1–4 of a privacy-preserving SL paper:

| Section of paper | Source artefact |
|---|---|
| Table 1 — final accuracy by mode | `all_runs.csv` (`final_accuracy`) |
| Table 2 — privacy budget ε | `summary.json` (`epsilon_total`) |
| Table 3 — attack metrics | `summary.json` (`decoder_metrics`, `mia`, `backdoor_asr`) |
| Fig. 1 — utility vs round | `figures/utility_vs_round.pdf` |
| Fig. 2 — privacy-utility Pareto | `figures/privacy_utility_pareto.pdf` |
| Fig. 3 — reconstruction quality | `figures/reconstruction_metrics.pdf` |
| Fig. 4 — MIA + Backdoor | `figures/mia_metrics.pdf` + `figures/backdoor_asr.pdf` |
