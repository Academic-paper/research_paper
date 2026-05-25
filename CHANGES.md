# Changes for paper-grade results

This document lists every change I made to the upstream
`Academic-paper/research_paper` repository, grouped by motivation, so reviewers
(and you) can audit the deltas.

---

## 1. New: `research/` — paper-grade harness (entirely new code)

The Docker code is unsuitable for a paper sweep: it runs only once, uses 6
containers, has no DP accounting, no baselines, no multi-seed support, and
exposes only a single privacy metric. I added a clean **single-process**
research harness alongside it.

| Path | Purpose |
|---|---|
| `research/p3sl/model.py`        | Reference CNN identical to the Docker version, with `forward_upto` / `forward_from` cleanly separated. Adds `MAX_SPLIT_LAYER`, `DEFAULT_SPLIT_POINTS` constants. |
| `research/p3sl/dp.py`           | **Formal DP**. Laplace + Gaussian mechanisms with **per-sample clipping** (the Docker code adds noise without bounding sensitivity → no formal guarantee). Includes closed-form `laplace_epsilon`, `laplace_epsilon_composed`, and an RDP-based `gaussian_rdp_epsilon` accountant (Mironov 2017). |
| `research/p3sl/metrics.py`      | **Multi-metric privacy evaluation**: MSE, PSNR, SSIM, FSIM, LPIPS (optional), nearest-neighbour identifiability, MIA AUC, MIA balanced-acc, MIA TPR @ FPR=0.1, backdoor ASR. |
| `research/p3sl/attacks.py`      | Cleaned reimplementation of `InversionDecoder`, optimisation-based reconstruction, `MembershipInferenceClassifier`, plus a `FrozenClientHead` wrapper for the white-box adversary. All hyper-parameters (epochs / iterations / lr) are arguments, not magic numbers. |
| `research/p3sl/simulator.py`    | **The core trainer**. Supports four modes: `p3sl` (personalised SL + DP + bi-level optimisation), `vanilla_sl` (fixed split), `fedavg` (no split), `central` (upper bound). Also supports IID and Dirichlet-non-IID partitioning. Implements the bi-level σ-update from Section 5.2 of the paper (the Docker code had it commented out). |
| `research/run_experiments.py`   | Single entry point. Reads a YAML plan, sweeps over configs × seeds, runs attacks, writes per-run `history.csv` + `summary.json` + a top-level `all_runs.csv`. |
| `research/plot.py`              | Generates 5 publication PDFs: `utility_vs_round`, `privacy_utility_pareto`, `reconstruction_metrics`, `mia_metrics`, `backdoor_asr`. Mean ± std-dev shading across seeds. |
| `research/configs/smoke.yaml`   | 45-second pipeline self-test (sub-samples to 2k examples × 2 rounds). |
| `research/configs/main.yaml`    | Full benchmark — 9 configurations × 3 seeds (~30 min CPU, < 5 min GPU). |
| `research/requirements.txt`     | Pinned dependency floor: `torch>=2.1`, `torchvision`, `numpy`, `matplotlib`, `pyyaml`, `piq`. |
| `research/README.md`            | How to reproduce every figure/table the paper needs. |

I smoke-tested the harness end-to-end: `python research/run_experiments.py --config research/configs/smoke.yaml` runs to completion in ~45 s, produces three CSVs/JSONs, and `python research/plot.py --results research/results` produces all five PDF figures.

---

## 2. Bug-fixes and hardening in `my_implementation/P3SL_Implementation/1st_Demo/`

### `server/server.py`

| Old | New | Rationale |
|---|---|---|
| `# TARGET_ACCURACY = 90.0` (commented out) | `TARGET_ACCURACY = float(os.environ.get("P3SL_TARGET_ACCURACY", "80.0"))` | The bi-level loop referenced `TARGET_ACCURACY` but it had been commented out → silent NameError if the loop ran. Restored as an env-driven config. |
| `MAX_SPLIT_LAYER = 10` (hard-coded) | `int(os.environ.get("P3SL_MAX_SPLIT_LAYER", "10"))` | Configurable. |
| `TOTAL_ROUNDS = 1` | `int(os.environ.get("P3SL_TOTAL_ROUNDS", "30"))` | **The single most important fix.** With 1 round the model can't possibly converge — accuracy stayed at random init level. Default raised to 30 (~5 min on a CPU). |
| pretrain epochs hard-coded to `1` | `PRETRAIN_EPOCHS = int(os.environ.get("P3SL_PRETRAIN_EPOCHS", "5"))` | At 1 epoch the decoder/MIA never learn → reported FSIM/MIA scores are meaningless. Bumped to 5. |
| optimisation-attack iterations hard-coded to `50` | `OPT_ATTACK_ITERS = int(os.environ.get("P3SL_OPT_ATTACK_ITERS", "200"))` | 50 iterations produces near-noise reconstructions; 200 is the literature standard. |
| Bi-level optimisation loop commented out, replaced with a "linear decay" loop that always runs `TOTAL_ROUNDS` rounds and never checks accuracy | Bi-level loop **restored** with the geometric σ-decay specified by the paper, plus an early-exit when `acc ≥ TARGET_ACCURACY`. | The whole point of P3SL is the bi-level optimisation; it had been disabled. |
| CSV had 3 columns: `Batch, Decoder_FSIM, MIA_Confidence` | CSV now has 7 columns: `Batch, Decoder_FSIM, Decoder_SSIM, Decoder_PSNR, Decoder_MSE, Decoder_NN_ID, MIA_Confidence` | Reviewers always pick whichever metric matches their prior work — reporting only one risks cherry-picking. |

### `server/attacks/server_attacks.py`

| Old | New | Rationale |
|---|---|---|
| `def pretrain_hacker_decoder(decoder, known_client_model, epochs=1)` | default `epochs=5` plus updated docstring explaining override via env var | Bring defaults to publication-grade. |
| `def pretrain_hacker_mia(... , epochs=1)` | default `epochs=5` | Same. |
| `def optimization_attack(... , iterations=10)` | default `iterations=200` plus docstring | Same. |

### `server/attacks/metrics.py`

| Old | New | Rationale |
|---|---|---|
| only `calculate_fsim(orig, recon)` | adds `calculate_mse`, `calculate_psnr`, `calculate_ssim`, `calculate_nn_identifiability`, `calculate_all` (returns a dict). `calculate_fsim` API preserved for backward-compat. | Multi-metric privacy reporting required by reviewers. |
| FSIM crashes if `piq` is unavailable | falls back to SSIM with a warning | Robustness — pipeline keeps running on minimal installs. |

### `docker-compose.yml`

* Added an `environment:` block to the `server` service exposing
  `P3SL_TOTAL_ROUNDS`, `P3SL_TARGET_ACCURACY`, `P3SL_PRETRAIN_EPOCHS`,
  `P3SL_OPT_ATTACK_ITERS`, `P3SL_MAX_SPLIT_LAYER`, `PYTHONUNBUFFERED`. Override
  at the command line with e.g. `P3SL_TOTAL_ROUNDS=5 docker compose up`.

---

## 3. `my_implementation/docker-socket-demo/` fixes

### `dataset/dataset_downloader.py`

* Removed the hard-coded Windows path (`D:/projects/...`) which broke the
  helper on every non-Windows machine. Now uses `Path(__file__).resolve().parent`.

---

## 4. Replaced `my_implementation/DP-mechanisms/lapasian noise.ipynb`

The original notebook was one cell that called `np.random.laplace` with
`sensitivity=1000, ε=8` and printed the noise — it tells a reviewer
*nothing* about whether the mechanism gives a useful guarantee.

I replaced it with `laplacian_noise.ipynb` (5 sections, fully runnable):

1. **Single-coordinate Laplace mechanism** — formal definition with sensitivity Δ.
2. **Vector Laplace + L1 clipping** — exactly the mechanism used in `research/p3sl/dp.py::LaplaceMechanism`.
3. **Privacy/utility curve** — log-log plot of MSE(noisy − clipped) vs. ε. The standard plot in any DP paper.
4. **Sequential composition** — total ε under T-fold composition. Matches the number reported in the paper's privacy table.
5. **Take-aways** — explicitly contrasts the proper mechanism with the original toy snippet.

(File renamed from `lapasian noise.ipynb` → `laplacian_noise.ipynb` because (a) the original spelling was wrong and (b) the space made it harder to import or shell-quote.)

---

## How to use the new harness

```bash
# 1) install once
python -m venv .venv && source .venv/bin/activate
pip install -r research/requirements.txt

# 2) verify everything works (~45 s)
python research/run_experiments.py --config research/configs/smoke.yaml

# 3) full benchmark — 9 configurations × 3 seeds (~30 min CPU, < 5 min GPU)
python research/run_experiments.py --config research/configs/main.yaml

# 4) generate every paper figure
python research/plot.py --results research/results
```

Each completed run lives under `research/results/<run_name>/` with:

* `history.csv`  — per-round `accuracy, mean_loss, sigma_mean, mean_split, client0_split, ..., client4_split`
* `summary.json` — final metrics, ε accounting, attack scores, full config

Aggregate `research/results/all_runs.csv` is your Table-1 source.

---

## What the harness already supports for the paper

| Section of paper | Source artefact |
|---|---|
| Table 1 — final accuracy by mode | `all_runs.csv` (`final_accuracy`, mean ± std over seeds) |
| Table 2 — privacy budget ε | `summary.json` (`epsilon_total`) — closed-form for Laplace, RDP for Gaussian |
| Table 3 — attack metrics | `summary.json` (`decoder_metrics`, `optimization_metrics`, `mia`, `clean_accuracy`, `backdoor_asr`) |
| Fig. 1 — utility vs round | `figures/utility_vs_round.pdf` (mean curve + std-dev band) |
| Fig. 2 — privacy-utility Pareto | `figures/privacy_utility_pareto.pdf` (log-ε scatter, all baselines vs P3SL) |
| Fig. 3 — reconstruction quality | `figures/reconstruction_metrics.pdf` (5-metric bar chart) |
| Fig. 4 — MIA + Backdoor | `figures/mia_metrics.pdf` + `figures/backdoor_asr.pdf` |

---

## Suggestions for further work (if you want to push beyond the current scope)

These are not implemented but are easy follow-ups now that the harness exists:

1. **CIFAR-10 cross-check** — already supported via `dataset: cifar10` in the YAML; just bump the model's `in_channels=3`. Adds a second dataset for stronger external validity.
2. **Real energy measurements** — the paper currently uses a *fabricated* energy table (`{1: 903, 2: 951, ...}`). Replace `default_energy_table` in `simulator.py` with measured energy via PyTorch FLOP counting (e.g. `fvcore.nn.FlopCountAnalysis`).
3. **DP-SGD baseline** — add `mode: dp_sgd` using `opacus` (`pip install opacus`) for a head-to-head comparison vs the gold-standard private learning method.
4. **Larger compositional ε** — for paper purposes, run all configurations at three ε levels (e.g. `{1, 4, 16}`) so you have a 3×N grid.
5. **Robustness under label-skew non-IID** — the harness already supports Dirichlet partitioning (`iid: false, dirichlet_alpha: 0.1`); add a row to `main.yaml`.
6. **Visual reconstruction galleries** — currently skipped to keep runs fast. Add `save_image(comparison, …)` calls in `_run_attacks`.
