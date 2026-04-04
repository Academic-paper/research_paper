# Research Paper — Split Learning Implementations

Implementations for personalized privacy-preserving split learning (P3SL), server-side attacks, and related experiments.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Optional: NVIDIA GPU + container toolkit for CUDA images (CPU works for smoke tests)

## Primary demo: P3SL multi-client (`1st_Demo`)

Location: [`my_implementation/P3SL_Implementation/1st_Demo/`](my_implementation/P3SL_Implementation/1st_Demo/)

### Architecture

- **Server** ([`server/server.py`](my_implementation/P3SL_Implementation/1st_Demo/server/server.py)): orchestrates 5 clients, bi-level noise table (target accuracy vs privacy), FashionMNIST evaluation, optional inversion/MIA attacks.
- **Client 1** ([`client1/client1.py`](my_implementation/P3SL_Implementation/1st_Demo/client1/client1.py)): malicious client (poisoning, model extraction).
- **Clients 2–5**: honest clients with different `ALPHA` privacy/energy trade-offs (0.2, 0.5, 0.9, 0.7).

### Run

```bash
cd my_implementation/P3SL_Implementation/1st_Demo
mkdir -p dataset/data
docker compose up --build
```

On first run, FashionMNIST is downloaded into `./dataset/data` (mounted at `/data` in containers). See [`dataset/README.md`](my_implementation/P3SL_Implementation/1st_Demo/dataset/README.md).

### Dependencies

Docker images use `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`; [`server/requirements.txt`](my_implementation/P3SL_Implementation/1st_Demo/server/requirements.txt) adds `matplotlib` and `piq` only. For local (non-Docker) runs, install PyTorch separately or use [`requirements-local.txt`](my_implementation/P3SL_Implementation/1st_Demo/requirements-local.txt).

### Security note

Demos bind `0.0.0.0:5000` and exchange messages with `pickle` over TCP. Use only in isolated lab environments.

## Secondary demo: docker-socket-demo

Two-container MNIST split-learning attack benchmark:

```bash
cd my_implementation/docker-socket-demo
docker compose up --build
```

Dataset: run [`dataset/dataset_downloader.py`](my_implementation/docker-socket-demo/dataset/dataset_downloader.py) or let torchvision download on first client start.

## Notebooks and utilities

- [`my_implementation/split_learning/`](my_implementation/split_learning/) — split learning experiments and socket utilities
- [`my_implementation/DP-mechanisms/`](my_implementation/DP-mechanisms/) — differential privacy (Laplacian noise) notebook

## Verify Python syntax

```bash
./scripts/check_python.sh
```
