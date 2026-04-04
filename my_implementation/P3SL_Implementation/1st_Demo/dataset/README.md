# P3SL dataset directory

Docker Compose mounts `./dataset/data` to `/data` in all containers.

## First run

You do not need to commit data files. On `docker compose up`, the server and attack modules download **FashionMNIST** via torchvision (`download=True`) into this folder.

To prefetch locally (optional):

```bash
cd my_implementation/P3SL_Implementation/1st_Demo
python dataset/dataset_downloader.py
```

This creates `dataset/data/FashionMNIST/` with processed files.

## Git

Contents under `dataset/data/` are gitignored. Only this README and `dataset_downloader.py` are tracked.
