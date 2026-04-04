"""Prefetch FashionMNIST for the P3SL demo (optional; Docker also downloads on first run)."""
from pathlib import Path

from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent / "data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.FashionMNIST(ROOT, train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(ROOT, train=False, download=True, transform=transform)

_ = train_ds[0]
_ = test_ds[0]

print(f"FashionMNIST ready under {ROOT}")
