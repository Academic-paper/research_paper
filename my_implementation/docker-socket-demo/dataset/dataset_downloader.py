from torchvision import datasets
import torchvision.transforms as transforms
from pathlib import Path

root = Path(__file__).resolve().parent

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ALLOW download ONCE
train_ds = datasets.MNIST(
    root=root,
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.MNIST(
    root=root,
    train=False,
    download=True,
    transform=transform
)

# FORCE processing
_ = train_ds[0]
_ = test_ds[0]

print("MNIST processed files created.")
