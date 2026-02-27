import torch
from piq import fsim

def calculate_fsim(original, reconstructed):
    """
    Grades the reconstruction. 1.0 = High Leakage, 0.0 = Perfect Privacy.
    """
    orig = torch.clamp(original, 0.0, 1.0)
    recon = torch.clamp(reconstructed, 0.0, 1.0)
    
    # ADD chromatic=False to support 1-channel Grayscale images like MNIST
    return fsim(orig, recon, data_range=1.0, chromatic=False).item()