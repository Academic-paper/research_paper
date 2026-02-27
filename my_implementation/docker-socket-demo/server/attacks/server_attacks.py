import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# ATTACK 1: Whitebox Inversion Decoder (MLP Version)
# ==========================================
class InversionDecoder(nn.Module):
    """
    The Server trains this reverse-network to upscale the 
    intercepted 64-length IR back into a 28x28 image.
    """
    def __init__(self, ir_size=64): 
        super().__init__()
        # Reverse the Client's Linear Layers
        self.decode = nn.Sequential(
            nn.Linear(ir_size, 128),
            nn.ReLU(),
            nn.Linear(128, 784), # 784 is 28x28 flattened
            nn.Sigmoid()         # Force pixel values between 0 and 1
        )

    def forward(self, intercepted_ir):
        # 1. Upscale from 64 back to 784
        flattened_reconstruction = self.decode(intercepted_ir)
        # 2. Reshape into an actual image [Batch, Channel, Height, Width]
        image_shape = flattened_reconstruction.view(-1, 1, 28, 28)
        return image_shape

# ==========================================
# ATTACK 2: Optimization-Based Inversion
# ==========================================
def optimization_attack(intercepted_ir, known_client_model, iterations=500):
    # Dummy image starts as flattened 784 to match client input
    dummy_image = torch.randn((intercepted_ir.size(0), 784), requires_grad=True)
    optimizer = optim.Adam([dummy_image], lr=0.1)
    criterion = nn.MSELoss()

    known_client_model.eval()

    for i in range(iterations):
        optimizer.zero_grad()
        dummy_ir = known_client_model(dummy_image)
        loss = criterion(dummy_ir, intercepted_ir.detach())
        loss.backward()
        optimizer.step()
        
    return torch.clamp(dummy_image.view(-1, 1, 28, 28), 0, 1)

# ==========================================
# ATTACK 3: Membership Inference Attack (MIA)
# ==========================================
class MembershipInferenceClassifier(nn.Module):
    def __init__(self, ir_size=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ir_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, intercepted_ir):
        return self.classifier(intercepted_ir)