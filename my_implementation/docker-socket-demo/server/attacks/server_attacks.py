import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# ATTACK 1: Whitebox Inversion Decoder
# ==========================================
class InversionDecoder(nn.Module):
    def __init__(self, ir_size=64): 
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(ir_size, 256),
            nn.ReLU(),
            nn.Linear(256, 784), 
            nn.Sigmoid()         
        )

    def forward(self, intercepted_ir):
        flattened_reconstruction = self.decode(intercepted_ir)
        return flattened_reconstruction.view(-1, 1, 28, 28)

def pretrain_hacker_decoder(decoder, known_client_model, epochs=5):
    """
    The Server downloads its own copy of MNIST and trains the decoder offline
    to map the Client's architecture output back to the original images.
    """
    print("\n[HACKER MODULE] Pre-training Decoder on Public Data...")
    
    # Standard MNIST dataset for the hacker to learn from
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    hacker_dataset = datasets.MNIST("./dataset", download=True, train=True, transform=transform)
    hacker_loader = DataLoader(hacker_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(decoder.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    known_client_model.eval()

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in hacker_loader:
            images_flat = images.view(images.shape[0], -1)
            
            # 1. Server generates "fake" intercepted IR using the known client architecture
            with torch.no_grad():
                simulated_ir = known_client_model(images_flat)
            
            # 2. Train decoder to reconstruct the original image from the IR
            optimizer.zero_grad()
            reconstructed = decoder(simulated_ir)
            
            # Un-normalize real images to 0-1 for accurate MSE loss against Sigmoid output
            real_images_01 = (images * 0.5) + 0.5
            
            loss = criterion(reconstructed, real_images_01)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f" -> Pre-training Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(hacker_loader):.4f}")
    
    print("[HACKER MODULE] Pre-training Complete. Decoder is armed and ready.\n")


# ==========================================
# ATTACK 2: Optimization-Based Inversion (UPGRADED)
# ==========================================
def total_variation_loss(img):
    """
    Forces adjacent pixels to be similar. This removes the "static" 
    and forces the optimizer to create smooth, contiguous shapes.
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def optimization_attack(intercepted_ir, known_client_model, iterations=500):
    # START WITH RANDOM NOISE, not zeros. This prevents "dead ReLUs".
    dummy_image = torch.randn((intercepted_ir.size(0), 784), requires_grad=True)
    
    # Use a slightly higher learning rate
    optimizer = optim.Adam([dummy_image], lr=0.1)
    criterion = nn.MSELoss()
    known_client_model.eval()

    for i in range(iterations):
        optimizer.zero_grad()
        dummy_ir = known_client_model(dummy_image)
        ir_loss = criterion(dummy_ir, intercepted_ir.detach())
        
        dummy_image_2d = dummy_image.view(-1, 1, 28, 28)
        # SLASH the TV loss weight from 0.1 to 0.005. 
        # It should gently smooth the image, not blur it aggressively.
        tv_loss = total_variation_loss(dummy_image_2d) * 0.005 
        
        total_loss = ir_loss + tv_loss
        total_loss.backward()
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

def pretrain_hacker_mia(mia_classifier, known_client_model, epochs=5):
    """
    Trains the MIA using a Shadow Model approach.
    It teaches the classifier to distinguish between data the model 
    has been trained on (1) and data it has never seen (0).
    """
    print("\n[HACKER MODULE] Pre-training MIA Shadow Classifier...")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST("./dataset", download=True, train=True, transform=transform)
    
    # Split the dataset into a Shadow "IN" set and Shadow "OUT" set
    in_set, out_set = torch.utils.data.random_split(dataset, [30000, 30000])
    in_loader = DataLoader(in_set, batch_size=128, shuffle=True)
    out_loader = DataLoader(out_set, batch_size=128, shuffle=True)

    # 1. Briefly train the Shadow Model (known_client_model) on the "IN" set
    # so it behaves like a real, trained client model.
    dummy_optim = optim.Adam(known_client_model.parameters(), lr=0.001)
    dummy_criterion = nn.CrossEntropyLoss()
    dummy_head = nn.Linear(64, 10) # Fake server-side head for shadow training
    head_optim = optim.Adam(dummy_head.parameters(), lr=0.001)

    print(" -> Phase 1: Training Shadow Model on 'IN' data...")
    for _ in range(2): # 2 quick epochs is enough to cause memorization
        for imgs, labels in in_loader:
            imgs_flat = imgs.view(imgs.shape[0], -1)
            
            dummy_optim.zero_grad()
            head_optim.zero_grad()
            
            irs = known_client_model(imgs_flat)
            preds = dummy_head(irs)
            
            loss = dummy_criterion(preds, labels)
            loss.backward()
            dummy_optim.step()
            head_optim.step()

    # 2. Train the MIA Classifier to tell the difference between IN and OUT
    mia_optim = optim.Adam(mia_classifier.parameters(), lr=0.005)
    mia_criterion = nn.BCELoss() # Binary Cross Entropy (1 or 0)

    print(" -> Phase 2: Training Attack Classifier to spot 'IN' vs 'OUT' IRs...")
    for epoch in range(epochs):
        total_loss = 0
        in_iter = iter(in_loader)
        out_iter = iter(out_loader)

        # Loop through both datasets simultaneously
        for _ in range(min(len(in_loader), len(out_loader))):
            in_imgs, _ = next(in_iter)
            out_imgs, _ = next(out_iter)

            in_imgs_flat = in_imgs.view(in_imgs.shape[0], -1)
            out_imgs_flat = out_imgs.view(out_imgs.shape[0], -1)

            # Generate IRs using the Shadow Model
            with torch.no_grad():
                in_irs = known_client_model(in_imgs_flat)
                out_irs = known_client_model(out_imgs_flat)

            # Assign labels: 1 for data in the training set, 0 for unseen data
            in_labels = torch.ones(in_irs.size(0), 1)
            out_labels = torch.zeros(out_irs.size(0), 1)

            # Combine them into one batch
            X = torch.cat([in_irs, out_irs])
            Y = torch.cat([in_labels, out_labels])

            # Train the MIA model
            mia_optim.zero_grad()
            preds = mia_classifier(X)
            loss = mia_criterion(preds, Y)
            loss.backward()
            mia_optim.step()
            total_loss += loss.item()

        print(f"   MIA Epoch {epoch+1}/{epochs} | Loss: {total_loss/(len(in_loader)):.4f}")
        
    print("[HACKER MODULE] MIA Pre-training Complete.\n")