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

def pretrain_hacker_decoder(decoder, known_client_model, epochs=1):# increase to 10 to 15 for research
    print("\n[HACKER MODULE] Pre-training Decoder on Public Data...")
    device = next(known_client_model.parameters()).device
    decoder.to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    hacker_dataset = datasets.FashionMNIST("/data", download=True, train=True, transform=transform)
    hacker_loader = DataLoader(hacker_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(decoder.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    known_client_model.eval()

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in hacker_loader:
            images = images.to(device) # EXACT CNN INPUT SHAPE: [B, 1, 28, 28]
            
            with torch.no_grad():
                raw_ir = known_client_model(images)
            
            # Pool down to 64 for the fast Decoder baseline
            raw_ir_flat = raw_ir.view(raw_ir.size(0), -1)
            if raw_ir_flat.size(1) != 64:
                pooler = nn.AdaptiveAvgPool1d(64).to(device)
                simulated_ir = pooler(raw_ir_flat.unsqueeze(1)).squeeze(1)
            else:
                simulated_ir = raw_ir_flat
                
            optimizer.zero_grad()
            reconstructed = decoder(simulated_ir)
            
            real_images_01 = (images * 0.5) + 0.5
            loss = criterion(reconstructed, real_images_01)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f" -> Pre-training Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(hacker_loader):.4f}")
    
    print("[HACKER MODULE] Pre-training Complete. Decoder is armed and ready.\n")

# ==========================================
# ATTACK 2: Optimization-Based Inversion (PURE WHITE-BOX)
# ==========================================
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def optimization_attack(intercepted_ir, known_client_model, iterations=10):# increase to 500 for research
    device = intercepted_ir.device
    
    # EXACT WHITEBOX: Dummy image is initialized as a 2D spatial tensor
    dummy_image = torch.randn((intercepted_ir.size(0), 1, 28, 28), requires_grad=True, device=device)
    
    optimizer = optim.Adam([dummy_image], lr=0.1)
    criterion = nn.MSELoss()
    known_client_model.eval()

    for i in range(iterations):
        optimizer.zero_grad()
        
        # Passes the 2D image directly into the CNN exact replica
        dummy_ir = known_client_model(dummy_image)
        ir_loss = criterion(dummy_ir, intercepted_ir.detach())
        
        tv_loss = total_variation_loss(dummy_image) * 0.005 
        
        total_loss = ir_loss + tv_loss
        total_loss.backward()
        optimizer.step()
        
    return torch.clamp(dummy_image, 0, 1)

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

def pretrain_hacker_mia(mia_classifier, known_client_model, epochs=1):# increase to 10 to 15 for research
    print("\n[HACKER MODULE] Pre-training MIA Shadow Classifier...")
    device = next(known_client_model.parameters()).device
    mia_classifier.to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST("/data", download=True, train=True, transform=transform)
    
    in_set, out_set = torch.utils.data.random_split(dataset, [30000, 30000])
    in_loader = DataLoader(in_set, batch_size=128, shuffle=True)
    out_loader = DataLoader(out_set, batch_size=128, shuffle=True)

    dummy_optim = optim.Adam(known_client_model.parameters(), lr=0.001)
    dummy_criterion = nn.CrossEntropyLoss()
    dummy_head = nn.Linear(64, 10).to(device)
    head_optim = optim.Adam(dummy_head.parameters(), lr=0.001)

    print(" -> Phase 1: Training Shadow Model on 'IN' data...")
    for _ in range(2): 
        for imgs, labels in in_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            dummy_optim.zero_grad()
            head_optim.zero_grad()
            
            raw_irs = known_client_model(imgs)
            
            # Fast pooler for the classification head
            raw_irs_flat = raw_irs.view(raw_irs.size(0), -1)
            if raw_irs_flat.size(1) != 64:
                pooler = nn.AdaptiveAvgPool1d(64).to(device)
                irs = pooler(raw_irs_flat.unsqueeze(1)).squeeze(1)
            else:
                irs = raw_irs_flat
                
            preds = dummy_head(irs)
            loss = dummy_criterion(preds, labels)
            loss.backward()
            dummy_optim.step()
            head_optim.step()

    mia_optim = optim.Adam(mia_classifier.parameters(), lr=0.005)
    mia_criterion = nn.BCELoss() 

    print(" -> Phase 2: Training Attack Classifier to spot 'IN' vs 'OUT' IRs...")
    for epoch in range(epochs):
        total_loss = 0
        in_iter = iter(in_loader)
        out_iter = iter(out_loader)

        for _ in range(min(len(in_loader), len(out_loader))):
            in_imgs, _ = next(in_iter)
            out_imgs, _ = next(out_iter)

            in_imgs = in_imgs.to(device)
            out_imgs = out_imgs.to(device)

            with torch.no_grad():
                raw_in_irs = known_client_model(in_imgs)
                raw_out_irs = known_client_model(out_imgs)
            
            # Pool to 64 for MIA
            pooler = nn.AdaptiveAvgPool1d(64).to(device)
            in_irs = pooler(raw_in_irs.view(raw_in_irs.size(0), -1).unsqueeze(1)).squeeze(1)
            out_irs = pooler(raw_out_irs.view(raw_out_irs.size(0), -1).unsqueeze(1)).squeeze(1)

            in_labels = torch.ones(in_irs.size(0), 1).to(device)
            out_labels = torch.zeros(out_irs.size(0), 1).to(device)

            X = torch.cat([in_irs, out_irs])
            Y = torch.cat([in_labels, out_labels])

            mia_optim.zero_grad()
            preds = mia_classifier(X)
            loss = mia_criterion(preds, Y)
            loss.backward()
            mia_optim.step()
            total_loss += loss.item()

        print(f"   MIA Epoch {epoch+1}/{epochs} | Loss: {total_loss/(len(in_loader)):.4f}")
        
    print("[HACKER MODULE] MIA Pre-training Complete.\n")