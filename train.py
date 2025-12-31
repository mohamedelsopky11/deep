import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FireDataset
from model import get_model
import os
import time

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Batch {i}: Loss {loss.item():.4f}")
            
    return running_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    BATCH_SIZE = 4 # Small batch for CPU/testing
    LR = 1e-4
    EPOCHS = 1 # Just 1 for demo
    
    # Dataset
    root = r"d:/Programming/Python/CV Project/FireData/Fire and Smoke Dataset"
    # Use fast_mode=True for speed validation
    train_ds = FireDataset(root, split='train', img_size=(256, 256), fast_mode=True)
    
    # Subsample for demo speed (use only first 50 images)
    train_ds.image_paths = train_ds.image_paths[:50]
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = get_model(num_classes=3, device=device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - Time: {time.time()-start:.0f}s")
        
    # Save
    save_path = "models/fire_seg_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
