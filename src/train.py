import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BrainTumorDataset
from model import get_model

def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]),
            ToTensorV2()
        ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    return running_loss / total, correct / total

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 1. Datasets
    # If separate train/test folders exist, use them. Otherwise, we might need a split function.
    # For this script, we assume args.data_dir points to parent folder containing 'Training' and 'Testing' usually found in Kaggle datasets.
    # If not, point directly to the specific folders using args.
    
    train_dir = os.path.join(args.data_dir, 'Training')
    test_dir = os.path.join(args.data_dir, 'Testing')
    
    if not os.path.isdir(train_dir):
        # Fallback: Maybe the user passed the training root directly
        train_dir = args.data_dir
        print(f"Note: 'Training' subfolder not found. Using {train_dir} as Training root.")
        # If no testing folder, we might just validate on training (not ideal) or split.
        # For simplicity in this v1 script, we assume the user provides a valid dir.

    train_dataset = BrainTumorDataset(train_dir, transform=get_transforms('train'), phase='train')
    if os.path.isdir(test_dir):
        val_dataset = BrainTumorDataset(test_dir, transform=get_transforms('val'), phase='val')
    else:
        print("Warning: No 'Testing' directory found. Using Training data for validation (Debugging mode).")
        val_dataset = BrainTumorDataset(train_dir, transform=get_transforms('val'), phase='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Model
    model = get_model(num_classes=4, pretrained=True)
    model = model.to(device)
    
    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # Only training head initially if frozen
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_brain_tumor_model.pth")
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root (containing Training/Testing subdirs or class folders)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    main(args)
