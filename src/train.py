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

BRAIN_MEAN = [0.1854, 0.1854, 0.1855]
BRAIN_STD = [0.1855, 0.1855, 0.1855]

def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=BRAIN_MEAN, std=BRAIN_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=BRAIN_MEAN, std=BRAIN_STD),
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
    model = get_model(model_name=args.model_name, num_classes=4, pretrained=True)
    model = model.to(device)
    
    # 3. Optimization
    # Calculate class weights for imbalance
    class_counts = [1321, 1339, 1595, 1457] # glioma, meningioma, notumor, pituitary
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * x) for x in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW often better
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_brain_tumor_model.pth")
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    main(args)
