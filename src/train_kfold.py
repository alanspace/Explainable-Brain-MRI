import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from local modules (assuming running from src or adding src to path)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import BrainTumorDataset
from model import get_model

# --- Reusing Transforms and Helper Functions ---

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
            A.VideoReader(p=0) if False else A.NoOp(), # Placeholder to avoid comma errors if needed, but better:
            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2), # Too slow for quick training
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
    
    for images, labels in loader:
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
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    return running_loss / total, correct / total

def get_all_data(data_dir):
    """
    Aggregates data from both 'Training' and 'Testing' folders
    to create a single dataset for Cross-Validation.
    """
    classes = sorted(['glioma', 'meningioma', 'notumor', 'pituitary'])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    
    # Check both subfolders
    subfolders = ['Training', 'Testing']
    
    print(f"Scanning data from {data_dir}...")
    
    for sub in subfolders:
        full_sub_path = os.path.join(data_dir, sub)
        if not os.path.isdir(full_sub_path):
            print(f"  Warning: {sub} folder not found in {data_dir}")
            continue
            
        for cls_name in classes:
            class_dir = os.path.join(full_sub_path, cls_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Grab images
            valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
            for ext in valid_exts:
                files = glob(os.path.join(class_dir, ext))
                for f in files:
                    all_paths.append(f)
                    all_labels.append(class_to_idx[cls_name])
    
    print(f"Total images found: {len(all_paths)}")
    return np.array(all_paths), np.array(all_labels)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 1. Prepare Data
    all_paths, all_labels = get_all_data(args.data_dir)
    
    if len(all_paths) == 0:
        print("Error: No images found. Check data path.")
        return

    # 2. K-Fold Cross Validation
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"\nStarting {args.k_folds}-Fold Cross-Validation...")
    print("="*50)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_paths, all_labels)):
        print(f"\nFOLD {fold+1}/{args.k_folds}")
        print("-" * 20)
        
        # Split data
        train_paths, val_paths = all_paths[train_ids], all_paths[val_ids]
        train_labels, val_labels = all_labels[train_ids], all_labels[val_ids]
        
        # Create Datasets
        train_dataset = BrainTumorDataset(image_paths=train_paths, labels=train_labels, 
                                          transform=get_transforms('train'), phase='train')
        val_dataset = BrainTumorDataset(image_paths=val_paths, labels=val_labels, 
                                        transform=get_transforms('val'), phase='val')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
        
        # Initialize Model (New model for each fold!)
        model = get_model(model_name=args.model_name, num_classes=4, pretrained=True)
        model = model.to(device)
        
        # Loss and Optimizer
        # Calculate weights based on THIS fold's training data if strictly needed, 
        # but using global or uniform weights is often fine. We'll stick to uniform for simplicity 
        # or re-calc if we want perfection.
        # Let's simple use the weights from the main script for consistency if we wanted, 
        # but here we'll let the optimizer handle it or basic CrossEntropy. 
        # Adding weights if specific class imbalance is severe.
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_fold_acc = 0.0
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                # Optional: Save best model for this fold
                # torch.save(model.state_dict(), f"results/model_fold_{fold+1}.pth")
            
            # Print only last epoch or every 5 to reduce spam
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"  Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        print(f"  >> Best Acc used for Fold {fold+1}: {best_fold_acc:.4f}")
        fold_results.append(best_fold_acc)
        
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for i, score in enumerate(fold_results):
        print(f"Fold {i+1}: {score*100:.2f}%")
        
    avg_acc = np.mean(fold_results) * 100
    std_acc = np.std(fold_results) * 100
    
    print(f"\nAverage Accuracy: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
    print("="*50)
    
    # Save results to text file
    os.makedirs("results", exist_ok=True)
    with open("results/kfold_results.txt", "w") as f:
        f.write(f"{args.k_folds}-Fold Cross Validation Results\n")
        f.write("="*30 + "\n")
        for i, score in enumerate(fold_results):
            f.write(f"Fold {i+1}: {score*100:.2f}%\n")
        f.write(f"\nAverage: {avg_acc:.2f}%\n")
        f.write(f"Std Dev: {std_acc:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k_folds', type=int, default=5)
    args = parser.parse_args()
    
    main(args)
