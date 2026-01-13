import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K

# Import from local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import BrainTumorDataset
from model import get_model

# --- GPU Augmentations (Kornia) ---
# We define a container for GPU augmentations
class GPUAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        # Kornia transforms work on tensors [B, C, H, W]
        # These will run on the GPU (much faster than CPU ElasticTransform)
        self.aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.2),
            K.RandomAffine(degrees=30.0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),
            # Elastic Transform on GPU!
            K.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(1.0, 1.0), p=0.2),
            K.ColorJitter(brightness=0.2, contrast=0.2, p=0.2),
        )

    def forward(self, input_tensor):
        return self.aug(input_tensor)

# --- Standard Normalization (Dataset Level) ---
BRAIN_MEAN = [0.1854, 0.1854, 0.1855]
BRAIN_STD = [0.1855, 0.1855, 0.1855]

def get_base_transforms():
    """
    Minimal CPU transforms: Just Resize/Normalize/ToTensor.
    Complex stuff moves to GPU.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=BRAIN_MEAN, std=BRAIN_STD),
        ToTensorV2()
    ])

# --- TTA Helper ---
def tta_inference(model, images):
    """
    Test Time Augmentation:
    1. Predict standard image
    2. Predict horizontally flipped
    3. Predict vertically flipped
    4. Average predictions
    """
    # 1. Standard
    out1 = model(images)
    
    # 2. H-Flip
    images_h = torch.flip(images, dims=[3]) # [B, C, H, W] -> flip W
    out2 = model(images_h)
    
    # 3. V-Flip (Optional, maybe risky if orientation matters, but we used it in training)
    # Let's stick to H-Flip + maybe slight intensity noise? 
    # For now, just H-Flip is the safest high-yield TTA.
    # out3 = model(torch.flip(images, dims=[2]))
    
    # Average logits (or softmax probabilities)
    # Averaging logits is usually fine
    return (out1 + out2) / 2.0

def train_epoch(model, loader, criterion, optimizer, device, gpu_aug=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Apply GPU Augmentations if provided
        if gpu_aug is not None:
            # Kornia expects 0-1 range usually if not normalized, but here we already normalized in Dataset.
            # Kornia works fine on normalized data for geometric transforms. 
            # For ColorJitter, it expects proper range, but robust implementations handle it.
            # Note: Applying color jitter on normalized data might behave slightly differently 
            # but is generally acceptable or we denorm -> aug -> norm. 
            # For speed, we just apply geometric directly.
            with torch.no_grad():
                images = gpu_aug(images)
        
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

def eval_epoch(model, loader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_tta:
                outputs = tta_inference(model, images)
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    return running_loss / total, correct / total

def get_all_data(data_dir):
    classes = sorted(['glioma', 'meningioma', 'notumor', 'pituitary'])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    all_paths = []
    all_labels = []
    
    subfolders = ['Training', 'Testing']
    for sub in subfolders:
        for cls_name in classes:
            files = glob(os.path.join(data_dir, sub, cls_name, '*'))
            for f in files:
                all_paths.append(f)
                all_labels.append(class_to_idx[cls_name])
    
    return np.array(all_paths), np.array(all_labels)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    print(f"Features Enabled: LabelSmoothing={args.smoothing}, TTA={args.tta}, GPU_Aug=Kornia")
    
    all_paths, all_labels = get_all_data(args.data_dir)
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    gpu_aug = GPUAugmentations().to(device)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_paths, all_labels)):
        print(f"\nFOLD {fold+1}/{args.k_folds}")
        
        train_dataset = BrainTumorDataset(image_paths=all_paths[train_ids], labels=all_labels[train_ids], 
                                          transform=get_base_transforms(), phase='train')
        val_dataset = BrainTumorDataset(image_paths=all_paths[val_ids], labels=all_labels[val_ids], 
                                        transform=get_base_transforms(), phase='val')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
        
        model = get_model(model_name=args.model_name, num_classes=4, pretrained=True).to(device)
        
        # LABEL SMOOTHING: Helps generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_fold_acc = 0.0
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, gpu_aug=gpu_aug)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, use_tta=args.tta)
            scheduler.step()
            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
            
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"  Epoch {epoch+1}/{args.epochs} | Train: {train_acc:.4f} | Val (TTA={args.tta}): {val_acc:.4f}")
        
        print(f"  >> Best Acc Fold {fold+1}: {best_fold_acc:.4f}")
        fold_results.append(best_fold_acc)
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS (Advanced)")
    print(f"Avg Accuracy: {np.mean(fold_results)*100:.2f}% (+/- {np.std(fold_results)*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing value (0.0 to 0.1)')
    parser.add_argument('--tta', action='store_true', help='Enable Test Time Augmentation')
    args = parser.parse_args()
    
    main(args)
