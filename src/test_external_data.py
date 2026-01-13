import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import BrainTumorDataset
from model import get_model

# --- Config ---
BRAIN_MEAN = [0.1854, 0.1854, 0.1855]
BRAIN_STD = [0.1855, 0.1855, 0.1855]
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=BRAIN_MEAN, std=BRAIN_STD),
        ToTensorV2()
    ])

def test_external(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 1. Load Data
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        print("Please create it and organize images as: folder/class_name/image.jpg")
        return

    print(f"Loading External Data from: {args.data_dir}")
    dataset = BrainTumorDataset(root_dir=args.data_dir, transform=get_val_transform(), phase='test')
    
    if len(dataset) == 0:
        print("No images found! Make sure they are in subfolders named: glioma, meningioma, notumor, pituitary")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load Model
    print(f"Loading Model: {args.model_path}")
    model = get_model(model_name='efficientnet_b0', num_classes=4, pretrained=False)
    
    try:
        # Determine if we have a state_dict or full checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model = model.to(device)
    model.eval()

    # 3. Run Inference
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Report
    print("\n" + "="*50)
    print("EXTERNAL VALIDATION RESULTS")
    print("="*50)
    
    # Accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("-" * 30)
    
    # Detailed Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(f'External Validation (Acc: {acc*100:.1f}%)')
    
    # Save correctly
    os.makedirs("results", exist_ok=True)
    save_path = "results/external_validation_matrix.png"
    plt.savefig(save_path)
    print(f"\nConfusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to external dataset root (containing class subfolders)')
    parser.add_argument('--model_path', type=str, default='best_brain_tumor_model.pth', help='Path to .pth model file')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    test_external(args)
