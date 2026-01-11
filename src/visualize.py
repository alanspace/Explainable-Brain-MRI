import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from model import get_model
from gradcam import GradCAM

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img

BRAIN_MEAN = [0.1854, 0.1854, 0.1855]
BRAIN_STD = [0.1855, 0.1855, 0.1855]

def preprocess(img, device):
    # Normalize like in training
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - BRAIN_MEAN) / BRAIN_STD
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
    return img_tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = get_model(model_name='efficientnet_b0', num_classes=4, pretrained=False) # Architecture only
    model_path = "best_brain_tumor_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run training first.")
        return
 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
 
    # Target Layer: Last layer of features for EfficientNet
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    # Classes
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Output dir
    os.makedirs("results", exist_ok=True)

    # Process one image per class
    test_dir = "data/Testing"
    if not os.path.exists(test_dir):
        # Fallback to Training if Testing missing
        test_dir = "data/Training"

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images = glob(os.path.join(cls_dir, "*"))
        if not images:
            print(f"No images found for {cls}")
            continue
        
        # Try to find a correct prediction
        found = False
        target_idx = classes.index(cls)
        
        # Shuffle images to get different likely correct ones if we run multiple times
        # But for reproducibility let's just iterate
        for img_path in images[:100]: # Check first 100 images
            original_img = load_image(img_path)
            input_tensor = preprocess(original_img, device)
            
            # Predict first to check correctness
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
            
            if pred_idx == target_idx:
                # Correct prediction! Generate Grad-CAM
                heatmap, _ = cam.forward(input_tensor, class_idx=target_idx)
                
                # Overlay
                overlay = cam.overlay_heatmap(heatmap, original_image=original_img, alpha=0.5)
                
                # Save side-by-side
                # Add text label
                combined = np.hstack((original_img, overlay))
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                
                # Draw text
                cv2.putText(combined, f"True: {cls}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(combined, f"Pred: {classes[pred_idx]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                save_path = f"results/{cls}_explanation.png"
                cv2.imwrite(save_path, combined)
                print(f"Saved CORRECT explanation for {cls} to {save_path}")
                found = True
                break
        
        if not found:
             print(f"Could not find a correctly classified example for {cls} in first 20 images.")

if __name__ == "__main__":
    main()
