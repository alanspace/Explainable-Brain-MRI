import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from model import get_model
from gradcam import GradCAM

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img

def preprocess(img, device):
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
    return img_tensor

def add_noise(image, noise_type="gaussian"):
    """
    Adds noise to an image to test robustness.
    """
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.01  # Variance
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image.astype(np.float32) / 255.0 + gauss
        noisy = np.clip(noisy, 0, 1)
        return (noisy * 255).astype(np.uint8)
    return image

def rotate_image(image, angle=15):
    """
    Rotates the image by a small angle to test geometric robustness.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_similarity(heatmap1, heatmap2):
    """
    Computes Cosine Similarity between two heatmaps (flattened).
    """
    h1 = heatmap1.flatten()
    h2 = heatmap2.flatten()
    
    # Normalize
    if np.linalg.norm(h1) == 0 or np.linalg.norm(h2) == 0:
        return 0.0
        
    cos_sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
    return cos_sim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device for Robustness Check: {device}")
    
    # Load Model
    model = get_model(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load("best_brain_tumor_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    test_dir = "data/Testing"
    if not os.path.exists(test_dir):
        test_dir = "data/Training"

    print("\n--- Starting Robustness Evaluation ---")
    print("Hypothesis: Small perturbations (noise/rotation) should NOT drastically change the explanation.\n")
    
    results = {"noise": [], "rotation": []}
    
    # Sample 10 images per class for speed
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images = glob(os.path.join(cls_dir, "*"))
        if len(images) > 10:
            images = images[:10]
            
        for img_path in images:
            original_img = load_image(img_path)
            input_tensor = preprocess(original_img, device)
            
            # Baseline Heatmap
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
            
            heatmap_base, _ = cam.forward(input_tensor, class_idx=pred_idx)
            
            # 1. Noise Test
            noisy_img = add_noise(original_img)
            noisy_tensor = preprocess(noisy_img, device)
            heatmap_noisy, _ = cam.forward(noisy_tensor, class_idx=pred_idx)
            sim_noise = compute_similarity(heatmap_base, heatmap_noisy)
            results["noise"].append(sim_noise)
            
            # 2. Rotation Test
            # Note: rotating the image means the heatmap SHOULD rotate too.
            # To compare, we rotate the *base heatmap* to match the rotated image's frame.
            rotated_img = rotate_image(original_img, angle=10)
            rotated_tensor = preprocess(rotated_img, device)
            heatmap_rotated, _ = cam.forward(rotated_tensor, class_idx=pred_idx)
            
            # Rotate base heatmap to align for comparison
            heatmap_base_aligned = rotate_image((heatmap_base * 255).astype(np.uint8), angle=10)
            heatmap_base_aligned = heatmap_base_aligned.astype(np.float32) / 255.0
            
            # Resize if needed (sometimes rotation changes dims slightly or artifacts appear)
            if heatmap_rotated.shape != heatmap_base_aligned.shape:
                 heatmap_base_aligned = cv2.resize(heatmap_base_aligned, (224, 224))

            sim_rot = compute_similarity(heatmap_base_aligned, heatmap_rotated)
            results["rotation"].append(sim_rot)

    avg_noise = np.mean(results["noise"])
    avg_rot = np.mean(results["rotation"])
    
    print(f"\nRobustness Results (Cosine Similarity, 1.0 = Perfect Stability):")
    print(f"  Gaussian Noise Stability: {avg_noise:.4f}")
    print(f"  Rotation Stability:      {avg_rot:.4f}")
    
    if avg_noise > 0.8:
        print(">> PASS: Model explanations are highly robust to noise.")
    else:
        print(">> WARN: Explanations are sensitive to pixel noise.")
        
    if avg_rot > 0.7: # Rotation is harder due to interpolation artifacts
        print(">> PASS: Model tracks anatomy consistently across rotation.")
    else:
        print(">> WARN: Rotation causes explanation drift.")

if __name__ == "__main__":
    main()
