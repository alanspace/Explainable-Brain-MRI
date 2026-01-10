import torch
import cv2
import numpy as np
import os
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

def perturb_image(img, heatmap, threshold=0.5):
    """
    Masks the top % of the heatmap regions in the original image.
    Standard 'Faithfulness' check: If we remove the important region, confidence should drop.
    """
    # Resize heatmap to match image dimensions (224, 224)
    if heatmap.shape != (224, 224):
        heatmap = cv2.resize(heatmap, (224, 224))

    # Create mask where heatmap is strong
    mask = heatmap > threshold
    
    # Perturbed image: Black out the important regions
    # (In a real clinical setting, we might replace with mean noise, but black is standard for first-pass)
    perturbed = img.copy()
    perturbed[mask] = 0 # Set to black
    
    return perturbed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device for XAI Verification: {device}")
    
    # Load Model
    model = get_model(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load("best_brain_tumor_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Grad-CAM
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    test_dir = "data/Testing" # Or Training if Testing is empty
    if not os.path.exists(test_dir):
        test_dir = "data/Training"

    # Metrics
    total_drop = 0.0
    total_images = 0
    scores_by_class = {c: [] for c in classes}
    
    print("\n--- Starting Faithfulness Evaluation (Occlusion Test) ---")
    print("Hypothesis: If Grad-CAM highlights the true tumor, removing that region should drastically lower model confidence.\n")
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images = glob(os.path.join(cls_dir, "*"))
        
        # Test on random sample of 50 images per class for speed
        if len(images) > 50:
            images = images[:50]
            
        for img_path in tqdm(images, desc=f"Verifying {cls}"):
            original_img = load_image(img_path)
            input_tensor = preprocess(original_img, device)
            
            # 1. Get Original Prediction
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = torch.argmax(output, dim=1).item()
                orig_conf = probs[pred_idx]
            
            # Only evaluate if model was correct initially (otherwise explanation is moot)
            if classes[pred_idx] != cls:
                continue
                
            # 2. Generate Heatmap
            heatmap, _ = cam.forward(input_tensor, class_idx=pred_idx)
            
            # 3. Perturb Image (Remove highlighted region)
            # Use threshold of 0.5 (top 50% importance roughly)
            perturbed_img = perturb_image(original_img, heatmap, threshold=0.5)
            perturbed_tensor = preprocess(perturbed_img, device)
            
            # 4. Get New Prediction
            with torch.no_grad():
                output_new = model(perturbed_tensor)
                probs_new = torch.softmax(output_new, dim=1).cpu().numpy()[0]
                new_conf = probs_new[pred_idx]
                
            # 5. Calculate Drop
            drop = max(0, orig_conf - new_conf)
            scores_by_class[cls].append(drop)
            total_drop += drop
            total_images += 1

    print("\n--- Results: Average Confidence Drop when 'Explanation' is Removed ---")
    print("(Higher 'Drop' is better -> means the explanation was actually important)")
    
    avg_total_drop = total_drop / total_images if total_images > 0 else 0
    print(f"\nOverall Average Faithfulness Score: {avg_total_drop:.4f} (Avg Probability Drop)")
    
    for cls in classes:
        scores = scores_by_class[cls]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {cls.title()}: {avg:.4f}")
        else:
            print(f"  {cls.title()}: N/A (No correct predictions to test)")

if __name__ == "__main__":
    main()
