import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image
from model import get_model
from gradcam import GradCAM

# Page Config
st.set_page_config(
    page_title="Explainable Brain MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utils ---
@st.cache_resource
def load_trained_model(model_path, device):
    try:
        model = get_model(num_classes=4, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(pil_image, device):
    # Convert PIL to CV2 format (RGB)
    img = np.array(pil_image)
    
    # Check if gray and convert
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    orig_img = cv2.resize(img, (224, 224))
    
    # Normalize for Model
    img_tensor = orig_img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
    
    return orig_img, img_tensor

# --- Sidebar ---
st.sidebar.title("🧠 Clinical Dashboard")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project**: Explainable Brain MRI\n\n"
    "**Model**: ResNet18 + Grad-CAM\n\n"
    "**Classes**: Glioma, Meningioma, Pituitary, No Tumor"
)

visualization_mode = st.sidebar.radio("Visualization Mode", ["Side-by-Side", "Overlay Only"])
alpha = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)

# --- Main Page ---
st.title("Explainable Brain Tumor Classification")
st.markdown("""
This tool uses **Deep Learning** and **XAI** to assist radiologists. 
Upload a T1-weighted MRI scan to classify the tumor type and visualize the regions driving the diagnosis.
""")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_PATH = "best_brain_tumor_model.pth"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load Model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found! Please run training first.")
    st.stop()

model = load_trained_model(MODEL_PATH, device)

# Initialize GradCAM
# Target Layer: Last layer of layer4 for ResNet18
target_layer = model.layer4[-1]
cam = GradCAM(model, target_layer)

# File Upload
uploaded_file = st.file_uploader("Upload Patient MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Load and Preprocess
    image_pil = Image.open(uploaded_file)
    original_img_np, input_tensor = preprocess_image(image_pil, device)
    
    # 2. Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(output, dim=1).item()
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]

    # 3. Grad-CAM
    heatmap, _ = cam.forward(input_tensor, class_idx=pred_idx)
    overlay = cam.overlay_heatmap(heatmap, original_img_np, alpha=alpha)

    # 4. Display Results
    st.markdown("### 🧬 Diagnostic Analysis")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Scan")
        st.image(original_img_np, caption="Original Input", use_column_width=True)

    with col2:
        st.subheader("AI Explainability (Grad-CAM)")
        if visualization_mode == "Side-by-Side":
             st.image(overlay, caption=f"Lesion Highlight ({pred_class})", use_column_width=True)
        else:
             st.image(overlay, caption="Heatmap Overlay", use_column_width=True)

    st.markdown("---")
    
    # 5. Metrics
    st.markdown("### 📊 Classification Confidence")
    
    # Custom colored metrics
    metric_cols = st.columns(4)
    for i, cls_name in enumerate(CLASSES):
        prob = probs[i]
        delta_color = "normal"
        if cls_name == pred_class:
            delta_color = "off" if cls_name == "notumor" else "inverse" # Highlight the winner
            st.metric(label=cls_name.title(), value=f"{prob*100:.1f}%", delta="Primary Diagnosis" if prob > 0.5 else None)
        else:
             st.metric(label=cls_name.title(), value=f"{prob*100:.1f}%")
             
    # Disclaimer
    st.warning("⚠️ **Disclaimer**: This tool is a research prototype. Do not use for definitive clinical diagnosis.")

else:
    st.info("👆 Please upload an MRI image to begin analysis.")
