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
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Robust Model Path finding
possible_paths = [
    "best_brain_tumor_model.pth", 
    "../best_brain_tumor_model.pth",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_brain_tumor_model.pth")
]

MODEL_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    st.error("Model file `best_brain_tumor_model.pth` not found in root or parent directory!")
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
    print(f"DEBUG: Image loaded from {uploaded_file.name}. Size: {image_pil.size}")
    
    original_img_np, input_tensor = preprocess_image(image_pil, device)
    
    # 2. Inference
    print(f"DEBUG: Running inference on device: {device}")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(output, dim=1).item()
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]
        
        print(f"DEBUG: Model Probabilities: {probs}")
        print(f"DEBUG: Predicted: {pred_class} ({confidence:.2f})")

    # 3. Grad-CAM
    heatmap, _ = cam.forward(input_tensor, class_idx=pred_idx)
    overlay = cam.overlay_heatmap(heatmap, original_img_np, alpha=alpha)

    # 4. Display Results
    st.markdown("### 🧬 Diagnostic Analysis")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Scan")
        st.image(original_img_np, caption="Original Input", use_container_width=True)

    with col2:
        st.subheader("AI Explainability (Grad-CAM)")
        if visualization_mode == "Side-by-Side":
             st.image(overlay, caption=f"Lesion Highlight ({pred_class})", use_container_width=True)
        else:
             st.image(overlay, caption="Heatmap Overlay", use_container_width=True)

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
             
    st.markdown("---")

    # 6. Gemini Integration
    st.markdown("### 🤖 Radiologist Assistant (Gemini 1.5 Pro)")
    st.info("Uses a Vision Language Model to verify the findings and draft a clinical report.")

    # Try to load from secrets first, else ask user
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from secrets.")
    else:
        api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Paste GOOGLE_API_KEY here")

    if api_key:
        import google.generativeai as genai
        
        if st.button("Generate Professional Report"):
            try:
                genai.configure(api_key=api_key)
                # Attempt to connect to the best available model
                available_models = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
                model_gemini = None
                
                print(f"DEBUG: API Key loaded: {api_key[:5]}...{api_key[-5:]}")
                
                for model_name in available_models:
                    try:
                        print(f"DEBUG: Trying model '{model_name}'...")
                        test_model = genai.GenerativeModel(model_name)
                        # Quick ping to verify access
                        test_model.generate_content("Ping")
                        model_gemini = test_model
                        print(f"DEBUG: Successfully connected to {model_name}")
                        break
                    except Exception as e:
                        print(f"DEBUG: Failed to connect to {model_name}: {e}")
                
                if model_gemini is None:
                    st.error("Could not connect to any Gemini models. Check your API Key permissions.")
                    st.stop()

                
                # Prompt Engineering for Medical Context
                prompt = f"""
                You are an expert Neuroradiologist assistant. 
                You are provided with a T1-weighted MRI brain scan of a patient.
                
                An automated Deep Learning classifier has analyzed this image and predicted:
                **Diagnosis**: {pred_class.upper()}
                **Confidence**: {confidence*100:.1f}%

                Please perform the following tasks professionally:
                1. **Visual Verification**: Analyze the image. Do you see features consistent with a {pred_class}? Describe the lesion location, shape, and intensity if present.
                2. **Critique**: Does the AI's confidence ({confidence*100:.1f}%) seem justified based on the visual evidence?
                3. **Clinical Report**: Draft a concise, formal radiology report section for this scan. Use standard medical terminology (e.g., "hyperintense", "mass effect", "midline shift").

                Note: If the image appears normal (No Tumor), describe the healthy anatomical structures.
                """
                with st.spinner("Consulting Gemini API..."):
                    # Send PIL image and prompt
                    print("DEBUG: Sending request to Gemini...")
                    st.toast("Sending request to Gemini...", icon="🚀")
                    response = model_gemini.generate_content([prompt, image_pil]) 
                    print(f"DEBUG: Response received. Type: {type(response)}")
                    if response.text:
                        print("DEBUG: Response text length:", len(response.text))
                        st.markdown(response.text)
                    else:
                        print("DEBUG: Empty response text!")
                        st.error("Gemini returned an empty response. Please check the logs.")
                    
                    
            except Exception as e:
                st.error(f"Gemini Error: {e}")
    else:
        st.warning("⚠️ Enter your Gemini API Key in the sidebar to unlock the AI Radiologist.")

    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: This tool is a research prototype. Deep Learning & LLM outputs can hallucinate. Always consult a human doctor.")

else:
    st.info("👆 Please upload an MRI image to begin analysis.")
