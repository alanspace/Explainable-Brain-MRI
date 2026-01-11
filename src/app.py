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
    page_title="LucidMed AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utils ---
@st.cache_resource
def load_trained_model(model_path, model_name, device):
    try:
        model = get_model(model_name=model_name, num_classes=4, pretrained=False)
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
    
    # Updated Mean/Std based on Brain Dataset
    BRAIN_MEAN = [0.1854, 0.1854, 0.1855]
    BRAIN_STD = [0.1855, 0.1855, 0.1855]
    
    img_tensor = orig_img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - BRAIN_MEAN) / BRAIN_STD
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
    
    return orig_img, img_tensor

# --- Sidebar ---
st.sidebar.title("🩺 LucidMed AI")
st.sidebar.caption("Intelligent Radiology Assistant")
st.sidebar.markdown("---")

# Roadmap / Multi-organ support (Placeholder)
st.sidebar.selectbox(
    "Organ System", 
    ["Brain MRI (v1.0)", "Chest X-Ray (Coming Soon)", "Retinal Fundus (Coming Soon)"],
    index=0,
    disabled=True,
    help="Multi-organ support is currently in development."
)

st.sidebar.info(
    "**Core Model**: EfficientNet-B0 + Grad-CAM++\n\n"
    "**Classes**: Glioma, Meningioma, Pituitary, No Tumor"
)

# Model selection in sidebar (optional, but good to have)
model_choice = st.sidebar.selectbox("Model Architecture", ["efficientnet_b0", "resnet50"], index=0)

visualization_mode = st.sidebar.radio("Visualization Mode", ["Side-by-Side", "Overlay Only"])
alpha = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)

# --- Main Page ---
# --- Main Page ---
st.title("LucidMed AI: Intelligent Radiology Assistant")
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

model = load_trained_model(MODEL_PATH, model_choice, device)

# Initialize GradCAM
# Target Layer selection based on model
if model_choice == 'efficientnet_b0':
    target_layer = model.features[-1]
elif model_choice == 'resnet50':
    target_layer = model.layer4[-1]
else:
    target_layer = model.layer4[-1] # default fallback

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
    st.markdown("### 🤖 Radiologist Assistant")
    st.markdown("We use **Gemini 2.0 Flash** by default for its speed and efficiency.")
    
    st.info("💡 **Suggestion**: If you need deeper clinical reasoning or have high-volume needs, you can provide your own API Key in the sidebar. This allows you to tap into higher rate limits or switch to models like **Gemini 1.5 Pro** if supported.")

    # Try to load from secrets first, else ask user
    # API Key Handling
    user_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="Enter your own key to override the default.", placeholder="Paste GOOGLE_API_KEY")
    
    if user_api_key:
        api_key = user_api_key
        st.sidebar.success("Using Custom API Key")
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        # st.sidebar.info("Using Default API Key") 
    else:
        api_key = None

    if api_key:
        import google.generativeai as genai
        
        if st.button("Generate Professional Report"):
            try:
                genai.configure(api_key=api_key)
                # Attempt to connect to the best available model
                available_models = ['gemini-2.0-flash', 'gemini-1.5-flash']
                model_gemini = None
                
                last_error = "Unknown error"
                for model_name in available_models:
                    try:
                        # Use a more direct instantiation
                        model_gemini = genai.GenerativeModel(model_name=model_name)
                        # We don't do a full generate_content ping here to avoid quota/limit issues
                        # We just test the instantiation
                        break
                    except Exception as e:
                        last_error = str(e)
                        model_gemini = None
                
                if model_gemini is None:
                    st.error(f"Could not connect to any Gemini models. Last error: {last_error}")
                    st.info("Ensure your API Key is correct and has 'Generative Language API' enabled in Google AI Studio.")
                    st.stop()

                
                # Prompt Engineering for Medical Context
                prompt = f"""
                You are **Dr. Lucid**, a Senior Consultant Neuroradiologist at LucidMed AI.
                You are collaborating with an AI Classification system to validate findings.
                
                **AI Findings**:
                - Predicted Class: {pred_class.upper()}
                - Confidence Score: {confidence*100:.1f}%
                
                **Your Mission**:
                1. **Clinical Validation**: Examine the provided T1-weighted MRI scan. Does the anatomical presentation align with a {pred_class}?
                2. **Explainability Review**: The AI has highlighted specific regions (Grad-CAM++). Assuming the AI is focusing on the most relevant features, evaluate if its "attention" is clinically sound or if it might be looking at artifacts. 
                3. **Diagnostic Report**: Draft a professional radiology report including:
                   - **Observations**: Lesion size, morphology, signal intensity, and location.
                   - **Differential Diagnosis**: If the AI confidence is low, what else could it be?
                   - **Impression**: Final summary and recommended next steps (e.g., Contrast-enhanced MRI, biopsy).
                
                Maintain a formal, objective, and analytical tone. Use precise neuroanatomical terms.
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
