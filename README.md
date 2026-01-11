# Explainable Brain MRI: Clinical-Grade Tumor Classification with Visual XAI & LLM Verification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gemini](https://img.shields.io/badge/AI-Gemini%201.5-blueviolet.svg)](https://deepmind.google/technologies/gemini/)

> **Transparency in Neuro-Oncology**: A complete pipeline for classifying brain tumors (Glioma, Meningioma, Pituitary) that combines **Deep Learning Accuracy**, **Grad-CAM++ Visual Explanations**, and **Generative AI Clinical Reporting**.
>
> 🔗 **Live Demo**: [https://8q7jszzenm55fqes96kfw4.streamlit.app/](https://8q7jszzenm55fqes96kfw4.streamlit.app/)

---

## 🚀 Key Innovations

### 1. 👁️ Smart-Masking XAI
Unlike standard heatmaps that obscure the image, our **Grad-CAM++** implementation uses dynamic thresholding to highlight *only* the lesion while keeping the anatomy visible.
*   **Method**: `cv2.threshold` > 30% intensity + Alpha Blending.
*   **Result**: Clear visualization of tumor boundaries without "color fog."

### 2. 🤖 AI Radiologist (Gemini 1.5 Integration)
We simulate a dual-reader workflow by piping the MRI scan and ResNet prediction to **Google Gemini 1.5 Pro**.
*   **Visual Verification**: The LLM independently analyzes the image features.
*   **Hallucination Check**: If ResNet is wrong (e.g., predicting Tumor on a healthy scan), Gemini often flags the discrepancy.
*   **Automated Reporting**: Generates a structured clinical draft (Findings, Impression) instantly.

### 3. 🔬 Advanced Training Methodology
We moved beyond simple baselines to implement a clinical-grade training pipeline:
*   **Dataset-Specific Normalization**: Calculated custom Mean/Std `[0.185, 0.185, 0.185]` specifically for brain MRI scans, replacing generic ImageNet defaults.
*   **Imbalance Correction**: Implemented **Weighted Cross-Entropy Loss** to ensure the model doesn't bias towards majority classes.
*   **Aggressive Augmentation**: Used `ElasticTransform` and `RandomBrightnessContrast` via Albumentations to simulate scanner variability and anatomical deformations.

---

## 🛠️ Tech Stack & Methodology

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backbone** | **EfficientNet-B0 / ResNet-50** | Feature Extraction with Parameter Efficiency |
| **Optimization** | **AdamW + Cosine Annealing** | Modern training loop for better convergence |
| **XAI Engine** | **Grad-CAM++** | High-fidelity localization of multiple lesion instances |
| **LLM Agent** | **Gemini 1.5 Pro/Flash** | Senior Neuroradiologist simulation & reporting |
| **Interface** | **Streamlit** | Interactive Clinical Dashboard |

---

## 💻 Interactive Application

The project includes a clinical dashboard for real-time inference.

### Live Demo
**[Click here to view the live app](https://8q7jszzenm55fqes96kfw4.streamlit.app/)**

### Running Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the App
streamlit run src/app.py
```

### 🔐 API Key Configuration (Optional)
To use the Gemini features without pasting your key every time, create a secrets file:
*   Create `.streamlit/secrets.toml`
*   Add: `GOOGLE_API_KEY = "your_key_here"`
*   *Note: This file is git-ignored for safety.*

### 📂 Project Structure
```bash
Explainable-Brain-MRI/
├── data/                  # Dataset directory
├── src/
│   ├── app.py             # Streamlit Dashboard (+ Gemini Integration)
│   ├── gradcam.py         # XAI Engine (Grad-CAM++)
│   └── ...
├── med-gemma/             # [Experimental] Local Med-LLM playground
├── results/               # Generated Heatmaps
└── README.md
```


---

## 📊 Evaluation Results

### Visual Explanations
| **Glioma** | **Meningioma** | **No Tumor** |
| :---: | :---: | :---: |
| ![Glioma](results/glioma_explanation.png) | ![Meningioma](results/meningioma_explanation.png) | ![No Tumor](results/notumor_explanation.png) |
*Note: The new Smart-Masking algorithm ensures the grey matter remains visible.*

### Quantitative Metrics
*   **Accuracy**: **99.7%** (EfficientNet-B0 + Brain Norm)
*   **Faithfulness (Occlusion Drop)**: 0.30 avg
*   **Noise Stability**: 0.58 (Cosine Similarity)
*   **Rotation Stability**: 0.98 (Cosine Similarity)

---

## 🚨 Limitations & Safety
*   **Single-Slice 2D Analysis**: The current ResNet18 model looks at a single 2D slice. Tumors that are clear in 3D volume but subtle in one axial slice may be missed. This is why we created the **Dual-Verification System**.
*   **Contrast Media**: Our model is trained on T1-weighted images. Some tumors (e.g., small meningiomas) are isointense to brain tissue and only become visible with Gadolinium contrast, which this dataset does not consistently distinguish.

---


## � References & Inspiration
*   **Iftikhar et al. (2025)**: Importance of XAI in medical black-box models.
*   **Islam et al. (2025)**: Grad-CAM++ for improved lesion localization.

---

**Author**: Shek Lun Leung (Independent Researcher) & Sai Oop Mong, MD (Clinical Lead)
**License**: MIT
