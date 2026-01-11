# Literature Review & Competitive Analysis
**Project**: Explainable Brain MRI Classification  
**Date**: January 2026

## 1. Executive Summary
This document analyzes three seminal papers from 2025 that define the current State-of-the-Art (SOTA) in Brain Tumor Classification with Explainable AI (XAI). We benchmark our current `Explainable-Brain-MRI` project against these works to identify our competitive standing, technical gaps, and opportunities for future development.

**Conclusion**: Our project (ResNet18 + Grad-CAM) aligns well with the industry standard for *baseline* transparency but must evolve towards **quantitative evaluation** (as seen in Iftikhar et al.) and **advanced architectures** (like DenseNet in Islam et al.) to be considered truly cutting-edge.

---

## 2. Paper Summaries

### 2.1 Iftikhar et al. (2025): "Explainable CNN for Brain Tumor Detection"
*   **Journal**: Brain Informatics (April 2025)
*   **Core Contribution**: Simplification of CNN architectures while maintaining high accuracy.
*   **Methodology**:
    *   Custom, lightweight CNN architecture to reduce computational complexity.
    *   **XAI Suite**: Integrated **Grad-CAM**, **SHAP** (Shapley Additive Explanations), and **LIME** (Local Interpretable Model-agnostic Explanations).
*   **Key Results**:
    *   **99% Accuracy** on unseen data.
    *   Demonstrated that lighter models can outperform deeper ones if features are XAI-validated.
*   **Gap for Us**: We currently lack LIME and SHAP integration. Our model (ResNet18) is deeper/heavier than their custom solution.

### 2.2 Gharaibeh et al. (2025): "Leveraging Grad-CAM and SHAP"
*   **Journal**: Applied Computer Science (Oct 2025)
*   **Core Contribution**: Focused on the **Xception** architecture and the complementary nature of global (SHAP) vs. local (Grad-CAM) explanations.
*   **Methodology**:
    *   **Backbone**: Xception (Extreme Inception) CNN.
    *   **Dual-XAI**: Used Grad-CAM for localization (where is the tumor?) and SHAP for feature attribution (which textures matter?).
*   **Key Results**: 98.78% Test Accuracy on the 4-class problem (Glioma, Meningioma, Pituitary, No Tumor).
*   **Gap for Us**: Their use of **Xception** suggests that "Inception-style" modules might capture multi-scale features better than our ResNet's fixed kernels.

### 2.3 Islam et al. (2025): "Revolutionizing Brain Tumor Detection"
*   **Journal**: NMR in Biomedicine (Feb 2025)
*   **Core Contribution**: Handling **small metastatic lesions** and low-grade gliomas using **DenseNet121** and **Grad-CAM++**.
*   **Methodology**:
    *   **Backbone**: DenseNet121 (Feature reuse leads to better small-object detection).
    *   **XAI**: **Grad-CAM++**, which offers better object localization than standard Grad-CAM.
*   **Key Results**: 99.3% Accuracy.

### 2.4 Anonymous et al. (Oct 2024): "XAI-enhanced EfficientNetB0 Framework"
*   **Journal**: NIH/PubMed (PMC1148123)
*   **Core Contribution**: Precision brain tumor detection using EfficientNet-B0 as a lightweight backbone.
*   **Key Results**: 98.72% Accuracy. Validated that Grad-CAM identifies ROI even in noisy MRI sequences.
*   **Alignment**: This paper validates our transition from ResNet18 to EfficientNet-B0 as a high-performance, parameter-efficient choice.

### 2.5 Future Focus (June 2025): "Efficient Data-based Classification"
*   **Source**: ResearchGate / Image Transformers Study
*   **Core Contribution**: Direct comparison of EfficientNet-B0 vs Image Transformers (ViT) with Grad-CAM.
*   **Key Finding**: EfficientNet-B0 often matches ViT in diagnostic performance while requiring 1/10th of the computational power.

---

## 3. Comparative Gap Analysis

| Feature | **Our Project** | **Iftikhar et al.** | **Gharaibeh et al.** | **Islam et al.** |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone** | ResNet18 | Custom CNN | Xception | DenseNet121 |
| **Accuracy** | ~85-90% (Est) | 99% | 98.7% | 99.3% |
| **Classes** | 4-Class | 4-Class | 4-Class | 4-Class |
| **XAI Method** | Grad-CAM | Grad-CAM + SHAP + LIME | Grad-CAM + SHAP | Grad-CAM++ |
| **Faithfulness Check** | **Occlusion Test** | Qualitative | Qualitative | Qualitative |

### Our Competitive Advantage
While our accuracy is lower (expected for a prototype vs. published research), our **Quantitative Faithfulness Verification (Occlusion Test)** is a standout feature. Most papers (as noted in survey sections of 2.1 and 2.3) rely on "visual inspection" by doctors. By calculating the **0.48 Faithfulness Score**, we provide a *metric* for trust, which is highly relevant for industrial applications (FDA/EU AI Act).

---

## 4. Recommendations for Next Steps

To upgrade this project from "Demo-Ready" to "Publication-Ready", we should:

1.  **Upgrade XAI**: Implement **Grad-CAM++**. It is a drop-in mathematical replacement for Grad-CAM that handles multi-instance objects better.
2.  **Add SHAP**: SHAP provides global interpretability (overall dataset trends) which complements our local (per-image) heatmaps.
3.  **Model Swap**: Fine-tune a **DenseNet121** to see if we can close the accuracy gap with Islam et al.
4.  **Clinical User Study**: We have the "App", but we need to record *how* a doctor interacts with it to claim "Clinical Validation" truly.

---

**Prepared by**: Shek Lun Leung & Sai Oop Mong, MD (Independent Researchers)
