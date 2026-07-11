<div align="center">

# 🐟 FishDisease-XAI
### Attention-Guided Deep Learning for Aquaculture

**Interpretable disease classification for freshwater fish using explainable AI**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20beff?style=flat-square&logo=kaggle)](https://www.kaggle.com/)

</div>

---

## 🎯 Overview

**FishDisease-XAI** is a state-of-the-art, interpretable deep learning pipeline designed to classify freshwater fish diseases in South Asian aquaculture. The system combines cutting-edge attention mechanisms with explainable AI techniques to provide veterinarians and aquaculture practitioners with transparent, trustworthy disease diagnostics.

### Why This Matters
- **99.43% accuracy** on 700 held-out test images across 7 disease classes
- **Explainable predictions** using Grad-CAM++, Eigen-CAM, and Score-CAM
- **Production-ready** with robust training protocols and comprehensive evaluation
- **Accessible** — runs natively on Kaggle GPU notebooks

---

## ⚡ Key Highlights

### 🏗️ Novel Architecture
```
EfficientNetV2-S Backbone
    ↓
CBAM (Channel + Spatial Attention)
    ↓
GeM Pooling (Generalized Mean)
    ↓
Regularized MLP Head
```

| Feature | Details |
|---------|---------|
| **Proposed Model** | CBAM-EfficientNetV2 with attention-refined features |
| **Baselines** | 7 models (EfficientNet-B0, MobileNetV3, CoAtNet, AlexNet, ConvNeXtV2, Swin, ViT) |
| **Training** | Mixed-precision AMP, class-weighted loss, cosine-annealing schedule, macro-F1 early stopping |
| **Explainability** | Grad-CAM++, Eigen-CAM, Score-CAM overlays for every class |

### 📊 Impressive Results

<table align="center">
<tr>
<td align="center">

**Overall Performance**

| Metric | Score |
|--------|-------|
| Accuracy | **99.43%** |
| Macro AUC | **~99.99%** |
| Micro AUC | **99.99%** |
| Errors | 4/700 |

</td>
<td align="center">

**Best-Performing Classes**

| Class | F1-Score |
|-------|----------|
| Healthy Fish | 1.0000 |
| Aeromoniasis | 0.9950 |
| Bacterial Red | 0.9950 |
| Bacterial Gill | 0.9950 |

</td>
</tr>
</table>

---

## 📁 Dataset

**Source:** [Kaggle — Freshwater Fish Disease (South Asia)](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia)

### 7 Disease Classes
| # | Disease | Type | Severity |
|---|---------|------|----------|
| 1 | Aeromoniasis | 🦠 Bacterial | Moderate |
| 2 | Bacterial Gill Disease | 🦠 Bacterial | High |
| 3 | Bacterial Red Disease | 🦠 Bacterial | High |
| 4 | Saprolegniasis | 🍄 Fungal | Moderate |
| 5 | Healthy Fish | ✅ N/A | — |
| 6 | Parasitic Disease | 🪱 Parasitic | Moderate |
| 7 | White Tail Disease | 🦠 Viral | Moderate |

**Dataset Split:**
- 1,400 training images (200/class) → 15% stratified validation split
- 700 test images (100/class, held-out)

---

## 🔍 Explainable AI (XAI)

The model includes three complementary CAM-based explanation techniques:

<div align="center">

| Method | Strength | Use Case |
|--------|----------|----------|
| **Grad-CAM++** | Improved multi-region localization | Complex lesions |
| **Eigen-CAM** | Principal component analysis; gradient-free | Robust to noise |
| **Score-CAM** | Confidence-driven explanations | Model confidence analysis |

</div>

**Key Benefit:** All three methods consistently highlight the diseased/anatomical region relevant to each diagnosis, providing veterinarians visual evidence for every prediction.

<img src="https://github.com/user-attachments/assets/7318c293-f88f-4325-b6c2-09bfc9e3a33a" alt="XAI Explanations" width="100%" />

---

## 📈 Performance Dashboard

### Training Dynamics
<img src="https://github.com/user-attachments/assets/522d1e59-f48e-44fe-a547-fee3d833110f" alt="Training Curves" width="100%" />

### Confusion Matrix
<img src="https://github.com/user-attachments/assets/7d9b8fe8-4910-4482-b5c5-56ecd87aafb7" alt="Confusion Matrix" width="48%" />

### ROC-AUC Curves
<img src="https://github.com/user-attachments/assets/db908f27-e029-4d49-9eeb-d277377e1fc8" alt="ROC-AUC" width="48%" />

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Kaggle account with GPU notebook access

### Installation

**Kaggle Notebook (Recommended):**
```bash
pip install -q -U timm grad-cam
```

**Local Environment:**
```bash
pip install torch torchvision timm grad-cam scikit-learn pandas matplotlib
```

### Running the Pipeline

1. **Open** the notebook (`fish-disease-cbam-xai.ipynb`) in Kaggle
2. **Ensure** dataset is mounted at `/kaggle/input/`
3. **Execute** cells sequentially:
   - Build stratified validation split
   - Train 7 baseline models
   - Train proposed CBAM-EfficientNetV2
   - Generate performance comparison
   - Generate XAI explanations
4. **Outputs** are saved to `/kaggle/working/outputs/`

### Expected Outputs

**Per-Model Artifacts:**
- Classification report (`.txt`)
- Metrics summary (`.json`)
- Confusion matrices — raw & normalized (`.png`)
- ROC-AUC curves with micro-average (`.png`)
- Calibration curves with ECE (`.png`)
- Training dynamics (`.png`)

**Proposed Model Only:**
- Per-class XAI overlays (`.png`)
- Combined XAI grid (`.png`)

---

## 📚 Architecture Details

### CBAM-EfficientNetV2 Components

**1. EfficientNetV2-S Backbone**
- ImageNet pretrained weights via `timm`
- Fully convolutional feature extraction
- Automatic fallback to compatible variants if needed

**2. CBAM (Convolutional Block Attention Module)**
- **Channel Attention:** Shared MLP over average & max-pooled features
- **Spatial Attention:** Convolution-based refinement over channel-pooled activations
- Sequential application for progressive feature refinement

**3. GeM (Generalized Mean) Pooling**
- Learnable parameter-based pooling
- Superior localization of lesion patterns vs. global average pooling
- Exposed as identity layer for clean CAM integration

**4. Regularized MLP Head**
- Label smoothing (ε = 0.1)
- Class-weighted cross-entropy loss
- Dropout for robustness

---

## 🎓 Training Recipe

| Hyperparameter | Value |
|---|---|
| **Optimizer** | AdamW (lr=1e-3) |
| **Scheduler** | Cosine Annealing + Warmup |
| **Loss** | Class-weighted CE + label smoothing |
| **Batch Size** | 32 |
| **Precision** | Mixed (AMP) |
| **Early Stopping** | Patience=7 on validation macro-F1 |
| **Epochs** | 14 (stopped early) |

---

## 📊 Class-Level Analysis

### Performance by Disease

```
✅ Healthy Fish:         F1 = 1.0000 (Perfect)
✅ Aeromoniasis:         F1 = 0.9950
✅ Bacterial Red:        F1 = 0.9950
✅ Bacterial Gill:       F1 = 0.9950
✅ Parasitic:            F1 = 0.9950
✅ White Tail:           F1 = 0.9950
⚠️  Saprolegniasis:      F1 = 0.9851 (Challenging)
```

### Insights
- **Healthy Fish** classified perfectly (zero false positives/negatives)
- **Saprolegniasis** is most challenging (fungal characteristics more subtle)
- Only **4 misclassifications** across 700 test images
- All baseline models included for comprehensive comparison

---

## 📁 Repository Structure

```
FishDisease-XAI/
├── fish-disease-cbam-xai.ipynb    # Main notebook
├── README.md                        # This file
├── LICENSE                          # MIT License
└── [outputs/]                       # Generated artifacts
    ├── metrics/                     # JSON summaries
    ├── plots/                       # Performance visualizations
    └── xai/                         # Explainability overlays
```

---

## 🔧 Advanced Features

- **Mixed-Precision Training** — faster convergence, reduced memory
- **Stratified Validation Split** — preserves class distribution
- **Macro-F1 Based Early Stopping** — prevents overfitting on imbalanced sets
- **Expected Calibration Error (ECE)** — confidence reliability metrics
- **One-vs-Rest ROC/AUC** — per-class diagnostic curves
- **Gradient-Free CAM Variants** — robust to noisy gradients

---

## 💡 Use Cases

✔️ **Aquaculture Farms** — Daily disease screening for farmed fish  
✔️ **Veterinary Clinics** — Diagnostic support for fish health assessments  
✔️ **Research** — Benchmark for XAI in aquaculture  
✔️ **Education** — Teaching interpretable deep learning  

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@project{fishdisease_xai_2024,
  title={FishDisease-XAI: Attention-Guided Deep Learning for Aquaculture},
  author={Abu-Bakar Rakib},
  year={2024},
  url={https://github.com/Abu-Bakar-Rakib/FishDisease-XAI}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or suggest improvements
- Submit pull requests
- Share results or use cases

---

<div align="center">

**Made with ❤️ for aquaculture practitioners and AI researchers**

[⬆ Back to Top](#-fishdisease-xai)

</div>
