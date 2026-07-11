# 🐟 FishDisease-XAI: Attention-Guided Deep Learning for Aquaculture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

An interpretable deep learning framework for classifying freshwater fish diseases in South Asian aquaculture. This project introduces **CBAM-EfficientNetV2**, a highly robust model that combines an EfficientNetV2-S backbone with a Convolutional Block Attention Module (CBAM) and Generalized Mean (GeM) pooling to achieve state-of-the-art disease recognition.

To ensure transparency, the framework integrates Explainable AI (XAI) techniques (Grad-CAM++, Eigen-CAM, Score-CAM) to visually highlight the diseased regions the model focuses on.

---

## ✨ Key Features
* **Proposed Architecture**: `CBAM-EfficientNetV2` (Attention-guided CNN).
* **Robust Baselines**: Benchmarked against 7 popular vision models (EfficientNet-B0, MobileNetV3, CoAtNet, AlexNet, ConvNeXtV2, Swin-Base, ViT-Base).
* **Comprehensive XAI**: Visualizes model attention across 7 disease classes in a single combined heatmap output using 3 distinct CAM methods.
* **Precision Metrics**: Automatically calculates accuracy, macro/micro AUC, calibration curves (ECE), and one-vs-rest ROC.
* **Kaggle-Ready**: Built as a seamless, one-click Kaggle Pipeline.

## 📊 Dataset
The dataset includes 7 classes of freshwater fish conditions commonly found in South Asia:
1. Aeromoniasis
2. Bacterial Gill Disease
3. Bacterial Red Disease
4. Saprolegniasis
5. Healthy Fish
6. Parasitic Disease
7. White Tail Disease

*(Dataset Source: `subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia`)*

---

## 🚀 Model Architecture (CBAM-EfficientNetV2)

The proposed model relies on three core components:
1. **EfficientNetV2-S Backbone**: Highly efficient feature extractor.
2. **CBAM (Convolutional Block Attention Module)**: Sequentially infers attention maps along both channel and spatial dimensions, multiplying them by the input feature map for adaptive feature refinement.
3. **GeM Pooling**: Generalized Mean Pooling allows the network to focus on highly localized lesion patterns better than standard Global Average Pooling.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have a GPU environment (like Kaggle T4x2).
```bash
pip install -U torch torchvision timm grad-cam scikit-learn pandas matplotlib
```

### 2. Running the Pipeline
Simply execute the main script. The script handles everything from dataset splitting (85/15 stratified train/val) to training all 8 models and saving the metric plots.
```bash
python fish.py
```

### 3. Outputs
All generated assets are saved completely flat in the `outputs/` directory. For each model, you will get:
* `{model}_classification_report.txt`
* `{model}_metrics.json`
* `{model}_confusion_counts.png` / `{model}_confusion_norm.png`
* `{model}_roc_auc.png`
* `{model}_calibration.png`
* `{model}_training_curve.png`

For the proposed model, it also generates:
* `proposed_CBAM_EffNetV2_xai_all.png` (The combined XAI heatmap grid)

---

## 👁️ Explainable AI (XAI)
Deep learning models shouldn't be black boxes, especially in agriculture/aquaculture. The pipeline applies three distinct Class Activation Mapping methods:
* **Grad-CAM++**: Better localization for multiple occurrences of the same disease class.
* **Eigen-CAM**: Computes the first principle component of 2D activations (robust against adversarial noise).
* **Score-CAM**: A gradient-free visual explanation approach.

These heatmaps allow veterinarians and farmers to trust *why* the model made a specific diagnosis.

---

## 📜 License
This project is licensed under the MIT License.
