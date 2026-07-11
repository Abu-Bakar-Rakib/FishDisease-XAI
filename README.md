# 🐟 FishDisease-XAI: Attention-Guided Deep Learning for Aquaculture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

An interpretable deep learning pipeline for classifying freshwater fish diseases in South Asian aquaculture. The project introduces **CBAM-EfficientNetV2**, an attention-guided classifier that augments an EfficientNetV2-S backbone with a Convolutional Block Attention Module (CBAM) and Generalized Mean (GeM) pooling, benchmarked against seven established vision backbones.

To keep predictions interpretable, the pipeline applies three Class Activation Mapping (CAM) methods — Grad-CAM++, Eigen-CAM, and Score-CAM — to visualize the image regions driving each diagnosis.

---

## ✨ Key Features

- **Proposed architecture** — `CBAM-EfficientNetV2`: EfficientNetV2-S backbone → CBAM (channel + spatial attention) → GeM pooling → regularized MLP head.
- **Seven baseline comparisons** — EfficientNet-B0, MobileNetV3-Large, CoAtNet-0, AlexNet, ConvNeXtV2-Tiny, Swin-Base, and ViT-Base, all trained with the same protocol for a fair comparison.
- **Full evaluation suite** — per-model classification report, confusion matrices (raw + normalized), one-vs-rest ROC/AUC (macro & micro), and calibration curves with Expected Calibration Error (ECE), all reported to 4 decimal places.
- **Explainable AI** — Grad-CAM++, Eigen-CAM, and Score-CAM overlays for the proposed model, saved both as per-class images and as a single combined grid.
- **Robust training loop** — mixed-precision (AMP) training, class-weighted cross-entropy with label smoothing, cosine-annealing LR schedule, and macro-F1-based early stopping.
- **Kaggle-native** — designed to run end-to-end inside a Kaggle notebook environment with GPU acceleration.

---

## 📊 Dataset

**Source:** [`subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia`](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia)

7 classes of freshwater fish conditions common in South Asian aquaculture:

1. Aeromoniasis (Bacterial)
2. Bacterial Gill Disease
3. Bacterial Red Disease
4. Saprolegniasis (Fungal)
5. Healthy Fish
6. Parasitic Disease
7. White Tail Disease (Viral)

The dataset ships with pre-defined `Train/` and `Test/` folders. The notebook carves a **stratified 15% validation split out of `Train/`** purely for early stopping and best-checkpoint selection — the `Test/` split is held out and used only once, for the final evaluation of each model, so there is no test leakage.

---

## 🚀 Model Architecture (CBAM-EfficientNetV2)

The proposed model combines three components:

1. **EfficientNetV2-S backbone** (via `timm`, ImageNet-pretrained) — a fully convolutional feature extractor, with automatic fallback to a compatible EfficientNet variant if the primary weights can't be loaded.
2. **CBAM (Convolutional Block Attention Module)** — sequentially applies channel attention (shared MLP over average- and max-pooled features) and spatial attention (convolution over channel-pooled maps) to refine the feature map.
3. **GeM (Generalized Mean) pooling** — a learnable pooling operator that emphasizes localized lesion patterns more effectively than standard global average pooling.

The attention-refined feature map is exposed as a dedicated (identity) layer, giving CAM-based methods a clean, well-defined target for generating explanations. The model is trained with label smoothing and a cosine-annealing learning rate schedule.

> This is framed as an *integration-style* contribution — a strong pretrained backbone augmented with attention and a tuned training recipe — rather than a novel architectural operator.

---

## 📈 Results — Proposed Model (CBAM-EfficientNetV2)

Trained for 14 epochs before early stopping (patience = 7 on validation macro-F1), best validation macro-F1 ≈ **0.980**.

**Test set performance (700 held-out images, 100/class):**

| Metric | Value |
|---|---|
| Accuracy | 0.9943 |
| Macro AUC (avg. one-vs-rest) | ~0.9999 |
| Micro AUC | 0.9999 |
| Errors | 4 / 700 |

**Confusion matrix highlights:** 4 total misclassifications out of 700 test images — 1 Aeromoniasis → Bacterial Red disease, 1 Bacterial Gill disease → Saprolegniasis, 1 Saprolegniasis → White Tail disease, 1 Parasitic disease → Saprolegniasis. All other classes were classified perfectly (100/100).

**ROC-AUC:** every class reaches an AUC of 1.0000 except Saprolegniasis (0.9996) and Parasitic disease (0.9999), reflecting the same few borderline cases seen in the confusion matrix.

**Training curves:** train/val loss converge smoothly with no significant overfitting gap; validation macro-F1 climbs sharply in the first 4 epochs (0.81 → 0.98) before plateauing, consistent with a well-regularized fine-tuning run on a fully pretrained backbone.

<img width="1650" height="600" alt="proposed_CBAM_EffNetV2_training_curve" src="https://github.com/user-attachments/assets/522d1e59-f48e-44fe-a547-fee3d833110f" />
<img width="1036" height="883" alt="proposed_CBAM_EffNetV2_confusion_counts" src="https://github.com/user-attachments/assets/7d9b8fe8-4910-4482-b5c5-56ecd87aafb7" />
<img width="1034" height="881" alt="proposed_CBAM_EffNetV2_roc_auc" src="https://github.com/user-attachments/assets/db908f27-e029-4d49-9eeb-d277377e1fc8" />

---

## 👁️ Explainable AI (XAI)

Three Class Activation Mapping methods are applied to the proposed model, one held-out test image per class:

- **Grad-CAM++** — improved localization when a class appears in multiple regions of an image.
- **Eigen-CAM** — takes the principal component of the activations; gradient-free and robust to noisy gradients.
- **Score-CAM** — gradient-free explanations derived from the model's own confidence scores rather than backpropagated gradients.

All three methods consistently localize attention on the diseased/anatomical region relevant to each class — the red lesion bands for Bacterial Red disease, the inflamed gill/mouth area for Bacterial Gill disease and Aeromoniasis, and the whole-body region for Healthy Fish, Parasitic disease, and White Tail disease.

Notably, the one Saprolegniasis sample shown was **misclassified as White Tail disease** — matching the single Saprolegniasis error seen in the confusion matrix above. All three CAM methods still focus on the fish's body rather than background, which is useful for debugging: the model is attending to the right *object*, but the visual cue it's keying on doesn't distinguish this fungal case correctly. This kind of failure case is exactly what the XAI overlays are meant to surface.

<img width="2315" height="4179" alt="proposed_CBAM_EffNetV2_xai_all" src="https://github.com/user-attachments/assets/7318c293-f88f-4325-b6c2-09bfc9e3a33a" />


These overlays are intended to give veterinarians and aquaculture practitioners visual insight into *why* the model reached a given diagnosis, rather than treating it as a black box.

---

## 🛠️ Installation & Usage

### 1. Environment

Built and tested for a **Kaggle GPU notebook** (Settings → Accelerator = GPU, Internet = On). Internet access is required inside the notebook to download pretrained `timm` weights and install `grad-cam`.

`torch`, `torchvision`, `scikit-learn`, `pandas`, and `matplotlib` are expected to already be available in the Kaggle Python environment. The notebook additionally installs, at the top of its first cell:

```bash
pip install -q -U timm grad-cam
```

To reproduce the environment elsewhere:

```bash
pip install torch torchvision timm grad-cam scikit-learn pandas matplotlib
```

### 2. Running the Pipeline

This project is distributed as a Jupyter notebook (`fish-disease-cbam-xai.ipynb`), not a standalone script. Open it in Kaggle (or Jupyter/Colab, with the dataset mounted under `/kaggle/input`) and run all cells in order. The notebook will:

1. Locate the `Train/` and `Test/` folders and build a stratified 15% validation split from `Train/`.
2. Train each of the 7 baseline models in turn, evaluating each on the untouched `Test/` set.
3. Train the proposed CBAM-EfficientNetV2 model.
4. Generate a comparison table/plot across all 8 models.
5. Generate Grad-CAM++, Eigen-CAM, and Score-CAM explanations for the proposed model.

All artifacts are written flat to `/kaggle/working/outputs/`.

### 3. Outputs

For each of the 8 models:

- `{model}_classification_report.txt`
- `{model}_metrics.json` — accuracy, macro precision/recall/F1, weighted F1, macro/micro AUC, ECE (all rounded to 4 decimals)
- `{model}_confusion_counts.png` / `{model}_confusion_norm.png`
- `{model}_roc_auc.png` — one-vs-rest ROC curves per class, plus micro-average
- `{model}_calibration.png` — reliability diagram with ECE
- `{model}_training_curve.png` — train/val loss and val macro-F1 per epoch


For the proposed model only:

- `proposed_CBAM_EffNetV2_xai_{class_name}.png` — one Grad-CAM++/Eigen-CAM/Score-CAM overlay row per class
- `proposed_CBAM_EffNetV2_xai_all.png` — all classes combined into a single grid

---

## 📜 License

This project is licensed under the MIT License.
