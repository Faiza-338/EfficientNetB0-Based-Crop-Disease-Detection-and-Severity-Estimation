# 🌿 EfficientNetB0-Based Crop Disease Detection and Severity Estimation

### 🚀 Overview
This project focuses on **detecting plant diseases** and **estimating their severity** using a deep learning pipeline built on the **EfficientNetB0** architecture.  
A **two-stage training approach** was employed:
1. **Stage 1 – PlantVillage Dataset:** General plant-disease classification.  
2. **Stage 2 – Cassava Dataset:** Fine-tuning for Cassava leaf-disease severity estimation.

---

## 🎯 Focus of the Project
The main goal of this project is to build a robust deep learning model that can:
- Identify plant diseases across multiple crops.  
- Estimate **disease-severity levels** using domain-specific datasets.  
- Leverage **transfer learning** via EfficientNetB0 and **multi-stage fine-tuning**.

---

## 🌾 Dataset Description

### 🧩 Datasets Used
| Dataset | Source | Images | Classes | Description |
|----------|---------|---------|----------|--------------|
| **PlantVillage** | `emmarex/plantdisease` | **17 405** | **12** | Images of various crops and their diseases. Used for Stage 1 training. |
| **Cassava Leaf Disease Classification** | `nirmalsankalana/cassava-leaf-disease-classification` | **16 822** | **3 (severity levels)** | Cassava-leaf images labeled by disease severity. Used for Stage 2 fine-tuning. |

### 🔧 Preprocessing
- **Image Size:** 224 × 224 px  
- **Normalization:** Standard ImageNet mean & std  
- **Augmentation:**  
  `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`, `RandomBrightnessContrast`,  
  `HueSaturationValue`, `MotionBlur`, `RandomShadow`, `RandomFog`

---

## ⚙️ Model Architecture and Features

### 🧠 Base Model
- **Architecture:** EfficientNetB0 (pretrained on ImageNet)  
- **Framework:** `timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)`  
- **Custom Head:** Final classification layer adapted for each stage (PlantVillage classes / Cassava severity levels)  

### 🧩 Training Configuration
| Component | Description |
|------------|-------------|
| **Optimizer** | AdamW |
| **Loss Function** | CrossEntropyLoss |
| **Scheduler** | ReduceLROnPlateau (reduces LR on plateauing validation loss) |
| **Early Stopping** | Stops if validation loss doesn’t improve for N epochs |
| **Metrics** | Accuracy · F1-score (macro) · Precision · Recall |
| **Stages** | 2 (PlantVillage pretraining → Cassava fine-tuning with new model) |

### 🌟 Special Features
- **Independent Two-Stage Transfer Learning:** Separate EfficientNetB0 instances for each stage.  
- **Severity Mapping:** Cassava labels interpreted as severity levels.  
- **Visualization Tools:** Confusion matrices, ROC curves, and loss/accuracy plots.

---

## 📊 Results and Evaluations

### 🌱 Stage 1 – PlantVillage Dataset

**Early Stopping:** Epoch 14 (no val-loss improvement for 4 epochs)  
**Best Validation Loss:** 0.0179 (at Epoch 10)

| Metric | Train | Validation |
|--------|--------|------------|
| **Loss** | 0.0367 | 0.0214 |
| **Accuracy** | 0.987 | **0.994** |
| **F1-Score** | 0.987 | **0.994** |
| **Precision** | 0.987 | **0.996** |
| **Recall** | 0.987 | **0.994** |

**Visuals:**  
*(Insert images below once generated)*  
1️⃣ **Loss & Accuracy Curves**  
![PlantVillage Loss Accuracy](images/stage1_loss_accuracy.png)  

2️⃣ **Confusion Matrix**  
![PlantVillage Confusion Matrix](images/stage1_confusion_matrix.png)  

3️⃣ **ROC Curves**  
![PlantVillage ROC](images/stage1_roc_curve.png)

---

### 🍃 Stage 2 – Cassava Dataset (Severity Estimation)

> 🔁 Initialized a **new EfficientNetB0 model** (pretrained on ImageNet) for this stage, not continued from Stage 1.

**Early Stopping:** Epoch 6 (no val-loss improvement for 3 epochs)  
**Best Validation Loss:** 0.2401 (at Epoch 3)

| Metric | Train | Validation |
|--------|--------|------------|
| **Loss** | 0.2950 | 0.3019 |
| **Accuracy** | 0.747 | **0.784** |
| **F1-Score** | 0.761 | **0.781** |
| **Precision** | 0.778 | **0.785** |
| **Recall** | 0.747 | **0.784** |

**Severity Estimation:**  
The Cassava model directly classifies into severity levels (e.g., *Low*, *Medium*, *High*).  
Metrics reflect multi-class severity estimation accuracy.

**Visuals:**  
*(Insert images below once generated)*  
1️⃣ **Loss & Accuracy Curves**  
![Cassava Loss Accuracy](https://github.com/Faiza-338/Crop-Disease-Detection-Severity-Estimation-using-EfficientNetB0/blob/main/images/Cassava%20Evaluation/Confusiuon%20Matrix%20(Cassava).png)  

2️⃣ **Confusion Matrix**  
![Cassava Confusion Matrix](images/stage2_confusion_matrix.png)  

3️⃣ **ROC Curves**  
![Cassava ROC](images/stage2_roc_curve.png)

---

## 📈 Summary

| Stage | Dataset | Accuracy | F1-Score | Best Epoch |
|--------|----------|-----------|-----------|-------------|
| **Stage 1** | PlantVillage | **0.994** | **0.994** | 10 |
| **Stage 2** | Cassava | **0.784** | **0.781** | 3 |

---

## 🔬 Key Takeaways
- **EfficientNetB0** achieved near-perfect accuracy on PlantVillage and strong performance on Cassava.  
- **Independent two-stage training** ensured dataset-specific specialization.  
- **Augmentation + early stopping** enhanced generalization and reduced overfitting.  

---

## 📚 Future Work
- Extend to other regional crop datasets for broader generalization.  
- Incorporate **explainability** (Grad-CAM / SHAP) for interpretability.  
- Deploy via **FastAPI** or **Streamlit** for interactive severity prediction.  

---

## 🧾 References
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- [Cassava Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification)  
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
