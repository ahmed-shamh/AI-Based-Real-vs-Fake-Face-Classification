# 🧠 AI-Based Real vs Fake Face Classification
DeepFake Detection using Xception & Swin Transformer v2

---

## 📌 Overview
This project focuses on detecting DeepFake images using two modern deep learning architectures:

- Xception  
- Swin Transformer v2  

The models were trained on the **140k Real and Fake Faces** dataset from Kaggle.  
Our goal is to build a highly accurate classifier that distinguishes real human faces from AI-generated fake faces.

The project includes:
- Full training pipeline  
- Checkpoint & resume system  
- Evaluation & visualization tools  
- GUI for inference  
- Two separate model implementations  
- Clear team collaboration structure  

---

## 👤 Team Members & Responsibilities

### Ahmed
- Presentation & Integration  
- Xception Model Engineering  
- Training Pipeline & Checkpoints  
- Evaluation & Testing  
- GitHub Management  

### Menna
- Dataset & Preprocessing  
- Augmentation & Regularization  
- Swin Model Engineering  
- Training Pipeline & Checkpoints  

### Malak
- Xception Model Engineering  
- Training Pipeline & Checkpoints  
- Evaluation & Testing  

### Elfeky
- Xception Model Engineering  
- Training Pipeline & Checkpoints  
- Evaluation & Testing  
- Swin Model Engineering  

### Noor
- Evaluation & Testing  
- Swin Model Engineering  
- Training Pipeline & Checkpoints  

### Elhdad
- GUI Development  
- Swin Model Engineering  
- Training Pipeline & Checkpoints
- <img width="1866" height="853" alt="image" src="https://github.com/user-attachments/assets/32046783-0d4d-45b8-a3ea-c15c8f20fb45" />
<img width="1851" height="857" alt="image" src="https://github.com/user-attachments/assets/98214222-88cd-4ab0-ae02-8654574f269a" />



---

## 📂 Project Structure

The repository contains two separate Jupyter Notebooks, one for each model:

│── xception_notebook.ipynb
│     - Dataset loading
│     - Preprocessing & augmentation (299x299)
│     - Xception model implementation
│     - Training loop with checkpoints
│     - Evaluation & test metrics
│     - Single-image prediction

│── swin_transformer_notebook.ipynb
│     - Dataset loading
│     - Preprocessing & augmentation (256×256)
│     - Swin Transformer v2 implementation
│     - Training loop with checkpoints
│     - Evaluation & metrics
│     - Single-image prediction

Additional files:

│── README.md
│── requirements.txt   (optional but recommended)


---

## 🗂 Dataset: 140k Real vs Fake Faces
Source: Kaggle — xhlulu/140k-real-and-fake-faces

### Dataset contains:
- 70k real images  
- 70k fake images  

### After splitting:
deepfake_dataset/
│── train/ (80%)
│── val/ (10%)
│── test/ (10%)



### Preprocessing includes:
- Removing corrupted files  
- Ensuring RGB format  
- Uniform transformation pipeline  

---

## 🧪 Data Augmentation (Training Only)
- Resize → RandomResizedCrop  
- Random Horizontal Flip  
- Color Jitter  
- Gaussian Blur  
- Posterize  
- Random Grayscale  
- Normalize `[-1, 1]`

Validation/Test only use:
- Resize  
- ToTensor  
- Normalize  

---

## 🧠 Model Architectures

### 1. Xception
- Depthwise separable convolutions  
- Strong performance on facial texture analysis  
- Custom head:
Dropout(0.3)
Linear(2048 → 1)


### 2. Swin Transformer v2
- Hierarchical Vision Transformer  
- Window-based self-attention  
- Custom head:
Dropout(0.3)
Linear(in_features → 1)


---

## ⚙️ Training Configuration

| Component | Value |
|----------|--------|
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Two LRs | Backbone: 1e-5, Head: 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 3 |
| Checkpoints | Best & per-epoch |
| AMP | Optional |

---

## 🧬 Training Pipeline
1. Load dataset  
2. Create dataloaders  
3. Initialize model (Xception or Swin v2)  
4. Train over epochs: forward → loss → backward → optimize  
5. Save checkpoints  
6. Reduce LR when accuracy plateaus  
7. Early stopping  
8. Save best model  

---

## 🧪 Evaluation Metrics
- Accuracy  
- Precision, Recall, F1  
- Confusion Matrix  
- ROC-AUC  
- Example predictions  

### Xception Example Results:
- Train Acc: ~99.07%  
- Val Acc: ~99.81%  
- Test Acc: ~99.7%  
- ROC-AUC: 0.999  

---

## 🖥 GUI Application
Created by: **Elhdad**

Features:
- Load an image  
- Run inference  
- Display Real/Fake probability  
- Simple interface for demonstrations  

---

## 🔁 Checkpoint System
Each checkpoint stores:
- Epoch  
- Model state  
- Optimizer state  
- Scheduler state  
- AMP scaler state  
- Best validation accuracy  

Allows:
- Resume training  
- Recover best model  
- Track progress  

---

## 📊 Model Comparison
| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| Xception | ~99.81% | Best performance |
| Swin Transformer v2 | High compute cost | Good generalization |

---

## 🚀 How to Run

### Install dependencies:
pip install torch torchvision timm pillow scikit-learn


### Prepare dataset:
Place dataset under:
A:\PyTorch\Project\140k_Real_Fake_Faces\real_vs_fake\real-vs-fake


### Train models:
python ../Xception15.py
python ../Swin.py


### Run GUI:
python ../app.py


---

## 🔮 Future Work
- DeepFake **video** detection  
- Add EfficientNet / ConvNeXt models  
- Real-time webcam detection  
- ONNX/TensorRT optimization  
- Deployment via FastAPI  
- Face alignment preprocessing  

---

## ⚠️ Limitations
- Works only on images, not videos  
- Limited generalization on unseen DeepFake generators  
- Dataset contains clean face crops (no occlusions)  

---

## 📚 Citation
@misc{140k_real_fake_faces,
author = {xhlulu},
title = {140k Real and Fake Faces},
year = 2020,
url = {https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces}
}

---

## 🙌 Acknowledgements
- Kaggle community  
- PyTorch  
- TIMM library  
- Project team members  

---

# 🎉 End of README
