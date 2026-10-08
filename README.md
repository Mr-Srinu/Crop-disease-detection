# 🌿 Crop Disease Detection

> **An end-to-end deep learning system for automated crop disease classification, disease-region detection, and explainable AI.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-111111)](https://docs.ultralytics.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

**Research:** CVR 2026 — *6th International Conference on Computer Vision and Robotics*, NIT Goa  
**Proceedings:** Springer — *Lecture Notes in Networks and Systems*

The paper **“An Efficient Deep Learning Framework for Automated Crop Disease Detection Using Convolutional Neural Networks”** is listed in the official CVR 2026 program. citeturn4search0turn0search5

---

## ✦ What this project does

```text
        LEAF IMAGE
             │
             ▼
      ┌──────────────┐
      │ Preprocessing│
      └──────┬───────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 CLASSIFICATION   DETECTION
      │             │
      │             ▼
      │        Disease Region
      │             │
      ▼             │
 Disease Class      │
      │             │
      └──────┬──────┘
             ▼
        GRAD-CAM / XAI
             │
             ▼
        FLASK WEB APP
             │
             ▼
     Visual Prediction
```

### Core capabilities

| Feature | What it provides |
|---|---|
| 🧠 **Disease Classification** | Identifies the predicted crop disease |
| 🎯 **Disease Detection** | Localizes affected regions using YOLO |
| 🔍 **Grad-CAM** | Visual explanation of classification decisions |
| 📊 **Model Evaluation** | Accuracy, precision, recall, F1, ROC and confusion matrix |
| ⚡ **Model Comparison** | Comparison across CNN and YOLO architectures |
| 🌐 **Web Interface** | Upload an image and view prediction results |

---

## 🏆 Key Results

| Task | Best Result |
|---|---:|
| CNN Classification | **99.07% accuracy — DenseNet121** |
| Object Detection | **78% mAP — YOLOv11** |

> Results reported from the experimental work associated with this project.

---

## 🛠 Tech Stack

**Language**
- Python

**Deep Learning**
- TensorFlow / Keras
- CNN architectures
- Transfer Learning

**Computer Vision**
- OpenCV
- YOLO
- Grad-CAM

**Data & Evaluation**
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

**Application**
- Flask
- HTML / CSS / JavaScript

---

# 🚀 Installation

Follow these steps in order.

### 1. Clone the repository

```bash
git clone https://github.com/Mr-Srinu/Crop-disease-detection.git
cd Crop-disease-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

You should now see `(venv)` in your terminal.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add the dataset

Download the project dataset from **[Kaggle](YOUR_KAGGLE_DATASET_URL)** and place it in the dataset directory expected by the project.

```text
Crop-disease-detection/
├── dataset/
│   └── ...
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

> If the repository already contains the required trained model files, you can run the application without retraining.

### 6. Start the application

```bash
python app.py
```

Open the local URL shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

# 🖥 Using the Application

```text
1. Open the web application
        ↓
2. Upload a crop / leaf image
        ↓
3. Run prediction
        ↓
4. View disease classification
        ↓
5. View detected disease regions
        ↓
6. Inspect Grad-CAM explanation
```

---

# 📦 Dataset

**Dataset:** [Kaggle Dataset](YOUR_KAGGLE_DATASET_URL)

**Size:** ~2.05 GB

The dataset is required for training/reproducing the experiments.  
For inference using existing trained models, the full dataset may not be required.

---

# 🔬 Reproducing the Research

To reproduce the experiments:

```text
Dataset
   ↓
Preprocessing
   ↓
Train CNN Models
   ↓
Evaluate Classification
   ↓
Train YOLO Models
   ↓
Evaluate Detection
   ↓
Generate Grad-CAM
   ↓
Compare Results
   ↓
Run Flask Application
```

Use the notebooks/scripts provided in the repository for the corresponding experiments.

---

# 📁 Project Structure

```text
Crop-disease-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── training & analysis
│
├── models/
│   └── trained model files
│
├── static/
│   └── web assets
│
├── templates/
│   └── Flask HTML pages
│
└── dataset/
    └── downloaded dataset
```

> Folder names may vary slightly depending on the current repository version.

---

# 📚 Research

**Title**

> *An Efficient Deep Learning Framework for Automated Crop Disease Detection Using Convolutional Neural Networks*

**Conference:** 6th International Conference on Computer Vision and Robotics (CVR 2026)  
**Venue:** National Institute of Technology Goa  
**Proceedings:** Springer, *Lecture Notes in Networks and Systems*

The paper is listed in the official CVR 2026 program under Computer Vision. citeturn4search0

**Official conference:**  
https://scrs.in/conference/cvr2026

---

# ⚠️ Usage & Rights

### No Open-Source License

This repository currently **does not grant an open-source license**.

Public availability on GitHub does **not** mean that the source code, research content, figures, trained models, or other project materials are free to copy, redistribute, modify, publish, or use commercially. GitHub notes that without a license, default copyright rules apply. citeturn0search3

**Please:**
- Do not present this work as your own.
- Do not republish the research or its results without appropriate permission.
- Do not commercially redistribute project materials without permission.
- Give proper academic citation when referring to this work.
- Check the separate licensing/terms of the dataset and third-party models or libraries.

For academic use, citation is expected.  
For reuse, redistribution, modification, or commercial use, obtain permission from the respective rights holder.

---

## 👤 Author

**B. Srinivasulu**

GitHub: **[@Mr-Srinu](https://github.com/Mr-Srinu)**

---

## ⭐ If this project helped you

Consider starring the repository and citing the research when using or discussing this work.
