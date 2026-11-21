# 📦 WarehouseVision AI
### Autonomous Warehouse Perception & Logistics Analytics System

WarehouseVision AI is a computer vision-based warehouse monitoring project that uses deep learning to detect and analyze objects in warehouse environments.

The system uses a custom-trained YOLO11n object detection model to identify warehouse objects such as people, forklifts, boxes, pallets and other logistics-related equipment.

It provides an interactive Streamlit dashboard for image analysis, object counting and visual analytics.

---

## 🎯 Project Objectives

- Detect warehouse objects using deep learning.
- Identify people, forklifts and warehouse materials.
- Analyze detected objects through automated counting.
- Visualize object distribution using interactive charts.
- Explore object tracking for warehouse monitoring.
- Establish a foundation for future logistics optimization.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLO11n | Object detection |
| Ultralytics | Model training and inference |
| PyTorch | Deep learning framework |
| OpenCV | Computer vision |
| NumPy | Numerical operations |
| Pandas | Data analysis |
| Plotly | Interactive visualizations |
| Streamlit | Web dashboard |
| SQLite | Planned data storage |

## 🏗️ Project Architecture

```text
WarehouseVision AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── sample/
│   └── processed/
│       ├── warehouse_yolo/
│       └── warehouse_yolo_clean/
│
├── models/
│   └── warehouse_yolo11_final.pt
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_warehouse_analytics.ipynb
│
├── outputs/
├── runs/
│
├── src/
│   ├── analytics/
│   │   └── object_counter.py
│   ├── optimization/
│   ├── utils/
│   └── vision/
│       └── yolo_test.py
│
├── tests/
├── requirements.txt
└── README.md
```

## 📂 Dataset Information

Dataset: [Warehouse Objects Dataset – Roboflow Universe](https://universe.roboflow.com/warehousedata/warehouse-objects-miafr)

- Total images: 7,695
- Number of classes: 9
- License: CC BY 4.0

### Object Categories

1. Box
2. Carrier
3. Crate
4. Forklift
5. Pallet
6. Pallet Truck
7. Person
8. Small Load Carrier
9. Stillage

The dataset contains training, validation and testing splits.

The original dataset was preserved. A separate cleaned copy was created by converting polygon-based segmentation annotations into bounding-box annotations for object detection training.

## 🤖 Model Training and Evaluation

Model architecture: YOLO11n

Training configuration:
- Epochs: 20
- Image size: 640 × 640
- Batch size: 4
- Execution device: CPU
- Detection classes: 9

### Test Dataset Performance

| Evaluation Metric | Result |
|---|---:|
| Precision | Not reported here |
| Recall | Not reported here |
| mAP@50 | 63.92% |
| mAP@50–95 | 36.42% |

Evaluation was performed on 656 test images containing 20,914 labelled objects.

These results represent the model's measured performance on the test split. They do not imply perfect detection in every image.

## 📊 Dashboard Features

The Streamlit dashboard currently supports:

- Uploading warehouse images.
- Adjusting the detection confidence threshold.
- Detecting warehouse objects.
- Displaying annotated images with bounding boxes.
- Calculating AI-predicted object counts.
- Displaying total detections and object categories.
- Identifying the most frequently detected class.
- Visualizing object distribution using Plotly charts.

### Important Note

Object counts represent AI predictions, not verified physical inventory.

Detection performance can vary depending on image quality, lighting, object overlap, occlusion, camera angle and confidence threshold.

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WarehouseVision-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the dashboard

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

### 6. Model and Dataset Setup

Place the trained model at:

```text
models/warehouse_yolo11_final.pt
```

Place the cleaned dataset and its `data.yaml` configuration in:

```text
data/processed/warehouse_yolo_clean/
```

The model and dataset must be available at their expected paths for the corresponding features to run.

## 🔍 Object Tracking

An experimental video-tracking workflow was tested using Ultralytics ByteTrack.

It supports tracking detected objects across video frames and associating detections with tracking IDs.

Tracking is a prototype capability and should not be confused with a fully integrated dashboard feature.

## ⚠️ Current Limitations

- Detection accuracy depends on the quality and diversity of input images.
- Occluded or overlapping people may be missed or detected incorrectly.
- Low-confidence detections may require threshold adjustment.
- Object counts are not equivalent to real-world inventory counts.
- Logistics optimization is a planned extension, not a completed production feature.
- The current model runs on CPU and may be slower for large images or videos.

## 🔮 Future Enhancements

- Real-time warehouse camera integration.
- Improved multi-object tracking.
- Detection accuracy improvements using additional training data.
- Warehouse activity and safety monitoring.
- Inventory movement analytics.
- Logistics route and resource optimization.
- Database integration for historical analytics.
- Cloud deployment.

## 👩‍💻 Project Information

**Project:** WarehouseVision AI

**Domain:** Artificial Intelligence and Machine Learning

**Specialization:** Computer Vision and Deep Learning

**Application:** Warehouse Object Detection and Analytics

**Framework:** YOLO11n + Streamlit

---

*Developed as an AI/ML portfolio project to explore computer vision applications in warehouse environments.*