# 🚗 Aerial Vehicle Detection with YOLOv11s/n and Streamlit

A simple web application built using **Streamlit** to detect vehicles in aerial imagery using custom-trained **YOLOv11s** and **YOLOv11n** models from Ultralytics.

## 📸 Features

- Upload aerial images for object detection
- Choose between two YOLOv11 models: `yolov11s` and `yolov11n`
- See original and predicted image with bounding boxes
- Built using `ultralytics` library for inference

---

## 🗂️ Project Structure

```
project/
├── app.py                  # Main Streamlit UI
├── models/                 # Folder containing trained YOLO models
│   ├── yolov11s.pt
│   └── yolov11n.pt
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/aerial-vehicle-detection.git
cd aerial-vehicle-detection
```

### 2. Install Requirements

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

---

## 🧠 Model Information

The application supports two custom-trained YOLO models:

- **YOLOv11s**: Small model, faster inference
- **YOLOv11n**: Nano model, lightest version

These models were trained using the [Ultralytics CLI](https://docs.ultralytics.com/) with a custom aerial vehicle dataset.

Example command used for training:

```bash
yolo detect train data=data.yaml model=yolov11s.pt imgsz=640 epochs=30
```

Make sure to rename the trained model as `yolov11s.pt` or `yolov11n.pt` and place it in the `models/` directory.

---

## 📁 Notes

- Ensure that uploaded images are clear and contain vehicles from aerial perspective.
- The model must be compatible with Ultralytics v8+. You can download pretrained weights or train your own.

---

## 🧩 Dependencies

- Python ≥ 3.8
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- Streamlit
- OpenCV
- Pillow

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact the project maintainer.

---

© 2025 – YOLOv11 Vehicle Detection Streamlit UI
