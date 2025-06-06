import streamlit as st
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Konfigurasi halaman
st.set_page_config(page_title="Aerial Vehicle Detection", layout="centered")
st.title("🚗 Vehicle Detection on Aerial Images")

# Inisialisasi session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "boxes" not in st.session_state:
    st.session_state.boxes = []
if "original_img" not in st.session_state:
    st.session_state.original_img = None
if "pred_img" not in st.session_state:
    st.session_state.pred_img = None
if "model_names" not in st.session_state:
    st.session_state.model_names = {}

# Upload gambar dan pilih model
uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])
model_option = st.selectbox("Select YOLOv11 Model", ["yolov11s", "yolov11n"])

# Tampilkan gambar yang di-upload
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Simpan gambar ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        # Load model dan prediksi
        model_path = os.path.join("models", f"{model_option}.pt")
        model = YOLO(model_path)
        results = model(temp_path)
        result = results[0]

        # Simpan data ke session state
        st.session_state.original_img = cv2.imread(temp_path)
        st.session_state.boxes = result.boxes
        st.session_state.pred_img = result.plot()
        st.session_state.model_names = model.names
        st.session_state.step = 0

        os.unlink(temp_path)  # hapus file temp

# Tampilkan hasil prediksi
if st.session_state.boxes:
    boxes = st.session_state.boxes
    total = len(boxes)

    # Step = 0 artinya tampilkan semua objek
    if st.session_state.step == 0:
        st.image(st.session_state.pred_img, caption="All Objects Detected", use_container_width=True)
        st.caption(f"Showing all {total} detected objects.")
    else:
        idx = st.session_state.step - 1
        if idx < total:
            box = boxes[idx]
            img_copy = st.session_state.original_img.copy()

            # Ambil koordinat dan label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            label = f"{st.session_state.model_names[cls_id]} {conf:.2f}"

            # Gambar bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            st.image(img_copy, caption=f"Object {idx + 1}/{total}", use_container_width=True)
        else:
            st.session_state.step = 0  # reset jika index melebihi jumlah box

    # Navigasi antar objek
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.step -= 1
            if st.session_state.step < 0:
                st.session_state.step = total  # kembali ke semua objek

    with col2:
        if st.button("➡️ Next"):
            st.session_state.step += 1
            if st.session_state.step > total:
                st.session_state.step = 0  # kembali ke semua objek
