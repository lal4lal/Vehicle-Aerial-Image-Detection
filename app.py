import streamlit as st
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Aerial Vehicle Detection", layout="centered")
st.title("🚗 Vehicle Detection on Aerial Images")

# Upload image
uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])

# Model selection
model_option = st.selectbox("Select YOLOv11 Model", ["yolov11s", "yolov11n"])

# Predict button
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Simpan ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        # Load model
        model_path = os.path.join("models", f"{model_option}.pt")
        model = YOLO(model_path)

        # Prediksi
        results = model(temp_path)

        # Tampilkan hasil prediksi
        result_img = results[0].plot()  # hasil sebagai ndarray
        st.image(result_img, caption="Detected Vehicles", use_container_width=True)

        # Opsi simpan hasil
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        os.unlink(temp_path)
