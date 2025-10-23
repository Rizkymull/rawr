pip install ultralytics tensorflow pillow numpy opencv-python-headless
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
pip install ultralytics tensorflow pillow numpy opencv-python-headless


# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Ganti path model sesuai file yang kamu upload
    yolo_model = YOLO("/mnt/data/57f21829-db26-4841-8055-021669ecf703.pt")  # YOLOv8 model
    classifier = tf.keras.models.load_model("/mnt/data/0ea8f8df-32bd-497d-80f2-82a0f7b3549c.h5")  # Keras classifier
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Jalankan deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (numpy array BGR)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing untuk model klasifikasi
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {class_index}")
        st.write(f"**Probabilitas:** {confidence:.4f}")
