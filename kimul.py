import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
from ultralytics import YOLO

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="🧠 Deteksi & Klasifikasi Gambar", layout="centered")

# ==========================
# Cache dan Load Model
# ==========================
@st.cache_resource
def load_models():
    """Memuat model YOLO dan model klasifikasi dari folder lokal"""
    # Pastikan folder model tersedia
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)

    # Path model disesuaikan dengan lokasi file kamu
    yolo_path = os.path.join(model_folder, "best.pt")
    classifier_path = os.path.join(model_folder, "muhammad rizki mulia_Laporan 2.h5")

    # Pastikan kedua file model ada
    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan di: {yolo_path}")
        st.stop()
    if not os.path.exists(classifier_path):
        st.error(f"❌ File classifier tidak ditemukan di: {classifier_path}")
        st.stop()

    # Load model
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(classifier_path)

    return yolo_model, classifier

# Panggil fungsi load_models
yolo_model, classifier = load_models()

# ==========================
# Label kelas
# ==========================
class_labels = {
    0: 'Cars',
    1: 'Planes',
    2: 'Trains'
}

# ==========================
# Fungsi Preprocessing
# ==========================
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================
# UI
# ==========================
st.title("🚀 Aplikasi Deteksi & Klasifikasi Gambar")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diupload', use_container_width=True)
else:
    image = None

# ==========================
# Mode 1: Deteksi Objek (YOLO)
# ==========================
if menu == "Deteksi Objek (YOLO)" and image is not None:
    try:
        st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
        results = yolo_model.predict(np.array(image))

        if len(results) > 0 and hasattr(results[0], "plot"):
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada hasil deteksi yang bisa ditampilkan.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat deteksi objek: {e}")

# ==========================
# Mode 2: Klasifikasi Gambar
# ==========================
elif menu == "Klasifikasi Gambar" and image is not None:
    try:
        st.subheader("🧠 Hasil Klasifikasi Gambar")

        img_array = preprocess_image(image)
        prediction = classifier.predict(img_array)[0]

        pred_idx = np.argmax(prediction)
        pred_conf = prediction[pred_idx] * 100
        pred_label = class_labels.get(pred_idx, "Unknown")

        if pred_conf >= 97.0:
            st.success(f"Label: **{pred_label}**")
            st.write(f"Confidence: **{pred_conf:.2f}%**")
        else:
            st.warning(f"Model tidak yakin. Confidence tertinggi: **{pred_conf:.2f}%**")

        # Confidence semua kelas
        df = pd.DataFrame({
            'Kelas': [class_labels[i] for i in range(len(prediction))],
            'Confidence': [round(p * 100, 2) for p in prediction]
        })
        st.bar_chart(df.set_index("Kelas"))

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
else:
    st.info("📷 Silakan unggah gambar terlebih dahulu.")



