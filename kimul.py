import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Gambar", layout="centered")
st.title("🧠 Klasifikasi Gambar Sederhana")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/best.pt"      # model YOLO (deteksi)
    keras_path = "model/muhammad rizki mulia_Laporan 2.h5"  # model Keras (klasifikasi)

    if not os.path.exists(yolo_path):
        st.error("❌ Model YOLO (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(keras_path):
        st.error("❌ Model Keras (.h5) tidak ditemukan.")
        st.stop()

    yolo_model = YOLO(yolo_path)
    keras_model = tf.keras.models.load_model(keras_path)
    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Input", use_container_width=True)

    # ==========================
    # YOLO DETECTION
    # ==========================
    st.subheader("🔍 Hasil Deteksi (YOLO)")
    try:
        results = yolo_model(img)                     # jalankan YOLO
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

        # Ambil bounding box terbaik (confidence tertinggi)
        boxes = results[0].boxes
        if len(boxes) > 0:
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area hasil deteksi (crop dari YOLO)", use_container_width=True)
        else:
            st.warning("Tidak ada objek terdeteksi, klasifikasi akan menggunakan gambar penuh.")
            cropped_img = img

    except Exception as e:
        st.error(f"❌ Error deteksi YOLO: {e}")
        # Jika deteksi gagal, kita hentikan proses klasifikasi selanjutnya
        cropped_img = None

    # ==========================
    # KERAS CLASSIFICATION (hanya jika cropped_img tersedia)
    # ==========================
    if cropped_img is not None:
        st.subheader("🔢 Hasil Klasifikasi (Keras)")
        try:
            # Pastikan urutan class_names sama dengan urutan saat training model Keras
            class_names = ["gharial", "alligator", "crocodile"]

            # Sesuaikan ukuran input dengan model kamu (contoh: 128x128)
            target_size = (128, 128)
            proc_img = cropped_img.resize(target_size)
            img_array = np.expand_dims(np.array(proc_img) / 255.0, axis=0)

            prediction = keras_model.predict(img_array)
            pred_index = np.argmax(prediction[0])
            pred_name = class_names[pred_index]
            confidence = np.max(prediction[0]) * 100

            st.success(f"Hasil Prediksi: **{pred_name.capitalize()}** 🐊 (Akurasi: {confidence:.2f}%)")

            # Debug (opsional): tampilkan skor mentah
            # st.write("Raw prediction:", prediction[0])

        except Exception as e:
            st.error(f"❌ Error klasifikasi Keras: {e}")

else:
    st.info("⬆ Silakan unggah gambar terlebih dahulu.")
