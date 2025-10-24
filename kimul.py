import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Klasifikasi Gambar", layout="centered")
st.title("🧠 Klasifikasi Gambar Berdasarkan YOLO")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/best.pt"      # model YOLO
    keras_path = "model/muhammad rizki mulia_Laporan 2.h5"  # model Keras (opsional)

    if not os.path.exists(yolo_path):
        st.error("❌ Model YOLO (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(keras_path):
        st.warning("⚠️ Model Keras (.h5) tidak ditemukan. Hanya YOLO yang digunakan.")
        keras_model = None
    else:
        keras_model = tf.keras.models.load_model(keras_path)

    yolo_model = YOLO(yolo_path)
    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ==========================
    # YOLO DETECTION
    # ==========================
    st.subheader("🔍 Hasil Deteksi (YOLO)")
    try:
        results = yolo_model(img)
        annotated_img = results[0].plot()  # Gambar dengan bounding box
        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

        boxes = results[0].boxes
        names = results[0].names  # Daftar nama kelas YOLO

        if len(boxes) > 0:
            # Ambil hasil deteksi dengan confidence tertinggi
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls_id = int(best_box.cls[0])
            yolo_label = names[cls_id]
            conf = float(best_box.conf[0])

            # Crop area hasil deteksi
            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area hasil deteksi (crop dari YOLO)", use_container_width=True)

            # ==========================
            # HASIL KLASIFIKASI (SINKRON YOLO)
            # ==========================
            st.subheader("🔢 Hasil Klasifikasi (Mengikuti YOLO)")
            st.success(f"Hasil Prediksi: **{yolo_label.capitalize()}** 🐊 (Akurasi YOLO: {conf*100:.2f}%)")

        else:
            st.warning("Tidak ada objek terdeteksi. Klasifikasi tidak dilakukan.")

    except Exception as e:
        st.error(f"❌ Error deteksi YOLO: {e}")

else:
    st.info("⬆ Silakan unggah gambar terlebih dahulu.")

