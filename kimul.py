import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Deteksi & Klasifikasi Buaya", layout="centered")

# ==========================
# CSS ANIMASI BACKGROUND RAWA
# ==========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.ibb.co/q0V7Dvq/swamp-bg.gif");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    animation: fadein 2s;
}
@keyframes fadein {
  from {opacity: 0;}
  to {opacity: 1;}
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==========================
# JUDUL APLIKASI
# ==========================
st.markdown("<h1 style='text-align:center; color:#004d00;'>🐊 Deteksi & Klasifikasi Jenis Buaya</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white;'>Aplikasi ini menggunakan model <b>YOLOv8</b> dan <b>Keras</b> untuk mendeteksi serta mengenali jenis buaya seperti <b>Crocodile</b>, <b>Alligator</b>, dan <b>Gharial</b>.</p>", unsafe_allow_html=True)
st.write("---")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/best.pt"  # model YOLO utama
    keras_path = "model/muhammad rizki mulia_Laporan 2.h5"  # model Keras tambahan

    if not os.path.exists(yolo_path):
        st.error(f"❌ File model YOLO tidak ditemukan di: {yolo_path}")
        st.stop()

    yolo_model = YOLO(yolo_path)

    if os.path.exists(keras_path):
        keras_model = tf.keras.models.load_model(keras_path)
    else:
        keras_model = None
        st.warning("⚠️ Model Keras tidak ditemukan. Hanya YOLO yang digunakan.")

    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# PILIH SUMBER GAMBAR
# ==========================
st.subheader("📸 Pilih Sumber Gambar")
input_option = st.radio("Pilih metode input:", ["Upload Foto", "Gunakan Kamera"], horizontal=True)

if input_option == "Upload Foto":
    uploaded_file = st.file_uploader("📤 Unggah Gambar Buaya", type=["jpg", "jpeg", "png"])
elif input_option == "Gunakan Kamera":
    uploaded_file = st.camera_input("📷 Ambil Foto dari Kamera")
else:
    uploaded_file = None

# ==========================
# PROSES DETEKSI DAN KLASIFIKASI
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Input", use_container_width=True)
    st.subheader("🔍 Proses Deteksi")

    try:
        results = yolo_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Hasil Deteksi YOLO", use_container_width=True)

        boxes = results[0].boxes
        names = results[0].names

        if len(boxes) > 0:
            # Ambil hasil deteksi confidence tertinggi
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls_id = int(best_box.cls[0])
            yolo_label = names[cls_id]
            conf = float(best_box.conf[0])

            # Crop area hasil deteksi
            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area Hasil Deteksi", use_container_width=True)

            # ==========================
            # HASIL AKHIR
            # ==========================
            st.success(f"✅ Jenis Buaya Terdeteksi: **{yolo_label.capitalize()}** (Akurasi: {conf*100:.2f}%)")

            st.markdown(
                """
                <div style='background:rgba(0,0,0,0.6); padding:15px; border-radius:10px; margin-top:20px; text-align:center; color:white;'>
                    ⚠️ <b>Jika Anda Melihat Buaya, Jangan Dekati!</b><br>
                    Segera hubungi <b>BKSDA</b> atau pihak berwenang setempat untuk penanganan lebih lanjut.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("🚫 Tidak ada objek buaya terdeteksi. Pastikan gambar jelas.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat deteksi: {e}")

else:
    st.info("⬆ Pilih mode input terlebih dahulu, lalu unggah atau ambil foto buaya.")

# ==========================
# FOOTER
# ==========================
st.markdown("<br><hr><p style='text-align:center; color:#e0ffe0;'>Dibuat oleh <b>Muhammad Rizki Mulia</b> | Proyek Klasifikasi Buaya 🐊 2025<br>Menggunakan Streamlit + YOLOv8 + TensorFlow</p>", unsafe_allow_html=True)
