import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Deteksi Buaya - YOLO App",
    page_icon="🐊",
    layout="centered",
)

# Gaya CSS tambahan untuk tampilan tema
st.markdown("""
<style>
body {
    background-color: #f1f8f4;
    font-family: "Segoe UI", sans-serif;
}
h1, h2, h3 {
    color: #155724;
}
div.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}
.alert-box {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #ffeeba;
    color: #856404;
    margin-bottom: 15px;
}
.contact-box {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #c3e6cb;
    color: #155724;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# JUDUL APLIKASI
# ==========================
st.title("🐊 Deteksi Buaya Menggunakan YOLOv8")
st.markdown(
    '<div class="alert-box">⚠️ Jika Anda melihat buaya di sekitar Anda, **jangan dekati**! '
    'Segera amankan diri Anda dan laporkan ke pihak berwenang.</div>',
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/best.pt"
    keras_path = "model/muhammad rizki mulia_Laporan 2.h5"

    if not os.path.exists(yolo_path):
        st.error("❌ Model YOLO (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(keras_path):
        st.warning("⚠ Model Keras (.h5) tidak ditemukan. Hanya YOLO yang digunakan.")
        keras_model = None
    else:
        keras_model = tf.keras.models.load_model(keras_path)

    yolo_model = YOLO(yolo_path)
    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# PILIH SUMBER GAMBAR
# ==========================
st.subheader("📸 Pilih Sumber Gambar")
option = st.radio("Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"], horizontal=True)

img = None

if option == "Unggah Gambar":
    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

elif option == "Gunakan Kamera":
    camera_file = st.camera_input("Ambil gambar menggunakan kamera")
    if camera_file:
        img = Image.open(camera_file).convert("RGB")

# ==========================
# PROSES DETEKSI
# ==========================
if img is not None:
    st.image(img, caption="🖼 Gambar Input", use_container_width=True)
    st.subheader("🔍 Hasil Deteksi (YOLO)")

    try:
        results = yolo_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="📦 Hasil Deteksi", use_container_width=True)

        boxes = results[0].boxes
        names = results[0].names

        if len(boxes) > 0:
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls_id = int(best_box.cls[0])
            yolo_label = names[cls_id]
            conf = float(best_box.conf[0])

            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area Deteksi (Crop dari YOLO)", use_container_width=True)

            st.success(f"✅ Objek terdeteksi: **{yolo_label.upper()}** (Akurasi: {conf*100:.2f}%)")

            if "buaya" in yolo_label.lower():
                st.warning("⚠️ Deteksi menunjukkan adanya *buaya*! Jangan dekati dan segera hubungi BKSDA!")

        else:
            st.info("🚫 Tidak ada objek terdeteksi di gambar ini.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat deteksi YOLO: {e}")

else:
    st.info("⬆ Silakan unggah gambar atau gunakan kamera terlebih dahulu.")

# ==========================
# KONTAK BKSDA
# ==========================
st.markdown(
    """
    <div class="contact-box">
        📞 <b>Hubungi BKSDA Terdekat</b><br>
        Jika Anda menemukan buaya atau satwa liar berbahaya, segera hubungi:
        <ul>
            <li><b>BKSDA Kalimantan Selatan</b>: 0813-4829-XXXX</li>
            <li><b>BKSDA Sumatera Selatan</b>: 0821-3456-XXXX</li>
            <li><b>BKSDA Jawa Timur</b>: 0812-7654-XXXX</li>
        </ul>
        🕐 Layanan 24 Jam
    </div>
    """,
    unsafe_allow_html=True
)
