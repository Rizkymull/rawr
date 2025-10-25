import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Deteksi Buaya YOLOv8", layout="centered")
st.title("🧠 Sistem Deteksi Buaya YOLOv8")

# ==================================
# 🎨 1️⃣ TAMBAHKAN KODE CSS DI SINI
# ==================================
st.markdown(
    """
    <style>
    /* Atur gambar latar belakang */
    .stApp {
        background-image: url("https://raw.githubusercontent.com/Rizkymull/rawr/main/Asal/bg 1.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        opacity: 0.95;
    }

    /* Efek overlay kabut lembut */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 30, 20, 0.3); /* hijau tua semi transparan */
        z-index: -1;
    }

    /* Warna teks disesuaikan agar kontras */
    h1, h2, h3, p, li, label {
        color: #f5f5f5 !important;
    }

    /* Gaya peringatan */
    .alert {
        background-color: rgba(255, 255, 0, 0.15);
        border: 1px solid #ffe100;
        padding: 10px 15px;
        border-radius: 10px;
        color: #fff700;
        text-align: center;
        font-weight: bold;
        font-size: 17px;
    }

    /* Kotak kontak BKSDA */
    .bksda-box {
        background-color: rgba(0, 70, 30, 0.4);
        border: 1px solid #2ecc71;
        padding: 15px;
        border-radius: 12px;
        color: #eaffea;
        font-size: 15px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==== ⚠️ Peringatan Keselamatan ====
st.markdown(
    """
    <div class="alert">
        ⚠️ Jika Anda melihat buaya di sekitar Anda, <b>jangan dekati!</b> 
        Segera amankan diri dan hubungi pihak berwenang.
    </div>
    """,
    unsafe_allow_html=True
)

# ==== ☎️ Kontak BKSDA ====
st.markdown(
    """
    <div class="bksda-box">
        <b>📞 Hubungi BKSDA Terdekat:</b><br>
        • BKSDA Kalimantan Selatan: <b>0813-4829-XXXX</b><br>
        • BKSDA Sumatera Selatan: <b>0821-3456-XXXX</b><br>
        • BKSDA Jawa Timur: <b>0812-7654-XXXX</b><br><br>
        <i>Layanan 24 Jam</i>
    </div>
    """,
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
# UPLOAD / KAMERA
# ==========================
upload_mode = st.radio("Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"])

if upload_mode == "Unggah Gambar":
    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("📸 Ambil foto dari kamera")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Gambar Input", use_container_width=True)

    st.subheader("🔍 Hasil Deteksi (YOLO)")
    try:
        results = yolo_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

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
            st.success(f"Objek terdeteksi: {yolo_label.upper()} (Akurasi: {conf*100:.2f}%)")

            # Box kontak BKSDA
            st.markdown("""
            <div class='contact-box'>
            <b>Hubungi BKSDA Terdekat</b><br>
            Jika Anda menemukan buaya atau satwa liar berbahaya, segera hubungi:<br><br>
            • <b>BKSDA Kalimantan Selatan:</b> 0813-4829-XXXX<br>
            • <b>BKSDA Sumatera Selatan:</b> 0821-3456-XXXX<br>
            • <b>BKSDA Jawa Timur:</b> 0812-7654-XXXX<br><br>
            🕐 Layanan 24 Jam
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error deteksi YOLO: {e}")
else:
    st.info("⬆ Silakan unggah gambar atau gunakan kamera terlebih dahulu.")
