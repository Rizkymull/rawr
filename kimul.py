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
st.markdown("""
<style>
/* ===== Tema Utama ===== */
body {
    background-color: #2e473b;
    background-image: url("https://images.unsplash.com/photo-1581093588401-4b62a2e4d0e9?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
}

/* ===== Overlay lembut agar teks kontras ===== */
.reportview-container, .main, .block-container {
    background: rgba(46, 71, 59, 0.75) !important;
    border-radius: 18px;
    padding: 1.8em;
    color: #f2f9f3;
}

/* ===== Judul dengan ornamen rawa ===== */
h1 {
    color: #d0e3d1;
    text-align: center;
    font-size: 2.5em;
    padding: 15px;
    background: rgba(0, 60, 30, 0.5);
    border-radius: 15px;
    position: relative;
}
h1::before {
    content: "🐊 ";
}
h1::after {
    content: " 🌿";
}

/* ===== Subjudul ===== */
h2, h3 {
    color: #d0e3d1;
    text-align: center;
}

/* ===== Alert box ===== */
.alert-box {
    background-color: rgba(255, 243, 205, 0.9);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ffeeba;
    color: #3e2723;
    margin-bottom: 15px;
    box-shadow: 0 0 10px rgba(255,255,255,0.1);
}

/* ===== Box Kontak ===== */
.contact-box {
    background-color: rgba(208, 227, 209, 0.9);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #a5d6a7;
    color: #1b5e20;
    margin-top: 25px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
.contact-box::before {
    content: "📞 ";
    font-size: 1.2em;
    font-weight: bold;
}

/* ===== Tombol ===== */
div.stButton > button {
    background-color: #3e8e41;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: 0.3s;
    border: none;
}
div.stButton > button:hover {
    background-color: #2e7d32;
    transform: scale(1.05);
}

/* ===== Radio button ===== */
.css-1a32fsj div[role='radiogroup'] label {
    color: #f2f9f3 !important;
}

/* ===== Footer ===== */
.footer {
    margin-top: 40px;
    text-align: center;
    color: #b2dfdb;
    font-size: 14px;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# 🏞️ 2️⃣ TAMBAHKAN GAMBAR ILUSTRASI DI SINI
# ============================================
st.image(
    "https://png.pngtree.com/png-clipart/20231011/original/pngtree-swamp-landscape-vector-illustration-png-image_13136005.png",
    use_container_width=True,
    caption="Habitat alami buaya (ilustrasi)",
    output_format="PNG"
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
