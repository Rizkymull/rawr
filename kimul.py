# ======================================
# 📦 Import Library
# ======================================
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# ======================================
# ⚙️ Konfigurasi Halaman
# ======================================
st.set_page_config(page_title="Deteksi Buaya YOLOv8", layout="centered")
st.title("🧠 Sistem Deteksi Buaya YOLOv8")

# ======================================
# 🎨 CSS Desain (DIPERHALUS)
# ======================================
st.markdown("""
<style>
/* ==== Background ==== */
.stApp {
    background-image: url("https://raw.githubusercontent.com/Rizkymull/rawr/main/Asal/bg%201.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

/* ==== Overlay Lembut ==== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.45);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > div {
    position: relative;
    z-index: 1;
}

/* ==== Font & Warna ==== */
h1, h2, h3, h4, h5, h6 {
    color: #E6F4EA !important;
    font-family: "Poppins", sans-serif;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
}

p, span, label, div, li {
    color: #E0E5E9 !important;
    font-family: "Poppins", sans-serif;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
}

[data-testid="stMarkdownContainer"] strong {
    color: #C7F464 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

/* ==== Tombol ==== */
.stButton>button {
    background-color: #C7F464 !important;
    color: #1B1B1B !important;
    border-radius: 10px;
    font-weight: bold;
    font-family: "Poppins", sans-serif;
    box-shadow: 0 0 10px rgba(199, 244, 100, 0.5);
}
.stButton>button:hover {
    background-color: #A4D65E !important;
    color: white !important;
}

/* ==== Kotak Upload ==== */
section[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255, 255, 255, 0.1);
    border: 2px dashed #C7F464;
    border-radius: 12px;
    color: #E0E5E9;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# ⚠️ Peringatan Umum
# ======================================
st.markdown("""
<div style='background-color: rgba(255, 222, 100, 0.15);
            border: 1px solid #C7F464;
            padding: 10px 15px; border-radius: 10px;
            color: #F8F8E8; text-align: center;
            font-weight: bold; font-size: 17px;'>
⚠ Jika Anda melihat buaya di sekitar Anda, <b>jangan dekati!</b> 
Segera amankan diri dan hubungi pihak berwenang.
</div>
""", unsafe_allow_html=True)

# ======================================
# ☎️ Kontak Resmi (Perbaikan Warna & Font)
# ======================================
st.markdown("""
<div style="
    background: rgba(0, 0, 0, 0.55);
    padding: 22px;
    border-radius: 15px;
    color: #E6F4EA;
    font-size: 15px;
    box-shadow: 0 0 12px rgba(200, 255, 150, 0.2);
    line-height: 1.6;
">
    <h3 style="color:#C7F464; text-align:center; margin-bottom:10px;">📞 KONTAK RESMI DARURAT</h3>
    <p style="text-align:center; color:#DCE8DD; font-size:14px;">
        Hubungi instansi berikut jika menemukan buaya atau satwa liar berbahaya. <br>
        <b>Layanan tersedia 24 jam.</b>
    </p>
    <hr style="border:0.5px solid #A4D65E; margin:12px 0;">
    <!-- BKSDA -->
    <div style="margin-top:10px;">
        <b style="color:#C7F464;">🦎 Balai Konservasi Sumber Daya Alam (BKSDA)</b><br>
        ☎️ <b style="color:#F6FFCC;">(021)5704501</b><br>
        📸 <a href="https://www.instagram.com/ksdae.menlhk" target="_blank" style="color:#9efeff;">Instagram</a>
    </div>
    <hr style="border:0.5px solid #A4D65E; margin:15px 0;">
    <!-- DAMKAR -->
    <div>
        <b style="color:#FFD180;">🚒 Pemadam Kebakaran (DAMKAR)</b><br>
        ☎️ <b style="color:#FFF1C1;">113</b><br>
        📸 <a href="https://www.instagram.com/damkarindonesia" target="_blank" style="color:#9efeff;">Instagram</a>
    </div>
    <hr style="border:0.5px solid #A4D65E; margin:15px 0;">
    <!-- POLRI -->
    <div>
        <b style="color:#89D1FF;">👮 Kepolisian Negara Republik Indonesia (POLRI)</b><br>
        ☎️ <b style="color:#EAF6FF;">110</b><br>
        📸 <a href="https://www.instagram.com/divisihumaspolri" target="_blank" style="color:#9efeff;">Instagram</a>
    </div>
    <hr style="border:0.5px solid #C7F464; margin:15px 0;">
    <p style="text-align:center; font-size:13.5px; color:#F6F8E7;">
        ⚠️ <b>Darurat Satwa Liar:</b> Jangan coba menangkap atau mengusir sendiri.<br>
        Laporkan ke BKSDA atau aparat setempat untuk penanganan aman.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================
# 📦 LOAD MODEL
# ======================================
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

# ======================================
# 📸 INPUT GAMBAR / KAMERA
# ======================================
upload_mode = st.radio("Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"])

if upload_mode == "Unggah Gambar":
    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("📸 Ambil foto dari kamera")

# ======================================
# 🔍 DETEKSI YOLOv8
# ======================================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Gambar Input", use_container_width=True)

    st.subheader("🔍 Hasil Deteksi (YOLOv8)")
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

            # Informasi kontak tambahan
            st.markdown("""
            <div style='background-color: rgba(0, 70, 30, 0.45);
                        border: 1px solid #A4D65E; padding: 15px; border-radius: 12px;
                        color: #EAF8E1; font-size: 15px; margin-top: 20px;'>
            <b>Hubungi BKSDA Terdekat</b><br>
            Jika Anda menemukan buaya atau satwa liar berbahaya, segera hubungi:<br><br>
            • <b>BKSDA Aceh:</b> 0853-6283-6024<br>
            • <b>BKSDA Kalimantan Selatan:</b> 0812-4849-4950<br>
            • <b>BKSDA Sumatera Selatan:</b> 0812-7141-2141<br>
            • <b>BKSDA Jawa Timur:</b> 0822-3211-5200<br><br>
            🕐 Layanan 24 Jam
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error deteksi YOLO: {e}")
else:
    st.info("⬆ Silakan unggah gambar atau gunakan kamera terlebih dahulu.")

