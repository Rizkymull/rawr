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
    page_title="🐊 Deteksi dan Klasifikasi Buaya",
    page_icon="🐊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================
# GAYA & HEADER
# ==========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8f1;
    }
    h1 {
        color: #006400;
        text-align: center;
        font-weight: 800;
    }
    .warning-box {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid red;
        font-size: 16px;
        color: #900;
        margin-top: 20px;
    }
    footer {
        text-align: center;
        color: gray;
        font-size: 13px;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐊 Deteksi & Klasifikasi Jenis Buaya")
st.markdown(
    """
    Aplikasi ini menggunakan model **YOLOv8** dan **Keras** untuk mendeteksi serta mengenali jenis buaya seperti:
    **Crocodile, Alligator**, dan **Gharial**.
    """
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
        st.warning("⚠️ Model Keras (.h5) tidak ditemukan. Hanya YOLO yang digunakan.")
        keras_model = None
    else:
        keras_model = tf.keras.models.load_model(keras_path)

    yolo_model = YOLO(yolo_path)
    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# INPUT GAMBAR / KAMERA
# ==========================
st.subheader("📸 Unggah atau Ambil Foto Buaya")

tab1, tab2 = st.tabs(["📤 Upload Gambar", "🎥 Gunakan Kamera"])

uploaded_file = tab1.file_uploader("Unggah gambar dari perangkat", type=["jpg", "jpeg", "png"])
camera_file = tab2.camera_input("Ambil foto menggunakan kamera")

image_source = uploaded_file or camera_file

if image_source:
    img = Image.open(image_source).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diproses", use_container_width=True)

    # ==========================
    # YOLO DETECTION
    # ==========================
    st.subheader("🔍 Hasil Deteksi (YOLO)")
    try:
        results = yolo_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="📊 Deteksi Otomatis", use_container_width=True)

        boxes = results[0].boxes
        names = results[0].names

        if len(boxes) > 0:
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls_id = int(best_box.cls[0])
            yolo_label = names[cls_id]
            conf = float(best_box.conf[0])

            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area Deteksi", use_container_width=True)

            st.success(f"✅ Jenis Buaya Terdeteksi: **{yolo_label.capitalize()}** (Akurasi {conf*100:.2f}%)")

            # ==========================
            # PESAN EDUKATIF
            # ==========================
            st.markdown(
                f"""
                <div class="warning-box">
                ⚠️ <b>Peringatan!</b><br>
                Anda baru saja mendeteksi <b>{yolo_label.capitalize()}</b>.<br>
                Jika Anda melihat buaya di alam liar, <b>jangan dekati</b> dan segera hubungi petugas <b>BKSDA</b> setempat.
                <br><br>
                📞 Nomor Darurat: <b>0813-9999-1234</b><br>
                🌍 BKSDA Wilayah Anda Siap Membantu.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("Tidak ada objek buaya yang terdeteksi.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam proses deteksi: {e}")

else:
    st.info("⬆ Silakan unggah gambar atau ambil foto menggunakan kamera.")

# ==========================
# FOOTER
# ==========================
st.markdown(
    """
    <footer>
    Dibuat oleh <b>Muhammad Rizki Mulia</b> | Proyek Klasifikasi Buaya 🐊 2025<br>
    Menggunakan Streamlit + YOLOv8 + TensorFlow
    </footer>
    """,
    unsafe_allow_html=True
)
