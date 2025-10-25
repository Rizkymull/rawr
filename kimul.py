import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

st.set_page_config(page_title="Deteksi Buaya YOLOv8", layout="centered")
st.title("🧠 Sistem Deteksi Buaya YOLOv8")

# ==================== CSS ====================
st.markdown("""
<style>
.stApp {
    background-image: url("https://raw.githubusercontent.com/Rizkymull/rawr/main/Asal/bg%201.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > div { position: relative; z-index: 1; }
[data-testid="stAppViewContainer"] * { position: relative; z-index: 2; }
h1, h2, h3, p, span, label, div, li {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.9);
    font-family: "Poppins", sans-serif;
}
.stButton>button {
    background-color: #FFD700 !important;
    color: black !important;
    border-radius: 10px;
    font-weight: bold;
    box-shadow: 0px 0px 10px rgba(255,215,0,0.5);
}
.stButton>button:hover {
    background-color: #FFA500 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== Peringatan ====================
st.markdown("""
<div style='background-color: rgba(255,255,0,0.15); border: 1px solid #ffe100;
            padding: 10px 15px; border-radius: 10px; color: #fff700; text-align: center;
            font-weight: bold; font-size: 17px;'>
⚠ Jika Anda melihat buaya di sekitar Anda, <b>jangan dekati!</b> Segera amankan diri dan hubungi pihak berwenang.
</div>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource(show_spinner="Memuat model YOLOv8 dan Keras...")
def load_models():
    yolo_path = "model/best.pt"
    keras_path = "model/muhammad rizki mulia_Laporan 2.h5"

    if not os.path.exists(yolo_path):
        st.error("❌ Model YOLO (.pt) tidak ditemukan.")
        st.stop()

    yolo_model = YOLO(yolo_path)

    if os.path.exists(keras_path):
        try:
            keras_model = tf.keras.models.load_model(keras_path)
        except Exception as e:
            st.warning(f"⚠ Gagal memuat model Keras (.h5): {e}")
            keras_model = None
    else:
        keras_model = None

    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==================== Upload ====================
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
        results = yolo_model.predict(source=np.array(img), verbose=False)
        annotated_img = results[0].plot()
        st.image(annotated_img[:, :, ::-1], caption="Hasil Deteksi", use_container_width=True)

        boxes = results[0].boxes
        names = yolo_model.names

        if len(boxes) > 0:
            confs = [float(b.conf[0]) for b in boxes]
            best_box = boxes[np.argmax(confs)]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
            cls_id = int(best_box.cls[0])
            yolo_label = names[cls_id]
            conf = float(best_box.conf[0])

            cropped_img = img.crop((x1, y1, x2, y2))
            st.image(cropped_img, caption="🧩 Area Deteksi (Crop dari YOLO)", use_container_width=True)
            st.success(f"Objek terdeteksi: {yolo_label.upper()} (Akurasi: {conf*100:.2f}%)")
        else:
            st.warning("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error deteksi YOLO: {e}")
else:
    st.info("⬆ Silakan unggah gambar atau gunakan kamera terlebih dahulu.")

