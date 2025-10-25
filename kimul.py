import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

# ==========================
# Fungsi Load Model
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    except Exception as e:
        st.error(f"Gagal memuat YOLO model: {e}")
        yolo_model = None

    try:
        keras_model = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat Keras model: {e}")
        keras_model = None

    return yolo_model, keras_model


# ==========================
# Fungsi Prediksi
# ==========================
def predict_image(img, yolo_model, keras_model):
    if yolo_model is None or keras_model is None:
        st.warning("Model belum dimuat dengan benar.")
        return None, None

    # Simpan sementara gambar ke file untuk YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
        img.save(tmpfile.name)
        results = yolo_model(tmpfile.name)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
    
    # Ambil deteksi pertama saja (jika ada)
    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img = cropped_img.resize((224, 224))  # Sesuai input CNN

        # Ubah ke numpy array
        img_array = image.img_to_array(cropped_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Prediksi Keras Model
        pred = keras_model.predict(img_array)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred))
        return class_idx, confidence
    else:
        st.warning("Tidak ada objek terdeteksi oleh YOLO.")
        return None, None


# ==========================
# Aplikasi Streamlit
# ==========================
st.title("🔍 Deteksi & Klasifikasi Gambar")

# Load model sekali saja
yolo_model, keras_model = load_models()

# Pilihan input: Kamera atau Upload
input_choice = st.radio(
    "Pilih sumber gambar:",
    ["📸 Gunakan Kamera", "🖼️ Upload Foto"],
    index=1
)

if input_choice == "📸 Gunakan Kamera":
    uploaded_img = st.camera_input("Ambil gambar dengan kamera")
else:
    uploaded_img = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_img)
    st.image(img, caption="Gambar Diuji", use_container_width=True)

    # Tombol prediksi
    if st.button("🔎 Prediksi"):
        with st.spinner("Sedang memproses..."):
            class_idx, confidence = predict_image(img, yolo_model, keras_model)

            if class_idx is not None:
                st.success(f"Hasil Prediksi: Kelas {class_idx} (Confidence: {confidence:.2f})")
            else:
                st.error("Gagal melakukan prediksi.")


# Footer
st.markdown("---")
st.caption("© 2025 Sistem Deteksi & Klasifikasi Gambar - by Riski Mulya")
