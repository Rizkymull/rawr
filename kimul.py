import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import gdown
# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Path model disesuaikan dengan lokasi file
    yolo_model = YOLO("model/best.pt")  # Ganti sesuai struktur folder GitHub kamu
    classifier = tf.keras.models.load_model("model/muhammad rizki mulia_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])


if menu == "Deteksi Objek (YOLO)":
    try:
        # Jalankan deteksi objek
        results = yolo_model(img)
        
        # Pastikan ada hasil deteksi
        if len(results) > 0 and hasattr(results[0], "plot"):
            result_img = results[0].plot()  # numpy array BGR
            if result_img is not None:
                if cv2 is not None and result_img is not None:
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)
else:
    st.warning("⚠️ OpenCV tidak tersedia atau hasil deteksi kosong.")
            else:
                st.warning("⚠️ Tidak ada hasil deteksi yang bisa ditampilkan.")
        else:
            st.warning("⚠️ Model YOLO tidak mengembalikan hasil deteksi.")
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat deteksi objek: {e}")

    elif menu == "Klasifikasi Gambar":
        try:
            # Preprocessing untuk model klasifikasi
            img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            st.write(f"**Kelas:** {class_index}")
            st.write(f"**Probabilitas:** {confidence:.4f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat klasifikasi gambar: {e}")

