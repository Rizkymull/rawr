import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Deteksi & Klasifikasi Jenis Buaya",
    page_icon="🦎",
    layout="centered"
)

# ==============================
# Judul dan Deskripsi
# ==============================
st.markdown("""
<h1 style='text-align: center; color: #2E8B57;'>🦎 Deteksi & Klasifikasi Jenis Buaya</h1>
<p style='text-align: center;'>
Aplikasi ini menggunakan model <b>YOLOv8</b> dan <b>Keras</b> untuk mendeteksi serta mengenali jenis buaya seperti:
<b>Crocodile</b>, <b>Alligator</b>, dan <b>Gharial</b>.
</p>
""", unsafe_allow_html=True)

# ==============================
# Peringatan Utama
# ==============================
st.warning("⚠️ Jika Anda melihat buaya di sekitar Anda, **jangan dekati** dan segera hubungi pihak berwenang atau petugas satwa liar terdekat.")

# ==============================
# Pilihan Sumber Gambar
# ==============================
st.subheader("📸 Pilih Sumber Gambar")
st.write("Silakan pilih metode input di bawah ini untuk mengunggah atau mengambil foto buaya:")

mode_input = st.radio(
    "Pilih metode input:",
    ["📁 Upload Foto", "📷 Gunakan Kamera"],
    horizontal=True
)

uploaded_image = None

# ==============================
# Upload Foto dari Perangkat
# ==============================
if mode_input == "📁 Upload Foto":
    uploaded_image = st.file_uploader(
        "Unggah gambar dari perangkat Anda (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

# ==============================
# Gunakan Kamera
# ==============================
elif mode_input == "📷 Gunakan Kamera":
    uploaded_image = st.camera_input("Ambil foto menggunakan kamera Anda")

# ==============================
# Proses Gambar Jika Ada Input
# ==============================
if uploaded_image is not None:
    st.subheader("🖼️ Gambar yang Dipilih / Diambil")

    # Baca gambar
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar input", use_column_width=True)

    # ==============================
    # Proses Deteksi Menggunakan YOLOv8
    # ==============================
    st.subheader("🔍 Proses Deteksi")
    with st.spinner("Sedang memproses gambar dan mendeteksi buaya..."):
        # Ganti path model sesuai file kamu
        model = YOLO("model/buaya_yolov8.pt")  
        results = model.predict(np.array(img))

    # Tampilkan hasil deteksi
    st.success("✅ Deteksi berhasil dilakukan.")
    st.image(results[0].plot(), caption="Hasil Deteksi Buaya", use_column_width=True)

    # ==============================
    # Peringatan Tambahan
    # ==============================
    st.warning("""
    ⚠️ **Peringatan Keamanan:**
    Jika Anda menemukan buaya di lingkungan sekitar, **jangan dekati**,
    **jangan berusaha memberi makan atau menangkap**, dan segera **laporkan kepada pihak berwenang**.
    """)

# ==============================
# Pesan Informasi Jika Belum Input
# ==============================
else:
    st.info("↑ Silakan pilih mode input, lalu unggah atau ambil foto buaya untuk dianalisis.")

# ==============================
# Footer
# ==============================
st.markdown("""
<hr>
<center>
<small>
Dibuat oleh <b>Muhammad Rizki Mulya</b> | Proyek Klasifikasi Buaya 🐊 2025<br>
Menggunakan <b>Streamlit</b> + <b>YOLOv8</b> + <b>TensorFlow</b>
</small>
</center>
""", unsafe_allow_html=True)
