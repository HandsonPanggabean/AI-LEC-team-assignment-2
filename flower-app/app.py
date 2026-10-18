import base64
import io
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path

IMG_SIZE = (128, 128)
BASE = Path(__file__).parent

st.set_page_config(page_title="Klasifikasi Bunga", page_icon="🌼", layout="wide")

@st.cache_resource
def muat_model():
    return tf.keras.models.load_model(str(BASE / "model_optimasi.keras"), compile=False)

@st.cache_data
def muat_kelas():
    with open(BASE / "class_names.json", encoding="utf-8") as f:
        return json.load(f)

model = muat_model()
kelas = muat_kelas()

st.title("🌼 Klasifikasi Bunga - AI LEC Team Assignment 2 (Group 8)")
with st.expander("Members"):
    st.markdown(
            """
    | Name | NIM |
    |---|---|
    | Khenaro Daffa Asyrof | 2902824590 |
    | Irfan Siswara | 2902813126 |
    | Muhammad Bayu Setiawan | 2602293852 |
    | Handson Panggabean | 2902820005 |
    """
    )
st.write("Unggah foto bunga, lalu model CNN (MobileNetV2) akan memprediksi jenisnya: daisy, dandelion, roses, sunflowers, atau tulips.")

kol_kiri, kol_kanan = st.columns(2)

with kol_kiri:
    file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px">'
            f'<p style="text-align:center;color:gray">Gambar yang diunggah</p>',
            unsafe_allow_html=True,
        )

with kol_kanan:
    if file is None:
        st.info("Hasil prediksi akan muncul di sini setelah gambar diunggah.")
    else:
        # Preprocessing sama dengan saat training: resize 128x128, nilai 0-1
        # (layer Rescaling untuk MobileNetV2 sudah ada di dalam model)
        x = np.asarray(img.resize(IMG_SIZE, Image.NEAREST), dtype="float32") / 255.0
        x = np.expand_dims(x, axis=0)

        with st.spinner("Memprediksi..."):
            probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))

        st.subheader(f"Prediksi: {kelas[idx]}")
        st.metric("Tingkat keyakinan", f"{probs[idx]:.1%}")

        if probs[idx] < 0.60:
            st.warning("Keyakinan rendah. Gambar mungkin bukan salah satu dari 5 jenis bunga yang dikenali model, atau fotonya kurang jelas.")

        st.write("Probabilitas semua kelas:")
        for k, p in sorted(zip(kelas, probs), key=lambda t: -t[1]):
            st.progress(float(p), text=f"{k} — {p:.1%}")

        st.bar_chart({k: float(p) for k, p in zip(kelas, probs)}, height=220)