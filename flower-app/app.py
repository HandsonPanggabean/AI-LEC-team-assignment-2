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
FOLDER_UJI = BASE / "pengujian" / "gambar_uji"

SECTION_CONTOH = [
    ("Berhasil", ["daisy__03.jpg", "dandelion__02.jpg", "roses__02.jpg"]),
    ("Gagal", ["roses__sulit01.png", "sunflowers__sulit01.jpg", "tulips__02.jpg"]),
    ("Luar kelas", ["luar__anggrek01.jpg", "luar__daun05.jpg", "luar__kursi04.webp"]),
]

st.set_page_config(page_title="Klasifikasi Bunga", page_icon="🌼", layout="wide")

@st.cache_resource
def muat_model():
    return tf.keras.models.load_model(str(BASE / "model_optimasi.keras"), compile=False)

@st.cache_data
def muat_kelas():
    with open(BASE / "class_names.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def muat_thumbnail(nama):
    img = ImageOps.exif_transpose(Image.open(FOLDER_UJI / nama)).convert("RGB")
    img.thumbnail((300, 300))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
 
 
def kelas_sebenarnya(nama):
    prefix = nama.split("__")[0]
    return "luar kelas" if prefix == "luar" else prefix

if "contoh" not in st.session_state:
    st.session_state.contoh = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def pilih_contoh(nama):
    st.session_state.contoh = nama
    st.session_state.uploader_key += 1

def hapus_contoh():
    st.session_state.contoh = None

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

img = None
sumber = None
benar = None

with kol_kiri:
    file = st.file_uploader(
        "Pilih gambar", 
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{st.session_state.uploader_key}",
        on_change=hapus_contoh,
    )
    
    if file is not None:
        img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        sumber = file.name
    elif st.session_state.contoh is not None:
        path = FOLDER_UJI / st.session_state.contoh
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        sumber = st.session_state.contoh
        benar = kelas_sebenarnya(sumber)

    if img is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px">'
            f'<p style="text-align:center;color:gray">Gambar yang digunakan: {sumber}</p>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Atau klik salah satu contoh gambar uji (dari folder pengujian/gambar_uji):**")
    for judul, daftar in SECTION_CONTOH:
        st.markdown(f"##### {judul}")
        kolom = st.columns(3)
        for kol, nama in zip(kolom, daftar):
            with kol:
                if (FOLDER_UJI / nama).exists():
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{muat_thumbnail(nama)}" '
                        f'style="width:100%;border-radius:8px">',
                        unsafe_allow_html=True,
                    )
                    st.caption(nama)
                    st.button(
                        "Pilih",
                        key=f"btn_{nama}",
                        on_click=pilih_contoh,
                        args=(nama,),
                        use_container_width=True,
                    )
                else:
                    st.caption(f"{nama} (file tidak ditemukan)")

with kol_kanan:
    if img is None:
        st.info("Hasil prediksi akan muncul di sini setelah gambar diunggah atau contoh dipilih.")
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
 
        if benar is not None:
            if benar == "luar kelas":
                st.info(f"Kelas sebenarnya: **luar kelas** (bukan salah satu dari 5 bunga) → dipaksa menjadi **{kelas[idx]}**.")
            elif benar == kelas[idx]:
                st.success(f"Kelas sebenarnya: **{benar}** → prediksi benar.")
            else:
                st.error(f"Kelas sebenarnya: **{benar}** → prediksi salah.")
 
        if probs[idx] < 0.60:
            st.warning("Keyakinan rendah. Gambar mungkin bukan salah satu dari 5 jenis bunga yang dikenali model, atau fotonya kurang jelas.")
 
        st.write("Probabilitas semua kelas:")
        for k, p in sorted(zip(kelas, probs), key=lambda t: -t[1]):
            st.progress(float(p), text=f"{k} — {p:.1%}")
 
        st.bar_chart({k: float(p) for k, p in zip(kelas, probs)}, height=220)