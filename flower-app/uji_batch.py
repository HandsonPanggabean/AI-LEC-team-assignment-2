import sys, json, csv, pathlib
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "pengujian/gambar_uji")
kelas = json.load(open("class_names.json"))
model = tf.keras.models.load_model("model_optimasi.keras", compile=False)

rows = []
for f in sorted(folder.iterdir()):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue
    label = f.name.split("__")[0]
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    x = np.asarray(img.resize((128, 128), Image.NEAREST), dtype="float32") / 255.0
    p = model.predict(x[None], verbose=0)[0]
    o = np.argsort(p)[::-1]
    rows.append({
        "file": f.name,
        "jenis": "luar" if label == "luar" else ("sulit" if "sulit" in f.name else "normal"),
        "kelas_sebenarnya": label,
        "prediksi": kelas[o[0]], "skor": round(float(p[o[0]]) * 100, 1),
        "kandidat_2": kelas[o[1]], "skor_2": round(float(p[o[1]]) * 100, 1),
        "benar": label == kelas[o[0]],
    })

with open("hasil_uji.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

for r in rows:
    print(f'{r["file"]:32} {r["kelas_sebenarnya"]:11} -> {r["prediksi"]:11} {r["skor"]:5}%  {"OK" if r["benar"] else "SALAH"}')

for jenis in ("normal", "sulit"):
    s = [r for r in rows if r["jenis"] == jenis]
    if s:
        print(f"\nAkurasi {jenis}: {sum(r['benar'] for r in s)}/{len(s)} = {100*sum(r['benar'] for r in s)/len(s):.1f}%")
l = [r for r in rows if r["jenis"] == "luar"]
if l:
    print(f"Luar kelas: {sum(r['skor'] < 60 for r in l)}/{len(l)} memicu peringatan (<60%)")