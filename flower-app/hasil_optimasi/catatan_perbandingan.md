CATATAN PERBANDINGAN PERFORMA (validation set, 731 gambar)

- Baseline CNN Tugas 1 (dimuat dari model_cnn_bunga.keras): akurasi 66.76%, F1 macro 0.6655.
- Setelah transfer learning (MobileNetV2 + BN + Dropout): akurasi 84.27%.
- Setelah fine-tuning: akurasi 87.82%, F1 macro 0.8776.
- Peningkatan total: +21.07 poin persentase.
- Kelas dengan peningkatan F1 terbesar: roses (+0.3450).
- Kelas dengan F1 terendah setelah optimasi: roses (0.8649).
- Hyperparameter terbaik: learning rate=0.001, batch size=32, dropout=0.5.
- Jumlah epoch diatur oleh EarlyStopping; fine-tuning memakai learning rate 1e-5 pada 30 layer teratas.
- Waktu training: transfer learning 1094 detik, fine-tuning 1681 detik.
- Jumlah parameter: baseline 101,829 vs optimasi 2,269,509.
- Keterbatasan: model terbaik dipilih dan dievaluasi pada validation set yang sama, sehingga angka bisa sedikit optimistis.
  Hasil pengujian dengan gambar baru (Orang 3) menjadi evaluasi independen.
- Baseline dilatih ulang pada sesi Colab terpisah, sehingga akurasinya dapat sedikit berbeda dari angka di dokumen Tugas 1 (non-determinisme GPU).
