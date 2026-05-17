# Itepee — Indonesia Tweet Prediction

Deteksi ujaran kebencian (*hate speech*) pada tweet berbahasa Indonesia menggunakan 4 model deep learning (TensorFlow/Keras).

## Fitur

- Deteksi apakah tweet mengandung ujaran kebencian
- Klasifikasi **sasaran**: Individu / Kelompok
- Klasifikasi **jenis**: Agama, Ras, Fisik, Gender, Lainnya
- Klasifikasi **tingkat**: Lemah, Sedang, Kuat
- Tampilan probabilitas detail dengan progress bar

## Dataset

Dataset yang digunakan: [id-multi-label-hate-speech-and-abusive-language-detection](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) oleh Okky Ibrohim & Indra Budi.

Publikasi: [Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter](https://www.aclweb.org/anthology/W19-3506.pdf) — ALW3 2019.

Lisensi dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Tech Stack

- **Django 4** — Web framework
- **TensorFlow / Keras** — 4 model deep learning (`.h5`)
- **Pandas, NumPy** — Data processing
- **Bootstrap 5** — Frontend
- **SQLite** — Database (default development)

## Cara Jalankan

```bash
conda activate satelit
python manage.py migrate
python manage.py runserver
```

Buka `http://localhost:8000`.

> Butuh environment conda dengan TensorFlow terinstal. Jika ingin pakai versi TF terbaru (2.x+), pastikan `tf_keras` terinstal untuk kompatibilitas model `.h5`.

## Struktur Proyek

```
Itepee/
├── Frontend/               # Template HTML (Bootstrap 5)
│   ├── Layout/main.html    # Layout utama (navbar, footer)
│   ├── index.html          # Beranda
│   ├── predict.html        # Form input tweet
│   ├── hate_speech.html    # Hasil: terdeteksi HS
│   ├── non_hate_speech.html# Hasil: aman
│   └── detail.html         # Detail probabilitas
├── Itepee/                 # Konfigurasi Django
├── ItepeeApp/              # Aplikasi utama
│   ├── views.py            # Logic prediksi
│   ├── preprocessor.py     # Preprocessing teks
│   ├── models.py           # Model DB (TweetModel)
│   ├── model/              # 4 model .h5 + tokenizer
│   └── data/               # Kamus alay, stopword, dll
├── static/style.css        # Styling kustom
└── manage.py
```

## Kontributor

1. [Ni Putu Sintia Wati](https://github.com/sintiasnn)
2. [Kaenova Mahendra](https://github.com/kaenova)
3. [Kelvin Mulyawan](https://github.com/Kelniter)
4. [Adrianus Wicaksono](https://github.com/chuck1z)
