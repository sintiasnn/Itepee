# Itepee — Indonesia Tweet Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Django](https://img.shields.io/badge/Django-4.1-green?logo=django)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?logo=bootstrap)

Deteksi ujaran kebencian (*hate speech*) pada tweet berbahasa Indonesia menggunakan 4 model deep learning TensorFlow/Keras.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Overview

Sistem ini mengklasifikasikan tweet bahasa Indonesia ke dalam 4 aspek ujaran kebencian menggunakan ensemble 4 model deep learning.

**Pipeline stages:**
1. **Preprocessing** — Pembersihan teks, normalisasi alay, stopword removal
2. **Deteksi** — Klasifikasi biner: hate speech atau non-hate speech
3. **Klasifikasi Multi-label** — Sasaran (Individu/Kelompok), Jenis (Agama/Ras/Fisik/Gender/Lainnya), Tingkat (Lemah/Sedang/Kuat)
4. **Web Interface** — Django dengan Bootstrap 5

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Web Framework | Django 4.1 |
| Deep Learning | TensorFlow / Keras (4 models) |
| Frontend | Bootstrap 5, Inter font |
| Database | SQLite (dev) / MySQL |
| Data Processing | Pandas, NumPy |

## Project Structure

```
Itepee/
├── Frontend/
│   ├── Layout/main.html       # Layout utama (navbar, footer)
│   ├── index.html             # Beranda
│   ├── predict.html           # Form input tweet
│   ├── hate_speech.html       # Hasil: terdeteksi HS
│   ├── non_hate_speech.html   # Hasil: aman
│   └── detail.html            # Detail probabilitas + progress bar
├── Itepee/
│   ├── settings.py            # Konfigurasi Django
│   ├── urls.py                # Routing URL
│   ├── asgi.py / wsgi.py
├── ItepeeApp/
│   ├── views.py               # Logic prediksi & rendering
│   ├── preprocessor.py        # Preprocessing teks (8 tahap)
│   ├── models.py              # Model database (TweetModel)
│   ├── admin.py               # Admin Django
│   ├── model/                 # 4 model .h5 + tokenizer pickle
│   └── data/                  # Kamus alay, stopword, HTML entities
├── static/
│   └── style.css              # Styling kustom
├── .env.example
├── manage.py
└── README.md
```

## Dataset

- **Source**: [id-multi-label-hate-speech-and-abusive-language-detection](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) oleh Okky Ibrohim & Indra Budi
- **Labels**:

| # | Label | Description |
|---|---|---|
| 1 | HS | Hate speech / ujaran kebencian |
| 2 | Abusive | Bahasa abusive |
| 3 | HS_Individual | Sasaran individu |
| 4 | HS_Group | Sasaran kelompok |
| 5 | HS_Religion | Agama |
| 6 | HS_Race | Ras |
| 7 | HS_Physical | Fisik/disabilitas |
| 8 | HS_Gender | Gender/orientasi seksual |
| 9 | HS_Weak | Tingkat lemah |
| 10 | HS_Moderate | Tingkat sedang |
| 11 | HS_Strong | Tingkat kuat |

Publikasi: [Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter](https://www.aclweb.org/anthology/W19-3506.pdf) — ALW3 2019

Lisensi dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Getting Started

### 1. Clone

```bash
git clone https://github.com/sintiasnn/Itepee.git
cd Itepee
```

### 2. Setup Environment

<details>
<summary>Using conda (recommended)</summary>

```bash
conda activate satelit
```
Jika environment `satelit` belum ada:
```bash
conda create -n satelit python=3.11 -y
conda activate satelit
pip install tensorflow django pandas numpy python-decouple
pip install tf_keras keras-preprocessing
```
</details>

<details>
<summary>Using venv</summary>

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
</details>

### 3. Migrate Database

```bash
python manage.py migrate
```

### 4. Run Server

```bash
python manage.py runserver
```

Buka **http://localhost:8000**

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)

Dibantu oleh:
- [Kaenova Mahendra](https://github.com/kaenova)
- [Kelvin Mulyawan](https://github.com/Kelniter)
- [Adrianus Wicaksono](https://github.com/chuck1z)
