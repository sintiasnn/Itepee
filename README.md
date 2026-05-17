# Itepee — Indonesia Tweet Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Django](https://img.shields.io/badge/Django-4.1-green?logo=django)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?logo=bootstrap)

Hate speech detection in Indonesian tweets using 4 TensorFlow/Keras deep learning models.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Overview

This system classifies Indonesian tweets into 4 aspects of hate speech using an ensemble of 4 deep learning models.

**Pipeline:**
1. **Preprocessing** — Text cleaning, slang normalization, stopword removal
2. **Detection** — Binary classification: hate speech or non-hate speech
3. **Multi-label Classification** — Target (Individual/Group), Type (Religion/Race/Physical/Gender/Other), Level (Weak/Moderate/Strong)
4. **Web Interface** — Django with Bootstrap 5

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
│   ├── Layout/main.html       # Main layout (navbar, footer)
│   ├── index.html             # Landing page
│   ├── predict.html           # Tweet input form
│   ├── hate_speech.html       # Result: HS detected
│   ├── non_hate_speech.html   # Result: safe
│   └── detail.html            # Probability detail + progress bar
├── Itepee/
│   ├── settings.py            # Django configuration
│   ├── urls.py                # URL routing
│   ├── asgi.py / wsgi.py
├── ItepeeApp/
│   ├── views.py               # Prediction & rendering logic
│   ├── preprocessor.py        # Text preprocessing (8 steps)
│   ├── models.py              # Database model (TweetModel)
│   ├── admin.py               # Django admin
│   ├── model/                 # 4 .h5 models + tokenizer pickle
│   └── data/                  # Slang dictionary, stopwords, HTML entities
├── static/
│   └── style.css              # Custom styling
├── .env.example
├── manage.py
└── README.md
```

## Dataset

- **Source**: [id-multi-label-hate-speech-and-abusive-language-detection](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) by Okky Ibrohim & Indra Budi
- **License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Labels:**

| # | Label | Description |
|---|---|---|
| 1 | HS | Hate speech |
| 2 | Abusive | Abusive language |
| 3 | HS_Individual | Targeted at individual |
| 4 | HS_Group | Targeted at group |
| 5 | HS_Religion | Religion |
| 6 | HS_Race | Race/ethnicity |
| 7 | HS_Physical | Physical/disability |
| 8 | HS_Gender | Gender/sexual orientation |
| 9 | HS_Weak | Weak level |
| 10 | HS_Moderate | Moderate level |
| 11 | HS_Strong | Strong level |

Paper: [Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter](https://www.aclweb.org/anthology/W19-3506.pdf) — ALW3 2019

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
If `satelit` environment doesn't exist:
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

Open **http://localhost:8000**

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)

Contributors:
- [Kaenova Mahendra](https://github.com/kaenova)
- [Kelvin Mulyawan](https://github.com/Kelniter)
- [Adrianus Wicaksono](https://github.com/chuck1z)
