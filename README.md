# ☕ Sistem Rekomendasi Kafe Bandung

Sistem rekomendasi kafe di Kota Bandung untuk tempat belajar, berbasis ulasan Google Maps dan Natural Language Processing + Large Language Model.

**Kelompok 30 — Capstone Project**

---

## ✨ Fitur Utama

- **Preprocessing Bahasa Indonesia** — pembersihan teks, normalisasi slang, stopwords removal (Sastrawi)
- **Sentiment Analysis** — klasifikasi positif/negatif/netral dengan lexicon + rating + handling negasi
- **Feature Extraction** — ekstraksi 10 fitur fasilitas (WiFi, stopkontak, kenyamanan, dll) dengan keyword matching + Gemini LLM (opsional)
- **Content-Based Recommendation** — cosine similarity antara preferensi user dan profil kafe + boost dari rating & sentimen
- **Database SQLite** — schema relasional (cafes, reviews, features) untuk persistensi
- **Web App Streamlit** — UI interaktif dengan 5 halaman: Beranda, Cari Rekomendasi, Detail Kafe, Analytics, Tentang

## 📁 Struktur Project

```
Code/
├── app.py                          # ← Streamlit UI utama
├── requirements.txt                # Python dependencies
├── .env.example                    # Template environment variables
├── README.md                       # Dokumentasi (ini)
│
├── data/
│   └── dataset_final.csv           # Dataset utama (13,664 ulasan)
│
├── outputs/                        # Output pipeline (auto-generated)
│   ├── cleaned_reviews.csv         # Hasil preprocessing
│   ├── reviews_with_sentiment.csv  # Hasil sentiment analysis
│   ├── cafe_features.csv           # Profil fitur per kafe
│   └── cafe_recommender.db         # SQLite database
│
├── src/                            # Source code modular
│   ├── config.py                   # Konfigurasi global
│   ├── preprocessing.py            # Cleaning, stopwords, slang norm
│   ├── sentiment_analysis.py       # Klasifikasi sentimen
│   ├── feature_extraction.py       # Keyword + Gemini LLM
│   ├── recommender.py              # Content-based recommender
│   └── database.py                 # SQLite operations
│
└── scripts/                        # Pipeline scripts
    ├── 01_preprocess.py            # Step 1
    ├── 02_sentiment.py             # Step 2
    ├── 03_extract_features.py      # Step 3
    ├── 04_build_db.py              # Step 4
    └── run_all.py                  # Run semua step sekaligus
```

---

## 🚀 Quick Start

### 1) Setup Python Environment

Buka terminal di folder `Code/` (di VSCode: `View → Terminal`).

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) (Opsional) Setup Gemini API Key

Tanpa API key, sistem tetap jalan dengan keyword extraction. Untuk hasil ekstraksi yang lebih akurat:

1. Buka https://aistudio.google.com/app/apikey
2. Login dengan akun Google
3. Klik **Create API Key** → pilih atau buat project
4. Copy API key
5. Salin file `.env.example` ke `.env`:
   ```bash
   # Windows
   copy .env.example .env

   # Mac/Linux
   cp .env.example .env
   ```
6. Edit `.env`:
   ```
   GEMINI_API_KEY=AIza...isi_key_anda...
   FEATURE_EXTRACTION_MODE=hybrid
   ```

### 3) Jalankan Pipeline (sekali saja)

```bash
python scripts/run_all.py
```

Pipeline akan:
1. Membersihkan 13,664 ulasan → ~9,400 ulasan unik
2. Klasifikasi sentimen
3. Ekstraksi fitur per kafe
4. Build database SQLite

> **Note**: Database `outputs/cafe_recommender.db` sudah di-prebuilt jadi Anda bisa skip step ini dan langsung ke step 4.

### 4) Jalankan Web App

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`. Selesai! 🎉

---

## 🧪 Cara Menjalankan Step Individual

```bash
python scripts/01_preprocess.py        # Hanya preprocessing
python scripts/02_sentiment.py         # Hanya sentiment
python scripts/03_extract_features.py  # Hanya ekstraksi fitur
python scripts/04_build_db.py          # Hanya build DB
```

---

## 🛠️ Konfigurasi (.env)

```ini
# Wajib jika ingin pakai LLM
GEMINI_API_KEY=

# Model Gemini (default cukup untuk free tier)
GEMINI_MODEL=gemini-2.0-flash-exp

# Mode ekstraksi fitur:
#   keyword - keyword matching saja (gratis, cepat)
#   llm     - hanya Gemini LLM
#   hybrid  - keyword + Gemini blend (paling akurat)
FEATURE_EXTRACTION_MODE=keyword
```

---

## 🔍 Menggunakan Sistem

### Halaman "🏠 Beranda"
Dashboard ringkasan: total kafe, total review, distribusi sentimen, top 10 kafe.

### Halaman "🔍 Cari Rekomendasi"
1. Pilih wilayah Bandung
2. Atur rating minimum & jumlah review minimum
3. Geser slider tiap fitur (0 = tidak penting, 1 = sangat penting)
4. Klik **Cari Rekomendasi**
5. Sistem menampilkan top-N kafe + alasan kenapa cocok

### Halaman "📋 Detail Kafe"
Pilih kafe → lihat profil fitur, semua ulasan, dan filter berdasarkan sentimen.

### Halaman "📊 Analytics"
Distribusi fitur, rating per wilayah, top kafe per fitur, korelasi antar fitur.

---

## 📊 Statistik Dataset (Setelah Preprocessing)

| Metrik | Nilai |
|---|---|
| Ulasan awal | 13,664 |
| Setelah cleaning | ~9,400 |
| Kafe unik | 161 |
| Wilayah | 5 (Utara/Selatan/Timur/Barat/Tengah) |
| % Positif | ~91% |
| % Netral | ~4% |
| % Negatif | ~5% |

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
→ Jalankan dari folder `Code/`, bukan dari folder lain. Atau aktifkan virtual environment dulu.

**`database is locked` saat run pipeline**
→ Tutup Streamlit dulu sebelum rebuild database.

**Streamlit lambat saat pertama kali load**
→ Wajar, library Sastrawi & data loading membutuhkan beberapa detik. Setelah cache terisi akan jauh lebih cepat.

**`OperationalError: disk I/O error` di SQLite**
→ Hapus file `outputs/cafe_recommender.db-journal` jika ada, lalu jalankan ulang.

---

## 📚 Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Sastrawi** — Bahasa Indonesia stopwords & stemmer
- **scikit-learn** — TF-IDF, cosine similarity (di recommender custom)
- **google-generativeai** — Gemini API client
- **Streamlit + Plotly** — UI
- **SQLite** — database (built-in Python)

---

## 👥 Tim Kelompok 30

| Nama | NIM | Peran |
|---|---|---|
| Zahra Nur Azizah | 1305223007 | Data Scraping & NLP |
| Winston Lokeswara Mangori | 103052300005 | NLP & LLM |
| Yan Andhinaya Ardika | 103052300062 | Recommendation System & Database |
| Yohana Lydia | 103052330068 | Backend & API |
| Yayang Ananda Setya | 103052300096 | Frontend & UI |
