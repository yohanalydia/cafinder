"""
Konfigurasi global untuk seluruh sistem.
Path, konstanta, dan setting environment dikumpulkan di sini.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Project paths
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"

# Pastikan folder outputs ada
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Path file utama
RAW_DATASET = DATA_DIR / "dataset_final.csv"
CLEANED_REVIEWS = OUTPUTS_DIR / "cleaned_reviews.csv"
SENTIMENT_REVIEWS = OUTPUTS_DIR / "reviews_with_sentiment.csv"
CAFE_FEATURES = OUTPUTS_DIR / "cafe_features.csv"
CAFE_PROFILES = OUTPUTS_DIR / "cafe_profiles.csv"
DATABASE_FILE = OUTPUTS_DIR / "cafe_recommender.db"

# =============================================================================
# Load .env
# =============================================================================
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp").strip()
FEATURE_EXTRACTION_MODE = os.getenv("FEATURE_EXTRACTION_MODE", "keyword").strip().lower()

GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    ""
).strip()

# =============================================================================
# Fitur target (sesuai proposal)
# =============================================================================
TARGET_FEATURES: list[str] = [
    "wifi",            # Ketersediaan WiFi
    "stopkontak",      # Ketersediaan stopkontak
    "kenyamanan",      # Tempat nyaman untuk belajar
    "suasana_tenang",  # Suasana tenang
    "harga_terjangkau",# Harga terjangkau
    "makanan_enak",    # Kualitas makanan/minuman
    "pelayanan_baik",  # Pelayanan ramah
    "tempat_luas",     # Ruangan luas/lega
    "ber_AC",          # Ber-AC / dingin
    "buka_lama",       # Buka 24 jam atau sampai malam
]

# Label yang dipakai di UI (lebih ramah pengguna)
FEATURE_LABELS = {
    "wifi": "WiFi tersedia",
    "stopkontak": "Banyak stopkontak",
    "kenyamanan": "Nyaman untuk belajar",
    "suasana_tenang": "Suasana tenang",
    "harga_terjangkau": "Harga terjangkau",
    "makanan_enak": "Makanan/minuman enak",
    "pelayanan_baik": "Pelayanan ramah",
    "tempat_luas": "Tempat luas",
    "ber_AC": "Ber-AC / Sejuk",
    "buka_lama": "Buka sampai malam",
}

# Wilayah Bandung
WILAYAH_BANDUNG = [
    "Bandung Utara",
    "Bandung Selatan",
    "Bandung Timur",
    "Bandung Barat",
    "Bandung Tengah",
]
