"""
Modul Preprocessing Teks untuk Ulasan Kafe.

Tahapan:
1. Cleaning: hapus emoji, URL, karakter non-alfanumerik
2. Lowercasing
3. Normalisasi slang Bahasa Indonesia (gw -> saya, bgt -> banget, dll)
4. Hapus stopwords (Sastrawi)
5. Stemming (Sastrawi)
6. Deduplikasi & filter ulasan kosong
"""

from __future__ import annotations
import re
import string
from typing import Optional

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =============================================================================
# Sastrawi singleton (dibuat sekali untuk performa)
# =============================================================================
_stopword_factory = StopWordRemoverFactory()
_default_stopwords = set(_stopword_factory.get_stop_words())

# Tambahan stopwords yang muncul di review kafe tapi tidak informatif
_extra_stopwords = {
    "kafe", "cafe", "coffee", "kopi", "tempat", "tempatnya", "tempatx",
    "banget", "bgt", "bnget", "bener", "bener2", "sangat", "sgt", "sekali",
    "kali", "yg", "yang", "udah", "udh", "sdh", "dah", "uda", "ud",
    "ya", "yah", "yaaa", "deh", "sih", "nih", "ah", "ih", "loh", "kok",
    "aja", "ajg", "doang", "dong", "kan", "saja", "juga", "jg", "jga",
    "bisa", "buat", "untuk", "utk", "udah", "kalau", "kalo", "klo",
    "harga", "menu", "rating", "review", "google", "maps",
}
ALL_STOPWORDS = _default_stopwords.union(_extra_stopwords)

# Stemmer Sastrawi (lazy karena lambat saat init)
_stemmer = StemmerFactory().create_stemmer()

# =============================================================================
# Slang dictionary Bahasa Indonesia (informal -> formal)
# =============================================================================
SLANG_DICT = {
    "gw": "saya", "gue": "saya", "gua": "saya", "aku": "saya",
    "lu": "kamu", "elu": "kamu", "lo": "kamu",
    "bgt": "banget", "bngt": "banget", "bener": "benar",
    "krn": "karena", "krna": "karena", "karna": "karena",
    "tdk": "tidak", "tak": "tidak", "ga": "tidak", "gak": "tidak",
    "gk": "tidak", "ngga": "tidak", "nggak": "tidak", "ngak": "tidak",
    "engga": "tidak", "kagak": "tidak",
    "aj": "saja", "aja": "saja",
    "udh": "sudah", "udah": "sudah", "uda": "sudah",
    "blm": "belum", "belom": "belum",
    "dgn": "dengan", "dg": "dengan",
    "klo": "kalau", "kalo": "kalau",
    "yg": "yang", "yng": "yang",
    "dr": "dari", "drpd": "daripada",
    "tp": "tapi", "tpi": "tapi", "tetapi": "tapi",
    "jg": "juga", "jga": "juga",
    "bs": "bisa", "bsa": "bisa",
    "sm": "sama", "sma": "sama",
    "msh": "masih", "msih": "masih",
    "bnyk": "banyak", "byk": "banyak",
    "skrg": "sekarang", "skrng": "sekarang",
    "tmn": "teman", "tmen": "teman",
    "ttg": "tentang", "tntg": "tentang",
    "trs": "terus", "trus": "terus",
    "stl": "setelah", "stlh": "setelah",
    "nyaman": "nyaman", "nyamn": "nyaman", "nyamaaan": "nyaman",
    "enak": "enak", "enk": "enak", "enaaak": "enak", "enakk": "enak",
    "mantap": "mantap", "mantul": "mantap", "mantab": "mantap",
    "rame": "ramai", "rame2": "ramai",
    "wifi": "wifi", "wi-fi": "wifi", "wi fi": "wifi",
    "stopkontak": "stopkontak", "stop kontak": "stopkontak",
    "colokan": "stopkontak",
    "ac": "ac", "a/c": "ac",
}


# =============================================================================
# Fungsi Cleaning
# =============================================================================
def remove_emoji(text: str) -> str:
    """Hapus emoji dengan regex unicode."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002500-\U00002BEF"  # chinese chars
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "♀-♂"
        "☀-⭕"
        "‍"
        "⏏"
        "⏩"
        "⌚"
        "️"
        "〰"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_urls(text: str) -> str:
    """Hapus URL."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_mentions_hashtags(text: str) -> str:
    """Hapus mention (@user) dan hashtag (#tag)."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text


def remove_special_chars(text: str) -> str:
    """Hapus karakter spesial, sisakan huruf, angka, dan whitespace."""
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)  # juga hapus angka
    return text


def normalize_whitespace(text: str) -> str:
    """Gabungkan whitespace berlebih jadi 1 spasi."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_slang(text: str, slang_dict: Optional[dict] = None) -> str:
    """Ubah kata slang ke bentuk formal."""
    if slang_dict is None:
        slang_dict = SLANG_DICT
    tokens = text.split()
    normalized = [slang_dict.get(t, t) for t in tokens]
    return " ".join(normalized)


def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    """Hapus stopwords."""
    if stopwords is None:
        stopwords = ALL_STOPWORDS
    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(filtered)


def stem_text(text: str) -> str:
    """Stemming dengan Sastrawi (mengubah kata jadi kata dasar)."""
    if not text:
        return text
    return _stemmer.stem(text)


# =============================================================================
# Pipeline lengkap
# =============================================================================
def clean_review(text: str, do_stem: bool = False) -> str:
    """
    Bersihkan satu teks ulasan dari raw -> teks bersih siap analisis.

    Args:
        text: teks ulasan mentah
        do_stem: jika True, lakukan stemming. Stemming lambat untuk dataset besar.

    Returns:
        teks bersih (lowercase, tanpa emoji/URL/stopwords/karakter spesial)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_emoji(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    text = normalize_whitespace(text)

    if do_stem and text:
        text = stem_text(text)

    return text


def clean_dataset(
    df: pd.DataFrame,
    review_col: str = "review",
    do_stem: bool = False,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
    min_words: int = 2,
) -> pd.DataFrame:
    """
    Bersihkan seluruh dataset.

    Args:
        df: dataframe dengan kolom review_col
        review_col: nama kolom review
        do_stem: aktifkan stemming
        drop_empty: hapus baris dengan review kosong
        drop_duplicates: hapus baris duplikat
        min_words: minimum jumlah kata setelah cleaning agar baris dipertahankan

    Returns:
        DataFrame baru dengan kolom 'review_clean' dan 'rating_num'
    """
    df = df.copy()

    # Konversi rating "5 bintang" -> 5
    if "rating" in df.columns:
        df["rating_num"] = (
            df["rating"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    # Cleaning teks
    print("[Preprocessing] Membersihkan teks ulasan...")
    df["review_clean"] = df[review_col].astype(str).apply(
        lambda t: clean_review(t, do_stem=do_stem)
    )

    if drop_empty:
        before = len(df)
        df = df[df["review_clean"].str.split().str.len() >= min_words]
        print(f"[Preprocessing] Dihapus {before - len(df)} baris kosong/terlalu pendek")

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["place_name", "review_clean"]).reset_index(drop=True)
        print(f"[Preprocessing] Dihapus {before - len(df)} baris duplikat")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Quick test
    sample = "Tempatnya cozy bgt!! WiFi nyaa kenceng dan ada banyak stopkontak 😍 nyaman banget buat ngerjain tugas. https://maps.google.com"
    print("Original :", sample)
    print("Cleaned  :", clean_review(sample))
