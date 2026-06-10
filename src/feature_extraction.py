"""
Feature Extraction - Mengekstrak fitur semantik dari ulasan kafe.

Dua mode:
1. Keyword-based (default, gratis, tanpa API)
2. LLM-based dengan Gemini API (lebih akurat, butuh API key)
3. Hybrid: keyword + LLM untuk verifikasi

Output: tiap kafe punya skor 0-1 untuk tiap fitur target.
"""

from __future__ import annotations
import json
import time
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.config import TARGET_FEATURES, GEMINI_API_KEY, GEMINI_MODEL


# =============================================================================
# Keyword dictionary untuk tiap fitur
# =============================================================================
FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "wifi": [
        "wifi", "wi-fi", "wi fi", "internet", "koneksi", "jaringan",
    ],
    "stopkontak": [
        "stopkontak", "stop kontak", "colokan", "colok", "charger", "charging",
        "powerbank", "outlet listrik",
    ],
    "kenyamanan": [
        "nyaman", "cozy", "homy", "homey", "santai", "rileks", "betah",
        "enak buat", "bagus buat", "cocok buat", "tempat belajar", "tempat kerja",
        "wfc", "wfh", "ngerjain tugas", "ngerjain", "nugas", "kerja", "belajar",
        "skripsi", "meeting", "diskusi",
    ],
    "suasana_tenang": [
        "tenang", "sepi", "kalem", "damai", "asri", "syahdu", "private",
        "tidak berisik", "ga berisik", "gak berisik",
    ],
    "harga_terjangkau": [
        "murah", "terjangkau", "worth", "worthit", "worth it", "affordable",
        "ekonomis", "sesuai harga", "harga sesuai", "ramah kantong", "ramah dompet",
        "student friendly", "harga mahasiswa", "diskon", "promo",
    ],
    "makanan_enak": [
        "enak", "lezat", "nikmat", "gurih", "fresh", "fresh banget",
        "kopinya enak", "makanan enak", "minuman enak", "rasa", "rasanya",
        "menu lengkap", "varian banyak", "kue enak", "dessert enak",
    ],
    "pelayanan_baik": [
        "ramah", "sopan", "cepat", "responsif", "helpful", "friendly",
        "pelayanan bagus", "pelayanan baik", "service bagus", "service baik",
        "barista ramah", "kasir ramah", "staff ramah", "waiter ramah",
    ],
    "tempat_luas": [
        "luas", "lega", "spacious", "roomy", "besar", "lapang",
        "muat banyak", "kapasitas besar",
    ],
    "ber_AC": [
        "ac ", " ac", "berac", "ber ac", "ber-ac", "dingin", "sejuk", "adem",
        "tidak panas", "ga panas",
    ],
    "buka_lama": [
        "24 jam", "24jam", "buka malam", "sampai malam", "tutup larut",
        "buka pagi", "buka subuh", "open late", "late night",
    ],
}

# Negasi keyword untuk membalik polaritas
NEGATION_PATTERNS = [
    "tidak ada", "tidak punya", "tidak tersedia", "tdk ada",
    "ga ada", "gak ada", "ngga ada", "nggak ada", "engga ada",
    "tanpa", "kekurangan",
]


def keyword_score(review_text: str, feature: str) -> float:
    """
    Hitung skor 0-1 dari keyword matching, dengan handling negasi sederhana.

    Returns:
        skor: 0 = tidak ada bukti, 1 = banyak bukti positif
    """
    if not isinstance(review_text, str) or not review_text:
        return 0.0

    text = " " + review_text.lower() + " "
    keywords = FEATURE_KEYWORDS.get(feature, [])

    pos_hits = 0
    neg_hits = 0
    for kw in keywords:
        # Cari semua occurrence
        idx = 0
        while True:
            pos = text.find(kw, idx)
            if pos == -1:
                break
            # Cek negasi 20 karakter sebelum kata
            window_start = max(0, pos - 25)
            window = text[window_start:pos]
            is_negated = any(neg in window for neg in NEGATION_PATTERNS)

            if is_negated:
                neg_hits += 1
            else:
                pos_hits += 1
            idx = pos + len(kw)

    if pos_hits == 0 and neg_hits == 0:
        return 0.0
    # Normalize ke [0, 1]; saturasi di 5 hits
    raw = (pos_hits - neg_hits) / 5.0
    return max(0.0, min(1.0, raw))


def aggregate_cafe_features(
    df_reviews: pd.DataFrame,
    text_col: str = "review_clean",
    place_col: str = "place_name",
    sentiment_col: str = "sentiment",
    only_positive: bool = True,
) -> pd.DataFrame:
    """
    Agregasi fitur per kafe dari seluruh ulasannya.

    Args:
        df_reviews: dataframe per-review yang sudah punya kolom review_clean & sentiment
        only_positive: jika True, hanya gunakan review positif untuk fitur (mengurangi noise dari komplain)

    Returns:
        DataFrame: 1 baris per kafe + kolom skor tiap fitur (0-1)
    """
    df = df_reviews.copy()

    if only_positive and sentiment_col in df.columns:
        # Untuk fitur fasilitas, kita pakai review positif saja
        df_pos = df[df[sentiment_col] == "positive"]
    else:
        df_pos = df

    print(f"[Feature Extraction] Memproses {df[place_col].nunique()} kafe...")
    rows = []
    for cafe, group in tqdm(df.groupby(place_col), desc="Ekstraksi fitur per kafe"):
        # Gabungkan semua review (positive jika ada, fallback ke semua)
        pos_group = group[group[sentiment_col] == "positive"] if sentiment_col in group.columns else group
        if len(pos_group) == 0:
            pos_group = group

        all_text = " ".join(pos_group[text_col].fillna("").astype(str).tolist())

        feature_scores: Dict[str, float] = {}
        for feat in TARGET_FEATURES:
            feature_scores[feat] = keyword_score(all_text, feat)

        # Statistik tambahan
        pos_pct = (group[sentiment_col] == "positive").mean() if sentiment_col in group.columns else None
        neg_pct = (group[sentiment_col] == "negative").mean() if sentiment_col in group.columns else None
        avg_rating = group["rating_num"].mean() if "rating_num" in group.columns else None
        n_reviews = len(group)
        lokasi = group["lokasi"].mode().iloc[0] if "lokasi" in group.columns and not group["lokasi"].mode().empty else None

        row = {
            "place_name": cafe,
            "lokasi": lokasi,
            "n_reviews": n_reviews,
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "positive_pct": round(pos_pct, 3) if pos_pct is not None else None,
            "negative_pct": round(neg_pct, 3) if neg_pct is not None else None,
        }
        row.update(feature_scores)
        rows.append(row)

    df_features = pd.DataFrame(rows)
    return df_features


# =============================================================================
# LLM-based extraction dengan Gemini
# =============================================================================
def extract_with_gemini(
    cafe_name: str,
    sample_reviews: List[str],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_reviews: int = 15,
    max_chars: int = 4000,
) -> Optional[Dict[str, float]]:
    """
    Pakai Gemini API untuk ekstraksi fitur dari sample review sebuah kafe.

    Returns:
        Dict {feature: score 0-1} atau None jika gagal.
    """
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        print("[Gemini] google-generativeai belum terinstall. Skip LLM extraction.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name or GEMINI_MODEL)

    # Susun prompt
    sample = sample_reviews[:max_reviews]
    joined = "\n- " + "\n- ".join(sample)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "..."

    feature_list = ", ".join(TARGET_FEATURES)
    prompt = f"""Anda adalah analis ulasan kafe. Berdasarkan ulasan-ulasan tentang kafe "{cafe_name}" berikut, beri skor 0.0 - 1.0 untuk setiap fitur:

{joined}

Fitur yang harus dinilai: {feature_list}

Definisi skor:
- 0.0 = tidak disebut sama sekali atau pengalaman buruk
- 0.5 = ambigu / kadang ada kadang tidak / netral
- 1.0 = banyak ulasan positif menyebutkan fitur ini

Jawab HANYA dengan JSON valid (tanpa markdown, tanpa penjelasan), contoh format:
{{"wifi": 0.8, "stopkontak": 0.6, ...}}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences kalau ada
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        # Validasi & normalisasi
        result = {}
        for feat in TARGET_FEATURES:
            val = data.get(feat, 0.0)
            try:
                result[feat] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                result[feat] = 0.0
        return result
    except Exception as exc:
        print(f"[Gemini] Error untuk {cafe_name}: {exc}")
        return None


def enhance_with_gemini(
    df_features: pd.DataFrame,
    df_reviews: pd.DataFrame,
    sample_size: int = 15,
    sleep_seconds: float = 1.0,
    max_cafes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Upgrade skor fitur menggunakan Gemini LLM.
    Skor final = rata-rata (keyword_score + gemini_score) / 2.

    Args:
        df_features: hasil aggregate_cafe_features (keyword-based)
        df_reviews: dataframe review per baris (kolom: place_name, review)
        max_cafes: kalau diisi, hanya proses N kafe pertama (untuk testing)
    """
    if not GEMINI_API_KEY:
        print("[Gemini] GEMINI_API_KEY tidak ditemukan. Mengembalikan df_features apa adanya.")
        return df_features

    df_features = df_features.copy()
    cafes = df_features["place_name"].tolist()
    if max_cafes:
        cafes = cafes[:max_cafes]

    for cafe in tqdm(cafes, desc="Gemini extraction"):
        cafe_reviews = df_reviews[df_reviews["place_name"] == cafe]
        # Prefer review positive yang panjang
        if "sentiment" in cafe_reviews.columns:
            cafe_reviews = cafe_reviews[cafe_reviews["sentiment"] == "positive"]
        if len(cafe_reviews) == 0:
            continue

        sample_reviews = (
            cafe_reviews.sort_values("review", key=lambda s: s.str.len(), ascending=False)
            ["review"].head(sample_size).tolist()
        )

        gemini_scores = extract_with_gemini(cafe, sample_reviews)
        if gemini_scores is None:
            continue

        # Blend dengan keyword score
        idx = df_features.index[df_features["place_name"] == cafe][0]
        for feat in TARGET_FEATURES:
            kw = float(df_features.at[idx, feat])
            llm = float(gemini_scores.get(feat, 0.0))
            df_features.at[idx, feat] = round((kw + llm) / 2, 3)

        time.sleep(sleep_seconds)  # rate limit

    return df_features


if __name__ == "__main__":
    text = "tempat nyaman wifi kenceng banyak stopkontak harga terjangkau pelayanan ramah"
    for f in TARGET_FEATURES:
        print(f"{f:20s} -> {keyword_score(text, f):.2f}")
