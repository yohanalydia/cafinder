"""
Sentiment Analysis untuk Ulasan Bahasa Indonesia.

Pendekatan: IndoBERT + Kalibrasi Rating (Opsi C)
-------------------------------------------------
Model: mdhugol/indonesia-bert-sentiment-classification

Label model:
    LABEL_0 -> positive
    LABEL_1 -> neutral
    LABEL_2 -> negative

Kalibrasi Rating (Opsi C):
  1. confidence < threshold          -> pakai rating
  2. rating >=4 tapi model=negative  -> override positive
  3. rating <=2 tapi model=positive  -> override negative
  4. selain itu                      -> percaya model
"""

from __future__ import annotations

import re
import warnings
from typing import List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
MAX_LENGTH = 128
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.60

LABEL_MAP = {
    "LABEL_0": "positive",
    "LABEL_1": "neutral",
    "LABEL_2": "negative",
    "positif":  "positive",
    "netral":   "neutral",
    "negatif":  "negative",
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
}

# ---------------------------------------------------------------------------
# Kalibrasi Rating
# ---------------------------------------------------------------------------

def _rating_to_sentiment(rating: float) -> str:
    """Konversi rating bintang (1-5) ke label sentimen."""
    if rating >= 4.0:
        return "positive"
    elif rating <= 2.0:
        return "negative"
    else:
        return "neutral"


def _calibrate(
    model_label: str,
    model_score: float,
    rating: Optional[float],
) -> Tuple[str, float]:
    """
    Gabungkan prediksi model dengan sinyal rating.
    Kembalikan (final_label, final_score).
    Score 1.0 menandai prediksi berasal dari kalibrasi rating.
    """
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        return model_label, model_score

    # Confidence terlalu rendah -> sepenuhnya pakai rating
    if model_score < CONFIDENCE_THRESHOLD:
        return _rating_to_sentiment(rating), 1.0

    # Rating sangat bertentangan dengan model
    if rating >= 4.0 and model_label == "negative":
        return "positive", 1.0
    if rating <= 2.0 and model_label == "positive":
        return "negative", 1.0

    return model_label, model_score


# ---------------------------------------------------------------------------
# Lazy Model Loader
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    """Muat model IndoBERT sekali, cache global."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers belum terpasang.\n"
            "Jalankan: pip install transformers torch sentencepiece"
        ) from exc

    import importlib
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("torch belum terpasang.\nJalankan: pip install torch") from exc

    device = 0 if torch.cuda.is_available() else -1
    print(f"[IndoBERT] Memuat model: {MODEL_NAME}")
    print(f"[IndoBERT] Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    _pipeline = hf_pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )
    print("[IndoBERT] Model berhasil dimuat.")
    return _pipeline


# ---------------------------------------------------------------------------
# Preprocessing ringan (sebelum kirim ke BERT)
# ---------------------------------------------------------------------------

def _light_clean(text: str) -> str:
    """
    Bersihkan noise teknis minimal.
    Emoji sengaja TIDAK dihapus agar BERT dapat membaca sinyal sentimen.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Klasifikasi tunggal
# ---------------------------------------------------------------------------

def classify_sentiment(
    text: str,
    rating: Optional[float] = None,
    rating_weight: float = 1.5,
) -> Tuple[str, float]:
    """
    Klasifikasi satu teks dengan IndoBERT + kalibrasi rating.

    Args:
        text   : teks ulasan
        rating : nilai bintang 1.0-5.0 (opsional, untuk kalibrasi)

    Returns:
        (label, score)  label in {positive, neutral, negative}
    """
    pipe = _get_pipeline()
    cleaned = _light_clean(text)

    if not cleaned:
        if rating is not None and not pd.isna(rating):
            return _rating_to_sentiment(rating), 1.0
        return "neutral", 0.0

    result = pipe(cleaned)[0]
    raw_label = result.get("label", "LABEL_1")
    score = float(result.get("score", 0.0))
    model_label = LABEL_MAP.get(raw_label, "neutral")

    return _calibrate(model_label, round(score, 4), rating)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def _batch_predict(
    texts: List[str],
    ratings: Optional[List[Optional[float]]] = None,
) -> List[Tuple[str, float]]:
    """Inferensi batch + kalibrasi rating per elemen."""
    pipe = _get_pipeline()
    safe_texts = [t if t.strip() else "biasa" for t in texts]
    results = pipe(safe_texts)

    output = []
    for i, res in enumerate(results):
        raw_label = res.get("label", "LABEL_1")
        score = float(res.get("score", 0.0))
        model_label = LABEL_MAP.get(raw_label, "neutral")

        if not texts[i].strip():
            r = ratings[i] if ratings else None
            if r is not None and not pd.isna(r):
                output.append((_rating_to_sentiment(r), 1.0))
            else:
                output.append(("neutral", 0.0))
            continue

        r = ratings[i] if ratings else None
        final_label, final_score = _calibrate(model_label, round(score, 4), r)
        output.append((final_label, final_score))

    return output


# ---------------------------------------------------------------------------
# Analisis DataFrame (fungsi utama)
# ---------------------------------------------------------------------------

def analyze_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_clean",
    rating_col: str = "rating_num",
) -> pd.DataFrame:
    """
    Tambahkan kolom 'sentiment' dan 'sentiment_score' ke DataFrame
    menggunakan IndoBERT + kalibrasi rating (Opsi C).

    Args:
        df         : DataFrame ulasan
        text_col   : kolom teks ulasan
        rating_col : kolom rating numerik untuk kalibrasi

    Returns:
        DataFrame dengan kolom tambahan sentiment & sentiment_score.
    """
    df = df.copy()
    total = len(df)

    print(f"[IndoBERT] Memulai untuk {total:,} ulasan...")
    print(f"[IndoBERT] Model       : {MODEL_NAME}")
    print(f"[IndoBERT] Kolom teks  : '{text_col}'")
    print(f"[IndoBERT] Kolom rating: '{rating_col}'")
    print(f"[IndoBERT] Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"[IndoBERT] Batch size: {BATCH_SIZE} | Max token: {MAX_LENGTH}")

    raw_texts: List[str] = df[text_col].fillna("").astype(str).tolist()
    cleaned_texts = [_light_clean(t) for t in raw_texts]

    if rating_col in df.columns:
        ratings: List[Optional[float]] = df[rating_col].tolist()
        print("[IndoBERT] Rating tersedia: kalibrasi aktif.")
    else:
        ratings = [None] * total
        print(f"[IndoBERT] Kolom '{rating_col}' tidak ditemukan; kalibrasi dinonaktifkan.")

    labels: List[str] = []
    scores: List[float] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    batch_iter = range(0, total, BATCH_SIZE)
    if use_tqdm:
        batch_iter = tqdm(batch_iter, desc="[IndoBERT] Inferensi", unit="batch")

    for start in batch_iter:
        end = min(start + BATCH_SIZE, total)
        batch_results = _batch_predict(cleaned_texts[start:end], ratings[start:end])
        for lbl, sc in batch_results:
            labels.append(lbl)
            scores.append(sc)

    df["sentiment"] = labels
    df["sentiment_score"] = scores

    dist = df["sentiment"].value_counts()
    n_cal = scores.count(1.0)
    pct_cal = n_cal / total * 100

    print("\n[IndoBERT] Distribusi sentimen final (model + kalibrasi rating):")
    for lbl, cnt in dist.items():
        pct = cnt / total * 100
        bar = "X" * int(pct / 2)
        print(f"  {lbl:10s}: {cnt:5,} ({pct:5.1f}%) {bar}")
    print(f"\n  Dikalibrasi oleh rating: {n_cal:,} ({pct_cal:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("tempat cozy nyaman kopi enak pelayan ramah recommended banget", 5.0),
        ("mahal banget pelayanan sangat lama berisik sekali sangat kecewa", 1.0),
        ("lumayan biasa standar harga sesuai tidak istimewa", 3.0),
        ("wifi cepat banyak stopkontak nyaman buat belajar kerja", 4.0),
        ("makanannya tidak enak dan tempatnya kotor tidak akan balik lagi", 1.0),
        ("suasana tenang kopi enak harga terjangkau cocok untuk ngerjain tugas", 5.0),
    ]

    print("=" * 65)
    print("IndoBERT + Kalibrasi Rating -- Quick Test (Opsi C)")
    print(f"Model             : {MODEL_NAME}")
    print(f"Conf. threshold   : {CONFIDENCE_THRESHOLD}")
    print("=" * 65)

    for text, rating in samples:
        label, score = classify_sentiment(text, rating=rating)
        mark = {"positive": "[+]", "neutral": "[-]", "negative": "[!]"}.get(label, "[?]")
        note = " <kalibrasi rating>" if score == 1.0 else ""
        print(f"{mark} {label:8s} conf={score:.4f} rating={rating}{note}")
        print(f"    {text!r}")
