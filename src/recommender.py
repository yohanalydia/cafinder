"""
Content-Based Recommendation System.

Pendekatan:
- Profil kafe = vektor skor fitur (TARGET_FEATURES) + bobot rating & sentiment.
- Preferensi pengguna = vektor binary/skor preferensi user atas tiap fitur.
- Skor rekomendasi = cosine similarity profil-vs-preferensi, ditambah faktor penyesuai
  (boost berdasarkan rating rata-rata dan persentase review positif).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import TARGET_FEATURES


@dataclass
class UserPreference:
    """Preferensi pengguna untuk pencarian kafe."""
    features: Dict[str, float]              # {feature: importance 0-1}
    lokasi: Optional[List[str]] = None      # filter wilayah Bandung (None = semua)
    min_rating: float = 0.0                 # minimum avg rating
    min_reviews: int = 0                    # minimum jumlah review (untuk filter kafe yang reliable)


def _vectorize(values: Dict[str, float]) -> np.ndarray:
    """Susun dict feature -> ndarray sesuai urutan TARGET_FEATURES."""
    return np.array([float(values.get(f, 0.0)) for f in TARGET_FEATURES])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hitung cosine similarity 2 vektor. Aman terhadap zero-vector."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def recommend(
    df_features: pd.DataFrame,
    user_pref: UserPreference,
    top_n: int = 10,
    rating_weight: float = 0.15,
    sentiment_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Berikan rekomendasi top_n kafe untuk preferensi pengguna.

    Skor akhir = (1 - rating_weight - sentiment_weight) * cosine_sim
                 + rating_weight * normalized_rating
                 + sentiment_weight * positive_pct

    Args:
        df_features: dataframe hasil aggregate_cafe_features
        user_pref: preferensi pengguna
        top_n: jumlah rekomendasi
        rating_weight: bobot rating (0-1)
        sentiment_weight: bobot positive_pct (0-1)

    Returns:
        DataFrame top_n kafe terurut berdasarkan skor, dengan kolom 'match_score'.
    """
    df = df_features.copy()

    # Filter berdasarkan lokasi
    if user_pref.lokasi:
        df = df[df["lokasi"].isin(user_pref.lokasi)]

    # Filter rating & jumlah review
    if user_pref.min_rating > 0:
        df = df[df["avg_rating"].fillna(0) >= user_pref.min_rating]
    if user_pref.min_reviews > 0:
        df = df[df["n_reviews"].fillna(0) >= user_pref.min_reviews]

    if len(df) == 0:
        return df.assign(match_score=[]).head(0)

    # Vektor preferensi user
    user_vec = _vectorize(user_pref.features)

    # Hitung skor per kafe
    scores: List[float] = []
    cosine_sims: List[float] = []
    for _, row in df.iterrows():
        cafe_vec = np.array([float(row.get(f, 0.0)) for f in TARGET_FEATURES])
        sim = _cosine_similarity(user_vec, cafe_vec)
        cosine_sims.append(sim)

        # Normalize rating ke [0,1]
        rating_norm = ((row.get("avg_rating") or 3.0) - 1) / 4.0
        rating_norm = max(0.0, min(1.0, rating_norm))

        # positive_pct ada di [0,1]
        pos_pct = row.get("positive_pct") or 0.5

        base_weight = 1 - rating_weight - sentiment_weight
        final = (
            base_weight * sim
            + rating_weight * rating_norm
            + sentiment_weight * pos_pct
        )
        scores.append(final)

    df = df.assign(
        cosine_sim=np.round(cosine_sims, 4),
        match_score=np.round(scores, 4),
    )

    df = df.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def explain_match(
    cafe_row: pd.Series,
    user_pref: UserPreference,
    top_features: int = 4,
) -> List[str]:
    """
    Hasilkan list alasan kenapa kafe ini direkomendasikan.
    """
    reasons = []

    # Rating
    avg_rating = cafe_row.get("avg_rating")
    if avg_rating and avg_rating >= 4.5:
        reasons.append(f"Rating sangat tinggi ({avg_rating:.1f}/5)")
    elif avg_rating and avg_rating >= 4.0:
        reasons.append(f"Rating bagus ({avg_rating:.1f}/5)")

    # Positive sentiment
    pos_pct = cafe_row.get("positive_pct")
    if pos_pct is not None and pos_pct >= 0.7:
        reasons.append(f"{int(pos_pct*100)}% review pengguna positif")

    # Top fitur yang dimiliki kafe & dibutuhkan user
    user_wanted = {f: w for f, w in user_pref.features.items() if w > 0.3}
    if user_wanted:
        cafe_scores = [(f, cafe_row.get(f, 0.0)) for f in user_wanted.keys()]
        cafe_scores.sort(key=lambda x: x[1], reverse=True)
        good_feats = [f for f, s in cafe_scores if s >= 0.4][:top_features]
        if good_feats:
            from src.config import FEATURE_LABELS
            label_str = ", ".join(FEATURE_LABELS.get(f, f) for f in good_feats)
            reasons.append(f"Cocok untuk: {label_str}")

    return reasons


if __name__ == "__main__":
    # Quick test
    rows = [
        {"place_name": "Cafe A", "lokasi": "Bandung Utara", "avg_rating": 4.6,
         "positive_pct": 0.85, "negative_pct": 0.05, "n_reviews": 120,
         "wifi": 0.9, "stopkontak": 0.8, "kenyamanan": 0.9, "suasana_tenang": 0.7,
         "harga_terjangkau": 0.6, "makanan_enak": 0.8, "pelayanan_baik": 0.9,
         "tempat_luas": 0.5, "ber_AC": 0.8, "buka_lama": 0.3},
        {"place_name": "Cafe B", "lokasi": "Bandung Tengah", "avg_rating": 4.2,
         "positive_pct": 0.7, "negative_pct": 0.15, "n_reviews": 80,
         "wifi": 0.4, "stopkontak": 0.3, "kenyamanan": 0.6, "suasana_tenang": 0.8,
         "harga_terjangkau": 0.9, "makanan_enak": 0.9, "pelayanan_baik": 0.7,
         "tempat_luas": 0.7, "ber_AC": 0.5, "buka_lama": 0.6},
    ]
    df = pd.DataFrame(rows)
    pref = UserPreference(
        features={"wifi": 1.0, "stopkontak": 1.0, "kenyamanan": 1.0, "suasana_tenang": 0.7},
        min_rating=4.0,
    )
    result = recommend(df, pref, top_n=5)
    print(result[["place_name", "lokasi", "avg_rating", "match_score"]])
