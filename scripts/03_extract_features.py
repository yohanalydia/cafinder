"""
Step 3: Feature extraction (keyword-based + opsional Gemini LLM).
Menghasilkan: outputs/cafe_features.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (
    SENTIMENT_REVIEWS, CAFE_FEATURES, FEATURE_EXTRACTION_MODE,
    GEMINI_API_KEY, TARGET_FEATURES
)
from src.feature_extraction import aggregate_cafe_features, enhance_with_gemini


def main():
    print("=" * 70)
    print("STEP 3: FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Mode: {FEATURE_EXTRACTION_MODE}")
    print(f"Loading reviews dari: {SENTIMENT_REVIEWS}")

    df = pd.read_csv(SENTIMENT_REVIEWS)
    print(f"Jumlah review: {len(df):,}")

    print("\n>> Tahap 1: Keyword-based extraction")
    df_features = aggregate_cafe_features(
        df,
        text_col="review_clean",
        place_col="place_name",
        sentiment_col="sentiment",
        only_positive=True,
    )
    print(f"Jumlah kafe diproses: {len(df_features)}")

    if FEATURE_EXTRACTION_MODE in {"llm", "hybrid"} and GEMINI_API_KEY:
        print("\n>> Tahap 2: Gemini LLM enhancement")
        df_features = enhance_with_gemini(df_features, df, sample_size=15, sleep_seconds=0.5)
    elif FEATURE_EXTRACTION_MODE in {"llm", "hybrid"}:
        print("\n[WARN] FEATURE_EXTRACTION_MODE diset 'llm/hybrid' tapi GEMINI_API_KEY kosong.")
        print("       Skip tahap LLM. Tambahkan API key di .env untuk hasil lebih akurat.")

    # Statistik singkat
    print("\nStatistik skor fitur (rata-rata across cafes):")
    for f in TARGET_FEATURES:
        avg = df_features[f].mean()
        print(f"  {f:20s} : {avg:.3f}")

    df_features.to_csv(CAFE_FEATURES, index=False)
    print(f"\n[OK] Disimpan ke: {CAFE_FEATURES}")


if __name__ == "__main__":
    main()
