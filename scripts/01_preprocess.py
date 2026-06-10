"""
Step 1: Preprocessing teks ulasan.
Menghasilkan: outputs/cleaned_reviews.csv
"""
import sys
from pathlib import Path

# Tambahkan root project ke sys.path supaya bisa import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import RAW_DATASET, CLEANED_REVIEWS
from src.preprocessing import clean_dataset


def main(do_stem: bool = False):
    print("=" * 70)
    print("STEP 1: PREPROCESSING")
    print("=" * 70)
    print(f"Loading dataset dari: {RAW_DATASET}")
    df = pd.read_csv(RAW_DATASET)
    print(f"Jumlah baris awal: {len(df):,}")

    df_clean = clean_dataset(df, review_col="review", do_stem=do_stem)

    print(f"Jumlah baris setelah cleaning: {len(df_clean):,}")
    print(f"Jumlah kafe unik: {df_clean['place_name'].nunique()}")

    df_clean.to_csv(CLEANED_REVIEWS, index=False)
    print(f"\n[OK] Disimpan ke: {CLEANED_REVIEWS}")


if __name__ == "__main__":
    main()
