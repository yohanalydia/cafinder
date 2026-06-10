"""
Step 4: Build SQLite database.
Menghasilkan: outputs/cafe_recommender.db
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import SENTIMENT_REVIEWS, CAFE_FEATURES, DATABASE_FILE
from src.database import populate_database, get_database_stats


def main():
    print("=" * 70)
    print("STEP 4: BUILD DATABASE")
    print("=" * 70)
    df_reviews = pd.read_csv(SENTIMENT_REVIEWS)
    df_features = pd.read_csv(CAFE_FEATURES)
    print(f"Reviews: {len(df_reviews):,} | Cafes: {len(df_features)}")

    populate_database(df_features, df_reviews, db_path=DATABASE_FILE)

    print("\nDatabase stats:")
    stats = get_database_stats(DATABASE_FILE)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
