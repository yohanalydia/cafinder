"""
Modul Database SQLite.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATABASE_FILE, TARGET_FEATURES


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cafes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_name TEXT UNIQUE NOT NULL,
    lokasi TEXT,
    avg_rating REAL,
    n_reviews INTEGER,
    positive_pct REAL,
    negative_pct REAL
);

CREATE INDEX IF NOT EXISTS idx_cafes_lokasi ON cafes(lokasi);
CREATE INDEX IF NOT EXISTS idx_cafes_rating ON cafes(avg_rating);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cafe_id INTEGER NOT NULL,
    username TEXT,
    rating INTEGER,
    review_raw TEXT,
    review_clean TEXT,
    sentiment TEXT,
    sentiment_score REAL,
    FOREIGN KEY (cafe_id) REFERENCES cafes(id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_cafe ON reviews(cafe_id);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment);

CREATE TABLE IF NOT EXISTS features (
    cafe_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (cafe_id, feature_name),
    FOREIGN KEY (cafe_id) REFERENCES cafes(id)
);

CREATE INDEX IF NOT EXISTS idx_features_name ON features(feature_name);
"""


def _safe_remove(path):
    """Coba hapus file. Kalau OS tidak mengizinkan, truncate isinya jadi 0 byte."""
    if not path.exists():
        return
    try:
        path.unlink()
        return
    except (PermissionError, OSError):
        pass
    try:
        with open(path, "w") as f:
            f.truncate(0)
    except OSError:
        pass


def init_database(db_path=DATABASE_FILE):
    """Buat database & tabel jika belum ada."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def populate_database(df_features, df_reviews, db_path=DATABASE_FILE):
    """Isi database dari DataFrame fitur & review (rebuild fresh)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    journal = db_path.with_suffix(db_path.suffix + "-journal")
    _safe_remove(db_path)
    _safe_remove(journal)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS reviews; DROP TABLE IF EXISTS features; DROP TABLE IF EXISTS cafes;")
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    print("[Database] Insert cafes...")
    cafe_id_map = {}
    for _, row in df_features.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO cafes (place_name, lokasi, avg_rating, n_reviews, positive_pct, negative_pct) VALUES (?, ?, ?, ?, ?, ?)",
            (row["place_name"], row.get("lokasi"), row.get("avg_rating"),
             int(row.get("n_reviews", 0) or 0), row.get("positive_pct"), row.get("negative_pct")),
        )
        cur.execute("SELECT id FROM cafes WHERE place_name = ?", (row["place_name"],))
        cafe_id_map[row["place_name"]] = cur.fetchone()[0]

    print("[Database] Insert features...")
    for _, row in df_features.iterrows():
        cafe_id = cafe_id_map[row["place_name"]]
        for feat in TARGET_FEATURES:
            cur.execute(
                "INSERT OR REPLACE INTO features (cafe_id, feature_name, score) VALUES (?, ?, ?)",
                (cafe_id, feat, float(row.get(feat, 0.0))),
            )

    print("[Database] Insert reviews...")
    for _, row in df_reviews.iterrows():
        cafe_id = cafe_id_map.get(row.get("place_name"))
        if cafe_id is None:
            continue
        cur.execute(
            "INSERT INTO reviews (cafe_id, username, rating, review_raw, review_clean, sentiment, sentiment_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cafe_id, row.get("username"),
             int(row["rating_num"]) if pd.notna(row.get("rating_num")) else None,
             row.get("review"), row.get("review_clean"),
             row.get("sentiment"), row.get("sentiment_score")),
        )

    conn.commit()
    conn.close()
    print(f"[Database] Selesai. File: {db_path}")


def load_cafe_features(db_path=DATABASE_FILE):
    """Load skor fitur tiap kafe dari database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT c.id AS cafe_id, c.place_name, c.lokasi, c.avg_rating, c.n_reviews, c.positive_pct, c.negative_pct FROM cafes c",
        conn,
    )
    feats = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    feat_pivot = feats.pivot(index="cafe_id", columns="feature_name", values="score").reset_index()
    df = df.merge(feat_pivot, on="cafe_id", how="left")

    for f in TARGET_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].fillna(0.0)
    return df


def load_reviews_for_cafe(place_name, db_path=DATABASE_FILE):
    """Ambil semua review untuk satu kafe."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT r.username, r.rating, r.review_raw, r.review_clean, r.sentiment, r.sentiment_score FROM reviews r JOIN cafes c ON r.cafe_id = c.id WHERE c.place_name = ? ORDER BY r.rating DESC",
        conn, params=(place_name,),
    )
    conn.close()
    return df


def get_database_stats(db_path=DATABASE_FILE):
    """Hitung statistik dasar database."""
    if not db_path.exists():
        return {"db_exists": False}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    stats = {"db_exists": True}
    cur.execute("SELECT COUNT(*) FROM cafes")
    stats["n_cafes"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM reviews")
    stats["n_reviews"] = cur.fetchone()[0]
    cur.execute("SELECT sentiment, COUNT(*) FROM reviews WHERE sentiment IS NOT NULL GROUP BY sentiment")
    stats["sentiment_dist"] = dict(cur.fetchall())
    cur.execute("SELECT lokasi, COUNT(*) FROM cafes WHERE lokasi IS NOT NULL GROUP BY lokasi")
    stats["lokasi_dist"] = dict(cur.fetchall())
    conn.close()
    return stats
