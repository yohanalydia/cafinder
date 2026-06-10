"""
Step 2: Sentiment Analysis menggunakan IndoBERT + Kalibrasi Rating.
Menghasilkan: outputs/reviews_with_sentiment.csv

Catatan:
- IndoBERT bekerja optimal dengan teks ASLI (atau lightly cleaned),
  bukan teks yang sudah di-stopword/stem.
- Script ini mengirim kolom 'review' (asli) ke IndoBERT.
  Jika tidak tersedia, fallback ke 'review_clean'.
- Kolom 'review_clean' dan 'rating_num' dari step 1 tetap dipertahankan
  di output agar step selanjutnya (feature extraction) tidak terpengaruh.
- Setelah analisis, ditampilkan metrik evaluasi menggunakan rating sebagai
  ground truth proxy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import CLEANED_REVIEWS, SENTIMENT_REVIEWS
from src.sentiment_analysis import analyze_dataframe, _rating_to_sentiment


# ---------------------------------------------------------------------------
# Fungsi evaluasi
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, labels=None):
    """
    Hitung precision, recall, F1 per kelas + accuracy keseluruhan.
    Tidak butuh sklearn -- murni dictionary/loop biasa.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    stats = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}

    for true, pred in zip(y_true, y_pred):
        if true in stats:
            stats[true]["support"] += 1
        if pred == true:
            if pred in stats:
                stats[pred]["tp"] += 1
        else:
            if pred in stats:
                stats[pred]["fp"] += 1
            if true in stats:
                stats[true]["fn"] += 1

    results = {}
    for lbl in labels:
        tp = stats[lbl]["tp"]
        fp = stats[lbl]["fp"]
        fn = stats[lbl]["fn"]
        support = stats[lbl]["support"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results[lbl] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    total_support = sum(r["support"] for r in results.values())
    weighted_f1 = sum(
        r["f1"] * r["support"] for r in results.values()
    ) / total_support if total_support > 0 else 0.0

    return results, round(accuracy, 4), round(weighted_f1, 4)


def _print_confusion_matrix(y_true, y_pred, labels):
    """Cetak confusion matrix sederhana ke stdout."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in labels:
            matrix[t][p] += 1

    col_w = 11
    header = f"{'':12s}" + "".join(f"{lbl:>{col_w}}" for lbl in labels) + "  <- PREDIKSI"
    print(header)
    print("-" * (12 + col_w * len(labels) + 12))
    for true_lbl in labels:
        row = f"  {true_lbl:10s}" + "".join(
            f"{matrix[true_lbl][pred_lbl]:>{col_w},}" for pred_lbl in labels
        )
        print(row)
    print("^ AKTUAL")


def evaluate_sentiment(df: pd.DataFrame, sentiment_col: str = "sentiment",
                       rating_col: str = "rating_num") -> None:
    """
    Evaluasi kualitas sentimen menggunakan rating sebagai ground truth proxy.

    Logika ground truth:
        rating >= 4  -> "positive"
        rating == 3  -> "neutral"
        rating <= 2  -> "negative"

    Metrik yang dicetak:
        - Confusion matrix
        - Per-kelas: Precision, Recall, F1, Support
        - Accuracy keseluruhan
        - Weighted F1
    """
    if rating_col not in df.columns:
        print(f"[EVAL] Kolom '{rating_col}' tidak ditemukan, evaluasi dilewati.")
        return
    if sentiment_col not in df.columns:
        print(f"[EVAL] Kolom '{sentiment_col}' tidak ditemukan, evaluasi dilewati.")
        return

    df_eval = df.dropna(subset=[rating_col, sentiment_col]).copy()
    df_eval["gt"] = df_eval[rating_col].apply(_rating_to_sentiment)

    y_true = df_eval["gt"].tolist()
    y_pred = df_eval[sentiment_col].tolist()
    labels = ["positive", "neutral", "negative"]

    per_class, accuracy, weighted_f1 = _compute_metrics(y_true, y_pred, labels)

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    print()
    print("=" * 70)
    print("  EVALUASI SENTIMEN  (ground truth = rating bintang)")
    print("=" * 70)
    print(f"  Jumlah ulasan dievaluasi : {total:,}")
    print(f"  Prediksi benar           : {correct:,}")
    print(f"  Accuracy                 : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Weighted F1              : {weighted_f1:.4f}")
    print()

    print("  Confusion Matrix:")
    _print_confusion_matrix(y_true, y_pred, labels)
    print()

    print(f"  {'Kelas':12s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 52)
    for lbl in labels:
        m = per_class[lbl]
        print(f"  {lbl:12s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10,}")
    print()

    # Interpretasi singkat
    print("  Interpretasi:")
    for lbl in labels:
        m = per_class[lbl]
        f1 = m["f1"]
        if f1 >= 0.80:
            grade = "Sangat Baik"
        elif f1 >= 0.65:
            grade = "Baik"
        elif f1 >= 0.50:
            grade = "Cukup"
        else:
            grade = "Perlu Diperbaiki"
        print(f"    - {lbl:10s}: F1={f1:.4f}  -> {grade}")

    print()
    if accuracy >= 0.80:
        print("  [HASIL] Model berjalan BAIK -- sentimen selaras dengan rating.")
    elif accuracy >= 0.65:
        print("  [HASIL] Model cukup baik, ada ruang perbaikan.")
    else:
        print("  [HASIL] Akurasi rendah -- periksa kembali model atau threshold.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 2: SENTIMENT ANALYSIS - IndoBERT + Kalibrasi Rating")
    print("=" * 70)

    print(f"Loading cleaned reviews dari: {CLEANED_REVIEWS}")
    df = pd.read_csv(CLEANED_REVIEWS)
    print(f"Jumlah baris: {len(df):,}")
    print("Kolom tersedia:", list(df.columns))

    # ------------------------------------------------------------------
    # Pilih kolom teks: prioritas 'review' (teks asli), fallback 'review_clean'
    # ------------------------------------------------------------------
    if "review" in df.columns:
        text_col = "review"
        print(f"\n[INFO] Menggunakan kolom '{text_col}' (teks asli) untuk IndoBERT.")
    else:
        text_col = "review_clean"
        print(f"\n[INFO] Kolom 'review' tidak ditemukan, fallback ke '{text_col}'.")

    # ------------------------------------------------------------------
    # Jalankan IndoBERT + kalibrasi rating
    # ------------------------------------------------------------------
    df = analyze_dataframe(df, text_col=text_col, rating_col="rating_num")

    # ------------------------------------------------------------------
    # Evaluasi metrik
    # ------------------------------------------------------------------
    evaluate_sentiment(df, sentiment_col="sentiment", rating_col="rating_num")

    # ------------------------------------------------------------------
    # Simpan output
    # ------------------------------------------------------------------
    df.to_csv(SENTIMENT_REVIEWS, index=False)
    print(f"[OK] Disimpan ke: {SENTIMENT_REVIEWS}")
    print("Kolom output:", list(df.columns))


if __name__ == "__main__":
    main()
