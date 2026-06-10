"""
Pipeline orchestrator: jalankan semua step preprocessing -> sentiment -> features -> db.

Cara pakai:
    python scripts/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

STEPS = [
    ("scripts.01_preprocess", "01 Preprocessing"),
    ("scripts.02_sentiment", "02 Sentiment Analysis"),
    ("scripts.03_extract_features", "03 Feature Extraction"),
    ("scripts.04_build_db", "04 Build Database"),
]


def main():
    print("\n" + "#" * 70)
    print("# CAFE RECOMMENDER - FULL PIPELINE")
    print("#" * 70)

    for module_name, label in STEPS:
        # Import dynamic karena nama modul mulai dengan angka
        # Gunakan importlib dengan path-style spec
        path = Path(__file__).parent / (module_name.split(".")[-1] + ".py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        print(f"\n>>> {label}")
        spec.loader.exec_module(mod)
        mod.main()

    print("\n" + "#" * 70)
    print("# PIPELINE SELESAI")
    print("# Sekarang jalankan: streamlit run app.py")
    print("#" * 70)


if __name__ == "__main__":
    main()
