"""
Query Parser: ubah teks bebas -> UserPreference.

Contoh input pengguna:
    "saya cari kafe yang tenang ada wifi dan stopkontak buat skripsian
     di bandung utara, harga jangan mahal"

Output:
    UserPreference(
        features={"suasana_tenang": 1.0, "wifi": 1.0, "stopkontak": 1.0,
                  "kenyamanan": 0.8, "harga_terjangkau": 1.0},
        lokasi=["Bandung Utara"],
        ...
    )

Dua mode:
1. Keyword-based   : pencocokan kata kunci (selalu jalan, gratis)
2. Gemini-based    : Pakai LLM untuk parsing lebih natural (butuh API key)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import re

from src.config import (
    TARGET_FEATURES, FEATURE_LABELS, WILAYAH_BANDUNG,
    GEMINI_API_KEY, GEMINI_MODEL,
)
from src.feature_extraction import FEATURE_KEYWORDS
from src.recommender import UserPreference


# =============================================================================
# Keyword-based parsing
# =============================================================================
LOCATION_PATTERNS = {
    "Bandung Utara":   [r"bandung\s*utara", r"\butara\b", r"dago", r"setiabudi",
                        r"lembang", r"ciumbuleuit"],
    "Bandung Selatan": [r"bandung\s*selatan", r"\bselatan\b", r"buahbatu",
                        r"kopo", r"soreang"],
    "Bandung Timur":   [r"bandung\s*timur", r"\btimur\b", r"cinambo", r"arcamanik",
                        r"ujungberung"],
    "Bandung Barat":   [r"bandung\s*barat", r"\bbarat\b", r"cimahi", r"pasirkaliki",
                        r"sukajadi"],
    "Bandung Tengah":  [r"bandung\s*tengah", r"\btengah\b", r"asia\s*afrika",
                        r"braga", r"alun\s*alun", r"riau"],
}

# Negasi pattern
NEGATION = ["tidak", "bukan", "jangan", "ga ", "gak ", "tanpa", "kurang"]


def _detect_lokasi(text: str) -> List[str]:
    """Deteksi wilayah Bandung dari teks."""
    text_low = text.lower()
    found = []
    for wilayah, patterns in LOCATION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_low):
                found.append(wilayah)
                break
    return found


def _detect_features(text: str) -> Dict[str, float]:
    """Deteksi fitur dari teks via keyword matching dengan handling negasi sederhana."""
    text_low = " " + text.lower() + " "
    features: Dict[str, float] = {}

    for feat, keywords in FEATURE_KEYWORDS.items():
        weight = 0.0
        for kw in keywords:
            kw_padded = kw.lower()
            if kw_padded in text_low:
                # Cek negasi 25 char sebelum keyword
                pos = text_low.find(kw_padded)
                window_start = max(0, pos - 25)
                window = text_low[window_start:pos]
                is_neg = any(n in window for n in NEGATION)
                if is_neg:
                    # Negatif berarti user tidak mau fitur ini
                    weight = max(weight, -0.5)
                else:
                    weight = max(weight, 1.0)
                break
        if weight != 0:
            features[feat] = weight

    return features


def _detect_constraints(text: str) -> Tuple[float, int]:
    """Deteksi rating minimum & jumlah review minimum dari kalimat."""
    text_low = text.lower()
    min_rating = 0.0
    min_reviews = 0

    # Pattern "rating minimal 4", "rating diatas 4.5", "minimal rating 4", dll
    patterns = [
        r"rating[^\d]{0,20}?(\d+(?:\.\d+)?)",
        r"(?:minimal|min|di\s*atas|diatas|>=?)\s*(\d+(?:\.\d+)?)",
        r"bintang\s*(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text_low)
        if m:
            try:
                val = float(m.group(1))
                if 1 <= val <= 5:
                    min_rating = val
                    break
            except ValueError:
                pass
    if min_rating == 0.0 and ("bintang 5" in text_low or "5 bintang" in text_low):
        min_rating = 4.5
    elif "bintang 4" in text_low or "4 bintang" in text_low:
        min_rating = 4.0

    # "popular" / "rame" / "banyak review"
    if any(w in text_low for w in ["populer", "popular", "rame", "ramai", "terkenal", "banyak review"]):
        min_reviews = 30

    return min_rating, min_reviews


def parse_query_keyword(text: str) -> UserPreference:
    """
    Parsing teks bebas dengan keyword matching (selalu jalan, gratis).
    """
    features = _detect_features(text)
    lokasi = _detect_lokasi(text)
    min_rating, min_reviews = _detect_constraints(text)

    # Default semua fitur lainnya = 0 (tidak penting)
    full_features: Dict[str, float] = {f: 0.0 for f in TARGET_FEATURES}
    for f, w in features.items():
        # Convert negatif ke 0 (artinya tidak diutamakan, tapi tidak block)
        full_features[f] = max(0.0, w)

    return UserPreference(
        features=full_features,
        lokasi=lokasi if lokasi else None,
        min_rating=min_rating,
        min_reviews=min_reviews,
    )


# =============================================================================
# Gemini-based parsing
# =============================================================================
GEMINI_PROMPT_TEMPLATE = """Anda adalah parser preferensi kafe. Diberi kalimat dari user, ekstrak preferensi mereka.

Kalimat user: "{text}"

Tugas Anda:
1. Untuk setiap fitur, beri skor 0.0 - 1.0:
   - 0.0 = user tidak peduli / tidak menyebut
   - 0.5 = user mungkin suka
   - 1.0 = user sangat ingin fitur ini
   Jika user TIDAK MAU fitur ini (negasi), tetap beri 0.0.

   Daftar fitur:
{feature_list}

2. Identifikasi wilayah Bandung yang user inginkan (boleh kosong):
   Pilihan: "Bandung Utara", "Bandung Selatan", "Bandung Timur", "Bandung Barat", "Bandung Tengah"

3. Identifikasi rating minimum (0-5, default 0).
4. Identifikasi minimum jumlah review (default 0; gunakan 30 jika user minta "populer/rame/terkenal").

Jawab HANYA dalam JSON valid TANPA markdown code fence, format persis:
{{
  "features": {{"wifi": 0.0, "stopkontak": 0.0, ...semua fitur target}},
  "lokasi": ["Bandung Utara", ...],
  "min_rating": 0.0,
  "min_reviews": 0
}}
"""


def parse_query_gemini(
    text: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[UserPreference]:
    """Parsing teks via Gemini LLM. Return None jika gagal."""
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        return None

    feature_list = "\n".join(
        f"   - {f}: {FEATURE_LABELS.get(f, f)}" for f in TARGET_FEATURES
    )
    prompt = GEMINI_PROMPT_TEMPLATE.format(text=text, feature_list=feature_list)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or GEMINI_MODEL)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown fence
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)

        full_features = {f: 0.0 for f in TARGET_FEATURES}
        for f, v in (data.get("features") or {}).items():
            if f in TARGET_FEATURES:
                try:
                    full_features[f] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    pass

        lokasi_list = [l for l in (data.get("lokasi") or []) if l in WILAYAH_BANDUNG]
        return UserPreference(
            features=full_features,
            lokasi=lokasi_list if lokasi_list else None,
            min_rating=float(data.get("min_rating", 0.0) or 0.0),
            min_reviews=int(data.get("min_reviews", 0) or 0),
        )
    except Exception as exc:
        print(f"[QueryParser] Gemini error: {exc}")
        return None


# =============================================================================
# Public API
# =============================================================================
def parse_query(text: str, prefer_llm: bool = True) -> Tuple[UserPreference, str]:
    """
    Parse teks bebas jadi UserPreference.

    Returns:
        (preference, mode_used) — mode_used = "gemini" atau "keyword"
    """
    if not text or not text.strip():
        empty = UserPreference(features={f: 0.0 for f in TARGET_FEATURES})
        return empty, "keyword"

    if prefer_llm and GEMINI_API_KEY:
        result = parse_query_gemini(text)
        if result is not None:
            return result, "gemini"

    return parse_query_keyword(text), "keyword"


if __name__ == "__main__":
    samples = [
        "saya cari kafe yang tenang ada wifi dan stopkontak buat skripsian di bandung utara",
        "kafe murah dan rame buat ngumpul, jangan yang sempit",
        "rekomen cafe dekat dago, makanannya enak, ber-AC",
        "cari tempat populer dengan rating minimal 4.5",
    ]
    for s in samples:
        pref, mode = parse_query(s, prefer_llm=False)
        print(f"\n[{mode}] {s!r}")
        active = {k: v for k, v in pref.features.items() if v > 0}
        print(f"  Fitur diutamakan: {active}")
        print(f"  Lokasi          : {pref.lokasi}")
        print(f"  Min rating      : {pref.min_rating}")
        print(f"  Min reviews     : {pref.min_reviews}")
