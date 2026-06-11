"""
Streamlit App: Sistem Rekomendasi Kafe Bandung untuk Tempat Belajar.

Cara menjalankan:
    streamlit run app.py

Halaman:
1. Beranda           - dashboard ringkasan dataset
2. Cari Rekomendasi  - form preferensi user, lalu tampilkan hasil
3. Detail Kafe       - lihat ulasan & sentimen per kafe
4. Analytics         - visualisasi distribusi sentimen, fitur, lokasi
"""

import sys
from pathlib import Path

# Pastikan src/ bisa diimport
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import (
    DATABASE_FILE, TARGET_FEATURES, FEATURE_LABELS, WILAYAH_BANDUNG,
    GEMINI_API_KEY, FEATURE_EXTRACTION_MODE,  GOOGLE_MAPS_API_KEY, RAW_DATASET,
    CLEANED_REVIEWS, SENTIMENT_REVIEWS, CAFE_FEATURES,
)
from src.database import load_cafe_features, load_reviews_for_cafe, get_database_stats
from src.recommender import UserPreference, recommend, explain_match
from src.query_parser import parse_query

from src.google_maps import (
    search_place,
    build_maps_link,
    get_map_df,
)

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Bandung Cafinder",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Cache loaders
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_features() -> pd.DataFrame:
    if not DATABASE_FILE.exists():
        return pd.DataFrame()
    return load_cafe_features()


@st.cache_data(ttl=3600, show_spinner=False)
def load_reviews(cafe_name: str) -> pd.DataFrame:
    return load_reviews_for_cafe(cafe_name)


@st.cache_data(ttl=3600, show_spinner=False)
def get_stats() -> dict:
    return get_database_stats()


# =============================================================================
# Sidebar Navigation
# =============================================================================
def sidebar_nav() -> str:
    st.sidebar.markdown("# ☕ Bandung Cafinder")
    st.sidebar.markdown("**Sistem Rekomendasi Kafe Bandung**")
    st.sidebar.caption("Berbasis ulasan Google Maps + NLP/LLM")

    page = st.sidebar.radio(
        "Navigasi",
        ["🏠 Beranda", "💬 Cari Kafe", "📋 Detail Kafe", "📊 Analytics", "ℹ️ Tentang"],
    )

    st.sidebar.markdown("---")

    # Status setup
    st.sidebar.markdown("### Status Setup")
    db_ok = DATABASE_FILE.exists()
    st.sidebar.markdown(f"{'✅' if db_ok else '❌'} Database `{DATABASE_FILE.name}`")
    
    # Gemini status
    if GEMINI_API_KEY:

        st.sidebar.markdown(
            "✅ Gemini API aktif"
        )

        st.sidebar.markdown(
            f"⚙️ Mode: "
            f"`{FEATURE_EXTRACTION_MODE}`"
        )

    else:

        st.sidebar.markdown(
            "⚠️ Gemini API kosong "
            "(pakai keyword)"
        )


    # Google Maps status

    if GOOGLE_MAPS_API_KEY:

        st.sidebar.markdown(
            "✅ Google Maps API aktif"
        )

    else:

        st.sidebar.markdown(
            "⚠️ Google Maps API kosong"
        )
    st.sidebar.markdown("---")
    st.sidebar.caption("Kelompok 30 - Capstone Project")
    return page


# =============================================================================
# Halaman: Beranda
# =============================================================================
def page_home(df: pd.DataFrame, stats: dict):
    st.title("☕ Bandung Cafinder")
    st.markdown(
        """
        Selamat datang! Sistem ini membantu Anda menemukan kafe yang sesuai untuk
        belajar atau bekerja di Kota Bandung, berdasarkan analisis ribuan ulasan
        Google Maps dengan teknik **NLP** dan **Large Language Model**.
        """
    )

    if not stats.get("db_exists"):
        st.error(
            "❌ Database belum dibangun. Jalankan dahulu:\n\n"
            "```\npython scripts/run_all.py\n```"
        )
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏪 Total Kafe", f"{stats.get('n_cafes', 0):,}")
    col2.metric("💬 Total Ulasan", f"{stats.get('n_reviews', 0):,}")
    pos = stats.get("sentiment_dist", {}).get("positive", 0)
    col3.metric("😊 Ulasan Positif", f"{pos:,}")
    neg = stats.get("sentiment_dist", {}).get("negative", 0)
    col4.metric("😞 Ulasan Negatif", f"{neg:,}")

    st.markdown("---")

    # Top cafe by rating
    st.subheader("🏆 Top 10 Kafe Berdasarkan Rating & Sentimen")
    top = df[df["n_reviews"] >= 20].copy()
    if "positive_pct" in top.columns:
        top["score"] = top["avg_rating"].fillna(0) * 0.6 + top["positive_pct"].fillna(0) * 5 * 0.4
    else:
        top["score"] = top["avg_rating"].fillna(0)
    top = top.nlargest(10, "score")[
        ["place_name", "lokasi", "avg_rating", "n_reviews", "positive_pct"]
    ].reset_index(drop=True)
    top.index = top.index + 1
    top.columns = ["Nama Kafe", "Lokasi", "Avg Rating", "Jumlah Review", "% Positif"]
    top["% Positif"] = (top["% Positif"] * 100).round(1).astype(str) + "%"
    st.dataframe(top, use_container_width=True)

    st.markdown("---")

    # Distribusi lokasi
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📍 Sebaran Kafe per Wilayah")
        lokasi_dist = stats.get("lokasi_dist", {})
        if lokasi_dist:
            df_lok = pd.DataFrame(
                {"Wilayah": list(lokasi_dist.keys()), "Jumlah": list(lokasi_dist.values())}
            )
            fig = px.bar(
                df_lok, x="Wilayah", y="Jumlah", text="Jumlah",
                color="Jumlah", color_continuous_scale="ylorbr",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💭 Distribusi Sentimen Ulasan")
        sent = stats.get("sentiment_dist", {})
        if sent:
            df_sent = pd.DataFrame(
                {"Sentimen": list(sent.keys()), "Jumlah": list(sent.values())}
            )
            color_map = {"positive": "#52C41A", "neutral": "#FAAD14", "negative": "#F5222D"}
            fig = px.pie(
                df_sent, values="Jumlah", names="Sentimen",
                color="Sentimen", color_discrete_map=color_map,
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Halaman: Cari Kafe
# =============================================================================
EXAMPLE_QUERIES = [
    "Saya cari kafe yang tenang ada wifi dan stopkontak buat skripsian di Bandung Utara",
    "Mau ngumpul dengan teman, cafe rame, makanan enak, harga terjangkau",
    "Kafe murah untuk mahasiswa, banyak colokan, tidak berisik",
    "Tempat populer di Bandung Tengah dengan suasana cozy untuk wfc",
    "Cafe nyaman dekat Dago, ber-AC, rating minimal 4.5",
]


def _render_recommendations(results: pd.DataFrame, user_pref: UserPreference, key_prefix: str = "txt"):
    """Tampilkan kartu hasil rekomendasi (digunakan di beberapa halaman)."""
    if len(results) == 0:
        st.warning("Tidak ada kafe yang cocok. Coba kurangi filter atau ubah kata kunci.")
        return

    for i, (_, row) in enumerate(
        results.iterrows()
    ):
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f"### {i+1}. {row['place_name']}"
                )

                st.markdown(
                    f"📍 **Lokasi:** {row['lokasi']}"
                )

                # =====================================
                # Google Maps API integration
                # =====================================

                maps_data = search_place(
                    row["place_name"],
                    row["lokasi"]
                )

                if maps_data:

                    st.caption(
                        f"🗺 {maps_data.get('address','-')}"
                    )

                    st.caption(
                        f"⭐ Google Maps Rating: "
                        f"{maps_data.get('rating','-')}"
                    )

                    map_df = get_map_df(
                        maps_data.get("lat"),
                        maps_data.get("lng")
                    )

                    if map_df is not None:

                        st.map(
                            map_df,
                            zoom=15,
                            use_container_width=True,
                        )

                    maps_link = build_maps_link(
                        maps_data.get("lat"),
                        maps_data.get("lng")
                    )

                    if maps_link:

                        st.link_button(
                            "📍 Buka di Google Maps",
                            maps_link,
                            key=f"maps_{i}"
                        )

                # =====================================

                for r in explain_match(
                    row,
                    user_pref
                ):
                    st.markdown(
                        f"• {r}"
                    )
            with col_b:
                st.metric("⭐ Match", f"{row['match_score']:.0%}")
                st.metric("📊 Rating", f"{row['avg_rating']:.2f}/5" if pd.notna(row['avg_rating']) else "—")
                st.caption(f"💬 {int(row['n_reviews'])} review")

            with st.expander("Lihat detail fitur"):
                feat_data = pd.DataFrame({
                    "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
                    "Skor": [row.get(f, 0) for f in TARGET_FEATURES],
                })
                fig = px.bar(
                    feat_data, x="Skor", y="Fitur", orientation="h",
                    range_x=[0, 1],
                    color="Skor", color_continuous_scale="ylorbr",
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_{i}")


def page_text_search(df: pd.DataFrame):
    st.title("💬 Cari Kafe")
    st.markdown(
        """
        Ketik apa yang Anda cari dengan kalimat biasa, sistem akan otomatis paham
        preferensi Anda. Contoh:
        """
    )

    # Tombol contoh untuk quick-fill
    cols = st.columns(
        min(
            3,
            len(EXAMPLE_QUERIES)
        )
    )

    for i, ex in enumerate(
        EXAMPLE_QUERIES[:3]
    ):

        if cols[i].button(
            f"💡 {ex[:50]}"
            f"{'...' if len(ex)>50 else ''}",
            key=f"ex_{i}",
            use_container_width=True,
        ):

            # langsung isi text area
            st.session_state[
                "text_query_input"
            ] = ex

            st.rerun()

    # Mode parser
    has_gemini = bool(GEMINI_API_KEY)
    col_q, col_m = st.columns([4, 1])
    with col_q:
        query = st.text_area(
            "Deskripsikan kafe yang Anda cari",
            placeholder=
            (
                "Contoh: cafe tenang, "
                "ada wifi & stopkontak..."
            ),
            height=100,
            key="text_query_input",
        )
    with col_m:
        st.caption("Mode parsing")
        prefer_llm = st.toggle(
            "Gunakan Gemini AI",
            value=has_gemini,
            disabled=not has_gemini,
            help="Aktifkan untuk parsing yang lebih cerdas (butuh API key di .env)",
        )
        if not has_gemini:
            st.caption("⚠️ API key kosong — pakai keyword saja")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        top_n = st.slider("Jumlah rekomendasi", 3, 20, 8)
    with col_filter2:
        min_reviews_override = st.slider("Min jumlah review", 0, 50, 5,
            help="Filter kafe yang punya cukup review (lebih reliable)")

    submit = st.button("🚀 Cari Kafe", type="primary", use_container_width=True)

    if not submit:
        st.info("👆 Ketik kalimat di atas, lalu tekan **Cari Kafe**")
        return
    if not query or not query.strip():
        st.warning("Mohon isi kalimat pencarian dulu.")
        return

    with st.spinner("Memahami preferensi Anda..."):
        user_pref, mode = parse_query(query, prefer_llm=prefer_llm)
        # Override min_reviews dari slider
        user_pref.min_reviews = max(user_pref.min_reviews, min_reviews_override)

    # Tampilkan hasil parsing supaya user paham apa yang sistem tangkap
    with st.expander("🧠 Apa yang sistem tangkap dari kalimat Anda", expanded=True):
        st.markdown(f"**Mode parsing:** `{mode}` " + ("🤖 (LLM)" if mode == "gemini" else "🔍 (keyword)"))

        active_features = {f: w for f, w in user_pref.features.items() if w > 0.1}
        if active_features:
            st.markdown("**Fitur yang Anda inginkan:**")
            cols_f = st.columns(min(4, len(active_features)))
            for i, (feat, w) in enumerate(active_features.items()):
                with cols_f[i % len(cols_f)]:
                    st.metric(FEATURE_LABELS.get(feat, feat), f"{w:.0%}")
        else:
            st.caption("(tidak terdeteksi fitur spesifik — sistem akan mencari berdasarkan rating tertinggi)")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**Lokasi:** {', '.join(user_pref.lokasi) if user_pref.lokasi else '_semua wilayah_'}")
        with col_b:
            st.markdown(f"**Min Rating:** {user_pref.min_rating if user_pref.min_rating else '_tidak dibatasi_'}")
        with col_c:
            st.markdown(f"**Min Review:** {user_pref.min_reviews if user_pref.min_reviews else '_tidak dibatasi_'}")

    with st.spinner("Mencari kafe terbaik..."):
        results = recommend(df, user_pref, top_n=top_n)

    st.markdown("---")
    if len(results) > 0:
        st.success(f"✅ Ditemukan **{len(results)} kafe** yang cocok!")
    _render_recommendations(results, user_pref, key_prefix="txt")


# =============================================================================
# Halaman: Detail Kafe
# =============================================================================
def page_detail(df: pd.DataFrame):
    st.title("📋 Detail Kafe")

    cafe_list = sorted(df["place_name"].dropna().unique().tolist())
    default_idx = 0
    selected_default = st.session_state.get("selected_cafe")
    if selected_default and selected_default in cafe_list:
        default_idx = cafe_list.index(selected_default)

    selected_cafe = st.selectbox("Pilih kafe", cafe_list, index=default_idx)

    cafe_row = df[df["place_name"] == selected_cafe].iloc[0]

    # Info utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📍 Lokasi", cafe_row["lokasi"])
    col2.metric("⭐ Rating", f"{cafe_row['avg_rating']:.2f}" if pd.notna(cafe_row['avg_rating']) else "—")
    col3.metric("💬 Total Review", f"{int(cafe_row['n_reviews']):,}")
    col4.metric("😊 % Positif", f"{cafe_row['positive_pct']*100:.1f}%" if pd.notna(cafe_row['positive_pct']) else "—")

    st.markdown("---")

    # Profil fitur
    st.subheader("📊 Profil Fasilitas Kafe")
    feat_df = pd.DataFrame({
        "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
        "Skor": [cafe_row.get(f, 0) for f in TARGET_FEATURES],
    })
    fig = px.bar(
        feat_df, x="Fitur", y="Skor",
        text=feat_df["Skor"].apply(lambda x: f"{x:.2f}"),
        color="Skor", color_continuous_scale="ylorbr",
        range_y=[0, 1],
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Ulasan
    st.subheader("💬 Ulasan Pengguna")
    reviews = load_reviews(
        selected_cafe or ""
    )
    if len(reviews) == 0:
        st.info("Tidak ada ulasan")
        return

    # Filter sentimen
    col_a, col_b = st.columns([1, 2])
    with col_a:
        sentiment_filter = st.radio(
            "Filter sentimen",
            ["Semua", "positive", "negative", "neutral"],
            horizontal=False,
        )
        st.caption(f"Total: {len(reviews)}")
        st.caption(f"Positive: {(reviews['sentiment']=='positive').sum()}")
        st.caption(f"Negative: {(reviews['sentiment']=='negative').sum()}")
        st.caption(f"Neutral: {(reviews['sentiment']=='neutral').sum()}")

    with col_b:
        if sentiment_filter != "Semua":
            filtered = reviews[reviews["sentiment"] == sentiment_filter]
        else:
            filtered = reviews

        SHOW_LIMIT = 30
        for _, r in filtered.head(SHOW_LIMIT).iterrows():
            sent = r["sentiment"]
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sent, "•")
            color = {"positive": "#52C41A", "negative": "#F5222D", "neutral": "#FAAD14"}.get(sent, "#888")
            with st.container(border=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{emoji} {r['username'] or 'Anonim'}**")
                    color_name = {"positive": "green", "negative": "red", "neutral": "orange"}.get(sent, "gray")
                    st.caption(f"⭐ Rating: {r['rating']}/5 — Sentimen: :{color_name}[{sent}] (skor: {r['sentiment_score']:.2f})")
                    st.write(r["review_raw"])
                with cols[1]:
                    st.markdown(f"<div style='font-size:24px; color:{color}; text-align:center'>{emoji}</div>", unsafe_allow_html=True)

        if len(filtered) > SHOW_LIMIT:
            st.caption(f"Menampilkan {SHOW_LIMIT} dari {len(filtered)} ulasan")


# =============================================================================
# Halaman: Analytics
# =============================================================================
def page_analytics(df: pd.DataFrame):
    st.title("📊 Analytics & Insight")

    st.subheader("📈 Rata-rata Skor Fitur Across Cafes")
    avg_features = pd.DataFrame({
        "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
        "Rata-rata Skor": [df[f].mean() for f in TARGET_FEATURES if f in df.columns],
    })
    fig = px.bar(
        avg_features, x="Fitur", y="Rata-rata Skor",
        text=avg_features["Rata-rata Skor"].apply(lambda x: f"{x:.2f}"),
        color="Rata-rata Skor", color_continuous_scale="ylorbr",
        range_y=[0, 1],
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🗺️ Rating Rata-rata per Wilayah")
    by_lokasi = (
        df.groupby("lokasi")
        .agg(avg_rating=("avg_rating", "mean"), n_cafes=("place_name", "count"))
        .reset_index()
    )
    fig2 = px.bar(
        by_lokasi.sort_values("avg_rating", ascending=False),
        x="lokasi", y="avg_rating",
        text="avg_rating",
        color="avg_rating", color_continuous_scale="ylorbr",
    )
    fig2.update_traces(texttemplate="%{text:.2f}")
    fig2.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🏆 Top Kafe per Fitur")
    selected_feat = st.selectbox(
        "Pilih fitur",
        TARGET_FEATURES,
        format_func=lambda f:
            FEATURE_LABELS.get(f)
            or str(f),
    )
    top10 = df.nlargest(10, selected_feat)[
        ["place_name", "lokasi", selected_feat, "avg_rating", "n_reviews"]
    ].reset_index(drop=True)
    top10.index = top10.index + 1
    top10.columns = ["Kafe", "Lokasi", "Skor", "Rating", "# Review"]
    top10["Skor"] = top10["Skor"].round(3)
    st.dataframe(top10, use_container_width=True)

    st.markdown("---")

    st.subheader("📐 Korelasi antar Fitur")
    feat_corr = df[TARGET_FEATURES].corr()
    fig3 = px.imshow(
        feat_corr,
        labels=dict(x="Fitur", y="Fitur", color="Korelasi"),
        x=[FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
        y=[FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# Halaman: Tentang
# =============================================================================
def page_about():
    st.title("ℹ️ Tentang Sistem Ini")

    st.markdown(
        """
### 🎯 Tujuan
Sistem rekomendasi kafe di Bandung sebagai tempat belajar, berbasis ulasan Google Maps,
dengan pendekatan **NLP** dan **Large Language Model**.

### 🏗️ Arsitektur Pipeline

1. **Preprocessing** — pembersihan teks ulasan: hapus emoji/URL/karakter spesial,
   normalisasi slang Bahasa Indonesia, penghapusan stopwords (Sastrawi), deduplikasi.
2. **Sentiment Analysis** — klasifikasi positif/netral/negatif menggunakan kombinasi
   *lexicon-based scoring* dan *rating-based scoring*, dengan handling negasi.
3. **Feature Extraction** — ekstraksi fitur fasilitas (WiFi, stopkontak, kenyamanan, dll)
   menggunakan keyword matching, dengan opsional enhancement via Gemini LLM.
4. **Database** — penyimpanan terstruktur SQLite (cafes, reviews, features).
5. **Recommendation Engine** — content-based recommendation menggunakan
   *cosine similarity* antara profil fitur kafe dan preferensi pengguna,
   dengan bobot tambahan dari rating & sentimen.
6. **Web App** — Streamlit untuk UI interaktif.

### 📊 Dataset
Data ulasan Google Maps kafe di Bandung (Utara, Selatan, Timur, Barat, Tengah)
hasil scraping dengan Playwright. Lihat `data/dataset_final.csv`.

### 🔑 Konfigurasi LLM
Untuk hasil ekstraksi fitur yang lebih akurat, sistem dapat menggunakan **Gemini API**:
1. Dapatkan API key gratis di https://aistudio.google.com/app/apikey
2. Salin file `.env.example` menjadi `.env`
3. Isi `GEMINI_API_KEY=...`
4. Ubah `FEATURE_EXTRACTION_MODE=hybrid`
5. Jalankan ulang `python scripts/03_extract_features.py`

### 👥 Tim Kelompok 30
- Zahra Nur Azizah — Data Scraping & NLP
- Winston Lokeswara Mangori — NLP & LLM
- Yan Andhinaya Ardika — Recommendation System & Database
- Yohana Lydia — Backend & API
- Yayang Ananda Setya — Frontend & UI
        """
    )


# =============================================================================
# Main
# =============================================================================
def main():
    page = sidebar_nav()
    df = load_features()
    stats = get_stats()

    if page == "🏠 Beranda":
        page_home(df, stats)
    elif page == "💬 Cari Kafe":
        if len(df) == 0:
            st.error("Database kosong. Jalankan `python scripts/run_all.py` terlebih dahulu.")
        else:
            page_text_search(df)
    elif page == "📋 Detail Kafe":
        if len(df) == 0:
            st.error("Database kosong. Jalankan `python scripts/run_all.py` terlebih dahulu.")
        else:
            page_detail(df)
    elif page == "📊 Analytics":
        if len(df) == 0:
            st.error("Database kosong. Jalankan `python scripts/run_all.py` terlebih dahulu.")
        else:
            page_analytics(df)
    elif page == "ℹ️ Tentang":
        page_about()


if __name__ == "__main__":
    main()
