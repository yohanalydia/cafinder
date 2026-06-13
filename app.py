"""
Streamlit App: Sistem Rekomendasi Kafe Bandung untuk Tempat Belajar.

Cara menjalankan:
    streamlit run app.py

Perbaikan v3:
- SMOOTH_SCROLL_JS pakai st.components.v1.html() agar persistent setelah rerun,
  + MutationObserver untuk re-attach listener + IntersectionObserver active nav state
- scroll_margin_top menggantikan top:-70px pada section-anchor
- fadeInUp hanya jalan sekali via sessionStorage flag (tidak re-trigger setiap rerun)
- st.rerun() setelah klik example query simpan & restore scroll position via JS
- search_place() dibungkus try/except per kafe di loop rekomendasi
- emoji_map / color_map dipindah ke luar loop di render_detail
- .metric-card-link dead CSS dihapus
- Semua widget key di-prefix nama section supaya tidak DuplicateWidgetID
- Hover border stVerticalBlockBorderWrapper diperbaiki ke terra-mid
- Plotly heatmap korelasi pakai diverging scale custom (mint ↔ terra) bukan RdBu
- render_tentang diperindah: cards per pipeline step + kolom tim
- Spacing antar section distandarisasi via CSS .section-spacer
- checker-strip hanya di section_heading, tidak duplikat
- Landing page CAFINDER ditambahkan sebelum dashboard utama
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from src.config import (
    DATABASE_FILE, TARGET_FEATURES, FEATURE_LABELS,
    GEMINI_API_KEY, FEATURE_EXTRACTION_MODE, GOOGLE_MAPS_API_KEY,
)
from src.database import load_cafe_features, load_reviews_for_cafe, get_database_stats
from src.recommender import UserPreference, recommend, explain_match
from src.query_parser import parse_query
from src.google_maps import search_place, build_maps_link, get_map_df


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="CAFINDER – Kafe Bandung",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================
VINTAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');

/* ── SCROLL ───────────────────────────────────────────────────────────────── */
html,
[data-testid="stAppViewContainer"],
section[data-testid="stMain"],
.main {
    scroll-behavior: smooth !important;
}

/* ── TOKENS ───────────────────────────────────────────────────────────────── */
:root {
    --mint-dark:   #4A9478;
    --mint-mid:    #6BB89A;
    --mint-light:  #A8D8C4;
    --mint-pale:   #E8F5EF;

    --terra-dark:  #B03A20;
    --terra-mid:   #D9603B;
    --terra-light: #E8845E;
    --terra-pale:  #F9E4DC;

    --cream:       #FFF8F0;
    --brown-dark:  #3D2B1F;
    --brown-mid:   #6B4C3B;
    --brown-light: #9C7B6A;

    --checker-red: #C44B2B;

    --shadow-warm: 0 2px 12px rgba(61,43,31,0.12);
    --shadow-deep: 0 4px 24px rgba(61,43,31,0.18);

    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 20px;
}

/* ── GLOBAL ───────────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--cream) !important;
    background-image:
        linear-gradient(45deg, var(--terra-mid) 25%, transparent 25%),
        linear-gradient(-45deg, var(--terra-mid) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, var(--terra-mid) 75%),
        linear-gradient(-45deg, transparent 75%, var(--terra-mid) 75%) !important;
    background-size: 40px 40px !important;
    background-position: 0 0, 0 20px, 20px -20px, -20px 0px !important;
    font-family: 'Lato', sans-serif !important;
    color: var(--brown-dark) !important;
}

.stApp > header,
section[data-testid="stMain"],
section[data-testid="stMain"] > div,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.appview-container > section.main,
.block-container {
    background-color: var(--cream) !important;
}

.main .block-container {
    background-color: var(--cream) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
    box-shadow: 0 8px 40px rgba(61,43,31,0.28) !important;
    border: 2.5px solid var(--terra-mid) !important;
}

/* ── ANIMATION ────────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

body:not(.app-loaded)
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > div {
    animation: fadeInUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}

body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(1)  { animation-delay: 0.00s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(2)  { animation-delay: 0.04s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(3)  { animation-delay: 0.08s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(4)  { animation-delay: 0.12s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(5)  { animation-delay: 0.16s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(6)  { animation-delay: 0.20s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(7)  { animation-delay: 0.24s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(8)  { animation-delay: 0.28s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(9)  { animation-delay: 0.32s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(10) { animation-delay: 0.36s; }
body:not(.app-loaded) [data-testid="stVerticalBlock"] > div:nth-child(n+11){ animation-delay: 0.40s; }

@media (prefers-reduced-motion: reduce) {
    [data-testid="stVerticalBlock"] > div { animation: none !important; }
}

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, var(--mint-dark) 0%, var(--mint-mid) 60%, #5DA890 100%) !important;
    border-right: 3px solid var(--terra-mid) !important;
}

[data-testid="stSidebar"] * { color: var(--cream) !important; }

[data-testid="stSidebar"] code {
    background: rgba(0,0,0,0.25) !important;
    color: #FFE8B0 !important;
    border-radius: 3px !important;
    padding: 0.1rem 0.35rem !important;
    font-size: 0.85em !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--cream) !important;
    font-family: 'Playfair Display', serif !important;
}

[data-testid="stSidebar"] hr { border-color: rgba(255,248,240,0.3) !important; }

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--mint-pale) !important;
    opacity: 0.85;
}

/* Sidebar anchor nav */
.nav-link {
    display: block;
    padding: 0.5rem 0.8rem;
    margin: 0.2rem 0;
    border-radius: var(--radius-sm);
    color: var(--cream) !important;
    text-decoration: none !important;
    font-family: 'Lato', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: background 0.18s ease, transform 0.15s ease, border-left 0.18s ease;
    border-left: 3px solid transparent;
    cursor: pointer;
}

.nav-link:hover,
.nav-link.active {
    background: rgba(255,248,240,0.18) !important;
    transform: translateX(4px);
    color: var(--cream) !important;
    border-left: 3px solid var(--terra-light);
}

/* In-content metric shortcut links */
.metric-nav-link {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    margin-top: 0.3rem;
    border-radius: var(--radius-sm);
    background: var(--mint-pale) !important;
    color: var(--brown-dark) !important;
    text-decoration: none !important;
    font-family: 'Lato', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    border: 1px solid var(--mint-light);
    transition: background 0.18s ease, color 0.18s ease, transform 0.15s ease;
    cursor: pointer;
}

.metric-nav-link:hover {
    background: var(--terra-pale) !important;
    color: var(--terra-dark) !important;
    border-color: var(--terra-light) !important;
    transform: translateX(2px);
}

/* ── TYPOGRAPHY ───────────────────────────────────────────────────────────── */
h1, .stTitle {
    font-family: 'Playfair Display', serif !important;
    color: var(--mint-dark) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    border-bottom: 3px solid var(--terra-mid) !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}

h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--brown-dark) !important;
    font-weight: 600 !important;
}

h4, h5, h6 {
    font-family: 'Lato', sans-serif !important;
    color: var(--brown-mid) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-size: 0.85rem !important;
}

p, li, .stMarkdown p {
    font-family: 'Lato', sans-serif !important;
    color: var(--brown-mid) !important;
    line-height: 1.7 !important;
}

/* ── SECTION ANCHOR & HEADER ──────────────────────────────────────────────── */
.section-anchor {
    display: block;
    visibility: hidden;
    pointer-events: none;
    scroll-margin-top: 20px;
    height: 0;
    margin: 0;
    padding: 0;
}

.section-spacer {
    height: 3rem;
    display: block;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0 0 0.2rem 0;
    padding: 1rem 1.4rem;
    background: linear-gradient(90deg, var(--mint-pale) 0%, transparent 100%);
    border-left: 4px solid var(--terra-mid);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
}

.section-header .section-emoji { font-size: 1.6rem; line-height: 1; }

.section-header .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--brown-dark);
    margin: 0;
    line-height: 1.2;
}

/* ── CHECKER STRIP ────────────────────────────────────────────────────────── */
.checker-strip {
    width: 100%;
    height: 14px;
    background-image: repeating-linear-gradient(
        90deg,
        var(--checker-red) 0px, var(--checker-red) 14px,
        var(--cream) 14px, var(--cream) 28px
    );
    border-radius: 4px;
    margin: 0.3rem 0 1.4rem 0;
    opacity: 0.65;
}

/* ── LANDING PAGE ─────────────────────────────────────────────────────────── */
.lp-wrap {
    text-align: center;
    padding: 2rem 2rem 1rem;
}

.lp-badge {
    display: inline-block;
    background: var(--mint-pale);
    border: 1px solid var(--mint-light);
    border-radius: var(--radius-sm);
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--mint-dark);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.lp-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    color: var(--brown-dark);
    letter-spacing: -1px;
    line-height: 1;
    margin: 0 0 0.5rem;
}

.lp-title span { color: var(--terra-mid); }

.lp-tagline {
    font-size: 1.1rem;
    color: var(--brown-mid);
    font-weight: 300;
    letter-spacing: 0.3px;
    margin-bottom: 0;
}

.lp-desc {
    font-size: 0.96rem;
    color: var(--brown-mid);
    line-height: 1.85;
    max-width: 600px;
    margin: 1rem auto 1.8rem;
    text-align: justify;
    display: block;
}

/* paksa wrapper Streamlit di sekitar lp-desc ikut center */
.lp-desc-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.lp-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}

.lp-pill {
    background: var(--terra-pale);
    border: 1px solid var(--terra-light);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--terra-dark);
}

.lp-members-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--brown-dark);
    font-weight: 600;
    margin-bottom: 0.9rem;
    text-align: center;
}

.lp-members {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.lp-member {
    background: linear-gradient(135deg, var(--mint-pale), var(--terra-pale));
    border: 1px solid var(--mint-light);
    border-radius: var(--radius-sm);
    padding: 0.45rem 1rem;
    font-size: 0.82rem;
    color: var(--brown-dark);
}

.lp-member strong { color: var(--mint-dark); }

.lp-note {
    font-size: 0.78rem;
    color: var(--brown-light);
    text-align: center;
    margin-top: 0.5rem;
}

/* ── METRIC CARDS ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--cream) !important;
    border: 1.5px solid var(--mint-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow-warm) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    cursor: default !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-deep) !important;
    border-color: var(--terra-mid) !important;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--brown-light) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--terra-dark) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── BUTTONS ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--terra-mid) 0%, var(--terra-dark) 100%) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 2px 8px rgba(217,96,59,0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--terra-light) 0%, var(--terra-mid) 100%) !important;
    color: var(--cream) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(217,96,59,0.45) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--terra-dark) 0%, #8B2E18 100%) !important;
    color: var(--cream) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--terra-mid) 0%, var(--terra-dark) 100%) !important;
    color: var(--cream) !important;
}

.stButton > button *, .stButton > button p, .stButton > button span {
    color: var(--cream) !important;
}

.stLinkButton a {
    background: var(--terra-pale) !important;
    color: var(--terra-dark) !important;
    border: 1.5px solid var(--terra-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
}

.stLinkButton a:hover {
    background: var(--terra-mid) !important;
    color: var(--cream) !important;
}

/* ── INPUTS ───────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: var(--cream) !important;
    border: 1.5px solid var(--mint-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--brown-dark) !important;
    font-family: 'Lato', sans-serif !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--terra-mid) !important;
    box-shadow: 0 0 0 2px rgba(217,96,59,0.2) !important;
}

.stToggle [role="switch"][aria-checked="true"] { background: var(--mint-dark) !important; }
.stRadio [data-testid="stWidgetLabel"] { color: var(--brown-mid) !important; font-weight: 700 !important; }

/* ── DATAFRAME ────────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1.5px solid var(--terra-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

.stDataFrame thead th {
    background: var(--mint-dark) !important;
    color: var(--cream) !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 0.8rem !important;
}

.stDataFrame tbody tr:nth-child(even) { background: var(--mint-pale) !important; }
.stDataFrame tbody tr:hover { background: var(--terra-pale) !important; }

[data-testid="stDataFrame"] {
    background: var(--cream) !important;
    border-radius: var(--radius-md) !important;
}

/* ── CONTAINERS ───────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.5px solid var(--terra-light) !important;
    border-radius: var(--radius-md) !important;
    background: var(--cream) !important;
    box-shadow: 0 2px 12px rgba(61,43,31,0.15) !important;
    padding: 1rem !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-deep) !important;
    border-color: var(--terra-mid) !important;
}

/* ── PIPELINE CARDS (render_tentang) ──────────────────────────────────────── */
.pipeline-card {
    background: var(--cream);
    border: 1.5px solid var(--mint-light);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow-warm);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.pipeline-card:hover {
    transform: translateY(-2px);
    border-color: var(--terra-mid);
    box-shadow: var(--shadow-deep);
}

.pipeline-card .pc-step {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--terra-dark);
    margin-right: 0.5rem;
}

.pipeline-card .pc-title {
    font-family: 'Lato', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--mint-dark);
}

.pipeline-card .pc-body {
    font-family: 'Lato', sans-serif;
    font-size: 0.88rem;
    color: var(--brown-mid);
    line-height: 1.6;
    margin-top: 0.4rem;
}

.team-card {
    background: linear-gradient(135deg, var(--mint-pale) 0%, var(--terra-pale) 100%);
    border: 1.5px solid var(--mint-light);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    text-align: center;
    box-shadow: var(--shadow-warm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.team-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-deep); }
.team-card .tc-name { font-family: 'Playfair Display', serif; font-weight: 700; color: var(--brown-dark); font-size: 0.95rem; }
.team-card .tc-role { font-family: 'Lato', sans-serif; font-size: 0.78rem; color: var(--brown-light); margin-top: 0.2rem; }

/* ── EXPANDER ─────────────────────────────────────────────────────────────── */
.stExpander {
    border: 1.5px solid var(--terra-light) !important;
    border-radius: var(--radius-md) !important;
    background: var(--cream) !important;
}

.stExpander summary {
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    color: var(--mint-dark) !important;
}

/* ── ALERTS ───────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left-width: 4px !important;
    font-family: 'Lato', sans-serif !important;
}

/* ── HR ───────────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 2px dashed var(--mint-light) !important;
    margin: 1.5rem 0 !important;
}

/* ── MAP / CHART ──────────────────────────────────────────────────────────── */
[data-testid="stDeckGlJsonChart"],
[data-testid="stMap"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1.5px solid var(--mint-light) !important;
}

[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--mint-pale); }
::-webkit-scrollbar-thumb { background: var(--terra-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--terra-mid); }

/* ── MISC ─────────────────────────────────────────────────────────────────── */
.stCaption, small { color: var(--brown-light) !important; font-family: 'Lato', sans-serif !important; }
.stSpinner > div { border-top-color: var(--terra-mid) !important; }

@media (max-width: 768px) {
    .main .block-container { padding: 1rem !important; border-radius: var(--radius-md) !important; }
    .lp-title { font-size: 2.8rem !important; }
}
</style>
"""

# =============================================================================
# JavaScript — persistent smooth scroll + active nav + one-time animation flag
# =============================================================================
SCROLL_JS = """
<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:transparent;">
<script>
(function () {
    function getMainDoc() {
        return window.parent.document || document;
    }

    function scrollToId(id) {
        const doc = getMainDoc();
        const el = doc.getElementById(id);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    (function setupAnimation() {
        const doc = getMainDoc();
        if (sessionStorage.getItem('cafe_app_loaded')) {
            doc.body.classList.add('app-loaded');
        } else {
            setTimeout(function () {
                doc.body.classList.add('app-loaded');
                sessionStorage.setItem('cafe_app_loaded', '1');
            }, 900);
        }
    })();

    function attachScrollHandlers(doc) {
        doc.addEventListener('click', function (e) {
            const a = e.target.closest('a[href^="#"]');
            if (!a) return;
            const hash = a.getAttribute('href');
            if (!hash || hash === '#') return;
            e.preventDefault();
            const id = hash.slice(1);
            scrollToId(id);
            try { window.parent.history.pushState(null, '', hash); } catch(_) {}
        }, true);
    }

    function setupActiveNav(doc) {
        const sections = doc.querySelectorAll('.section-anchor[id]');
        if (!sections.length) return;

        const navLinks = doc.querySelectorAll('.nav-link[href^="#"]');
        if (!navLinks.length) return;

        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                const id = entry.target.id;
                navLinks.forEach(function (link) {
                    const linkId = link.getAttribute('href').slice(1);
                    link.classList.toggle('active', linkId === id);
                });
            });
        }, {
            root: null,
            rootMargin: '0px 0px -70% 0px',
            threshold: 0
        });

        sections.forEach(function (el) { observer.observe(el); });
    }

    (function restoreScroll() {
        const doc = getMainDoc();
        const savedY = sessionStorage.getItem('cafe_scroll_y');
        if (savedY) {
            sessionStorage.removeItem('cafe_scroll_y');
            setTimeout(function () {
                const scrollable = doc.querySelector('[data-testid="stAppViewContainer"]')
                    || doc.querySelector('section[data-testid="stMain"]')
                    || doc.documentElement;
                scrollable.scrollTop = parseInt(savedY, 10);
            }, 120);
        }
    })();

    function init() {
        const doc = getMainDoc();
        attachScrollHandlers(doc);
        setupActiveNav(doc);

        const mo = new MutationObserver(function () {
            setupActiveNav(doc);
        });
        const target = doc.querySelector('[data-testid="stSidebar"]') || doc.body;
        mo.observe(target, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window._cafeSaveScroll = function () {
        const doc = getMainDoc();
        const scrollable = doc.querySelector('[data-testid="stAppViewContainer"]')
            || doc.querySelector('section[data-testid="stMain"]')
            || doc.documentElement;
        sessionStorage.setItem('cafe_scroll_y', scrollable.scrollTop);
    };
})();
</script>
</body></html>
"""

CHECKER_STRIP = '<div class="checker-strip"></div>'

SECTIONS = [
    ("beranda",   "🏠", "Beranda"),
    ("cari",      "💬", "Cari Kafe"),
    ("detail",    "📋", "Detail Kafe"),
    ("analytics", "📊", "Analytics"),
    ("tentang",   "ℹ️",  "Tentang"),
]

CORR_SCALE = [
    [0.0,  "#4A9478"],
    [0.25, "#A8D8C4"],
    [0.5,  "#FFF8F0"],
    [0.75, "#E8845E"],
    [1.0,  "#B03A20"],
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,251,247,1)",
    font=dict(family="Lato, sans-serif", color="#3D2B1F"),
)

TERRA_SCALE = [[0, "#F9E4DC"], [0.5, "#D9603B"], [1, "#B03A20"]]

_SENT_EMOJI  = {"positive": "😊", "negative": "😞", "neutral": "😐"}
_SENT_COLOR  = {"positive": "#6B4C3B", "negative": "#B03A20", "neutral": "#D9603B"}
_SENT_NAME   = {"positive": "green",   "negative": "red",     "neutral": "orange"}


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
# Helpers
# =============================================================================
def section_anchor(section_id: str) -> None:
    st.markdown(
        f'<div id="{section_id}" class="section-anchor"></div>',
        unsafe_allow_html=True,
    )


def section_spacer() -> None:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)


def section_heading(emoji: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-emoji">{emoji}</span>'
        f'<span class="section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(CHECKER_STRIP, unsafe_allow_html=True)


# =============================================================================
# Landing Page
# =============================================================================
def render_landing() -> None:
    """Halaman awal CAFINDER — ditampilkan sebelum masuk ke dashboard utama."""

    # Hero + deskripsi dalam satu blok HTML agar centering terkontrol penuh
    st.markdown(
        """
        <div class="lp-wrap">
            <div class="lp-badge">☕ Kelompok 30 &nbsp;·&nbsp; Capstone Project</div>
            <div class="lp-title">CAFI<span>NDER</span></div>
            <div class="lp-tagline">Temukan kafe terbaik untuk belajar di Bandung, berbasis AI</div>
            <div class="checker-strip"></div>
            <div class="lp-desc-wrap">
                <p class="lp-desc">
                    <strong>CAFINDER</strong> adalah sistem rekomendasi kafe di Kota Bandung yang
                    dirancang khusus untuk mendukung aktivitas belajar dan bekerja. Dengan menganalisis
                    ribuan ulasan Google Maps menggunakan teknik <em>Natural Language Processing (NLP)</em>
                    dan <em>Large Language Model</em>, CAFINDER memahami preferensi Anda secara alami,
                    cukup ketik kalimat biasa dan sistem akan menemukan kafe yang paling cocok untuk Anda.
                </p>
            </div>
            <div class="lp-pills">
                <span class="lp-pill">🤖 NLP &amp; Large Language Model</span>
                <span class="lp-pill">📍 5 Wilayah Bandung</span>
                <span class="lp-pill">💬 Analisis Sentimen Ulasan</span>
                <span class="lp-pill">📶 Filter WiFi, Colokan, AC &amp; Fasilitas Lain</span>
                <span class="lp-pill">⭐ Ribuan Review Google Maps</span>
                <span class="lp-pill">🗺️ Integrasi Google Maps</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Anggota tim
    st.markdown(
        """
        <div class="lp-members-title">👥 Dibuat oleh Kelompok 30</div>
        <div class="lp-members">
            <div class="lp-member"><strong>Zahra Nur Azizah</strong>&nbsp;·&nbsp;Data Scraping &amp; NLP</div>
            <div class="lp-member"><strong>Winston Lokeswara Mangori</strong>&nbsp;·&nbsp;NLP &amp; LLM</div>
            <div class="lp-member"><strong>Yan Andhinaya Ardika</strong>&nbsp;·&nbsp;Recommendation System &amp; Database</div>
            <div class="lp-member"><strong>Yohana Lydia</strong>&nbsp;·&nbsp;Backend &amp; API</div>
            <div class="lp-member"><strong>Yayang Ananda Setya</strong>&nbsp;·&nbsp;Frontend &amp; UI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(CHECKER_STRIP, unsafe_allow_html=True)

    # Tombol masuk — ditengahkan pakai kolom
    col_l, col_btn, col_r = st.columns([2, 3, 2])
    with col_btn:
        if st.button(
            "☕  Mulai Cari Kafe  →",
            type="primary",
            use_container_width=True,
            key="landing_enter_btn",
        ):
            st.session_state["landing_done"] = True
            st.rerun()

    st.markdown(
        '<p class="lp-note">Klik tombol di atas untuk masuk ke dashboard CAFINDER</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> None:
    st.sidebar.markdown("# ☕ CAFINDER")
    st.sidebar.markdown("**Sistem Rekomendasi Kafe Bandung**")
    st.sidebar.caption("Berbasis ulasan Google Maps + NLP/LLM")
    st.sidebar.markdown("---")

    nav_html = "<nav>" + "".join(
        f'<a href="#{sec_id}" class="nav-link" data-section="{sec_id}">'
        f'{emoji}&nbsp;&nbsp;{label}</a>\n'
        for sec_id, emoji, label in SECTIONS
    ) + "</nav>"
    st.sidebar.markdown(nav_html, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status Setup")

    db_ok = DATABASE_FILE.exists()
    st.sidebar.markdown(f"{'✅' if db_ok else '❌'} Database `{DATABASE_FILE.name}`")

    if GEMINI_API_KEY:
        st.sidebar.markdown("✅ Gemini API aktif")
        st.sidebar.markdown(f"⚙️ Mode: `{FEATURE_EXTRACTION_MODE}`")
    else:
        st.sidebar.markdown("⚠️ Gemini API kosong (pakai keyword)")

    st.sidebar.markdown(
        f"{'✅' if GOOGLE_MAPS_API_KEY else '⚠️'} Google Maps API "
        f"{'aktif' if GOOGLE_MAPS_API_KEY else 'kosong'}"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Kelompok 30 - Capstone Project")


# =============================================================================
# Section: Beranda
# =============================================================================
def render_beranda(df: pd.DataFrame, stats: dict) -> None:
    section_anchor("beranda")
    section_heading("🏠", "Sistem Rekomendasi Kafe Bandung")

    st.markdown(
        "Selamat datang! Sistem ini membantu Anda menemukan kafe yang sesuai untuk "
        "belajar atau bekerja di Kota Bandung, berdasarkan analisis ribuan ulasan "
        "Google Maps dengan teknik **NLP** dan **Large Language Model**."
    )

    if not stats.get("db_exists"):
        st.error(
            "❌ Database belum dibangun. Jalankan dahulu:\n\n"
            "```\npython scripts/run_all.py\n```"
        )
        return

    pos = stats.get("sentiment_dist", {}).get("positive", 0)
    neg = stats.get("sentiment_dist", {}).get("negative", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏪 Total Kafe", f"{stats.get('n_cafes', 0):,}")
        st.markdown(
            '<a href="#detail" class="metric-nav-link">→ Lihat detail kafe</a>',
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("💬 Total Ulasan", f"{stats.get('n_reviews', 0):,}")
        st.markdown(
            '<a href="#analytics" class="metric-nav-link">→ Lihat analytics</a>',
            unsafe_allow_html=True,
        )
    with col3:
        st.metric("😊 Ulasan Positif", f"{pos:,}")
        st.markdown(
            '<a href="#analytics" class="metric-nav-link">→ Distribusi sentimen</a>',
            unsafe_allow_html=True,
        )
    with col4:
        st.metric("😞 Ulasan Negatif", f"{neg:,}")
        st.markdown(
            '<a href="#analytics" class="metric-nav-link">→ Distribusi sentimen</a>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

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

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📍 Sebaran Kafe per Wilayah")
        lokasi_dist = stats.get("lokasi_dist", {})
        if lokasi_dist:
            df_lok = pd.DataFrame({
                "Wilayah": list(lokasi_dist.keys()),
                "Jumlah": list(lokasi_dist.values()),
            })
            fig = px.bar(
                df_lok, x="Wilayah", y="Jumlah", text="Jumlah",
                color="Jumlah", color_continuous_scale=TERRA_SCALE,
            )
            fig.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💭 Distribusi Sentimen Ulasan")
        sent = stats.get("sentiment_dist", {})
        if sent:
            df_sent = pd.DataFrame({
                "Sentimen": list(sent.keys()),
                "Jumlah": list(sent.values()),
            })
            fig = px.pie(
                df_sent, values="Jumlah", names="Sentimen",
                color="Sentimen",
                color_discrete_map={"positive": "#6B4C3B", "neutral": "#D9603B", "negative": "#B03A20"},
                hole=0.4,
            )
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Lato, sans-serif", color="#3D2B1F"))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Section: Cari Kafe
# =============================================================================
EXAMPLE_QUERIES = [
    "Saya cari kafe yang tenang ada wifi dan stopkontak buat skripsian di Bandung Utara",
    "Mau ngumpul dengan teman, cafe rame, makanan enak, harga terjangkau",
    "Cafe nyaman dekat Dago, ber-AC, rating minimal 4.5",
    "Tempat populer di Bandung Tengah dengan suasana cozy untuk wfc",
    "Kafe murah untuk mahasiswa, banyak colokan, tidak berisik",
]


def _render_recommendations(
    results: pd.DataFrame, user_pref: UserPreference, key_prefix: str = "txt"
) -> None:
    if len(results) == 0:
        st.warning("Tidak ada kafe yang cocok. Coba kurangi filter atau ubah kata kunci.")
        return

    for i, (_, row) in enumerate(results.iterrows()):
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### {i+1}. {row['place_name']}")
                st.markdown(f"📍 **Lokasi:** {row['lokasi']}")

                try:
                    maps_data = search_place(row["place_name"], row["lokasi"])
                except Exception:
                    maps_data = None

                if maps_data:
                    st.caption(f"🗺 {maps_data.get('address', '-')}")
                    st.caption(f"⭐ Google Maps Rating: {maps_data.get('rating', '-')}")
                    try:
                        map_df = get_map_df(maps_data.get("lat"), maps_data.get("lng"))
                        if map_df is not None:
                            st.map(map_df, zoom=15, use_container_width=True)
                    except Exception:
                        pass
                    maps_link = build_maps_link(maps_data.get("lat"), maps_data.get("lng"))
                    if maps_link:
                        st.link_button(
                            "📍 Buka di Google Maps", maps_link,
                            key=f"{key_prefix}_maps_{i}",
                        )

                for reason in explain_match(row, user_pref):
                    st.markdown(f"• {reason}")

            with col_b:
                st.metric("⭐ Match", f"{row['match_score']:.0%}")
                st.metric(
                    "📊 Rating",
                    f"{row['avg_rating']:.2f}/5" if pd.notna(row["avg_rating"]) else "—",
                )
                st.caption(f"💬 {int(row['n_reviews'])} review")

            with st.expander("Lihat detail fitur"):
                feat_data = pd.DataFrame({
                    "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
                    "Skor":  [row.get(f, 0) for f in TARGET_FEATURES],
                })
                fig = px.bar(
                    feat_data, x="Skor", y="Fitur", orientation="h",
                    range_x=[0, 1], color="Skor", color_continuous_scale=TERRA_SCALE,
                )
                fig.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"{key_prefix}_featchart_{i}")


def render_cari(df: pd.DataFrame) -> None:
    section_anchor("cari")
    section_heading("💬", "Cari Kafe")

    st.markdown(
        "Ketik apa yang Anda cari dengan kalimat sehari-hari — sistem akan otomatis "
        "memahami preferensi Anda. Contoh:"
    )

    cols = st.columns(min(3, len(EXAMPLE_QUERIES)))
    for i, ex in enumerate(EXAMPLE_QUERIES[:3]):
        if cols[i].button(
            f"💡 {ex[:50]}{'...' if len(ex) > 50 else ''}",
            key=f"cari_ex_{i}",
            use_container_width=True,
        ):
            st.markdown(
                "<script>if(window._cafeSaveScroll) window._cafeSaveScroll();</script>",
                unsafe_allow_html=True,
            )
            st.session_state["cari_text_query"] = ex
            st.rerun()

    has_gemini = bool(GEMINI_API_KEY)
    col_q, col_m = st.columns([4, 1])
    with col_q:
        query = st.text_area(
            "Deskripsikan kafe yang Anda cari",
            placeholder="Contoh: cafe tenang, ada wifi & stopkontak...",
            height=100,
            key="cari_text_query",
        )
    with col_m:
        st.caption("Mode parsing")
        prefer_llm = st.toggle(
            "Gunakan Gemini AI",
            value=has_gemini,
            disabled=not has_gemini,
            help="Aktifkan untuk parsing lebih cerdas (butuh API key di .env)",
            key="cari_prefer_llm",
        )
        if not has_gemini:
            st.caption("⚠️ API key kosong — pakai keyword saja")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        top_n = st.slider("Jumlah rekomendasi", 3, 20, 8, key="cari_top_n")
    with col_f2:
        min_reviews_override = st.slider(
            "Min jumlah review", 0, 50, 5,
            help="Filter kafe yang punya cukup review (lebih reliable)",
            key="cari_min_reviews",
        )

    submit = st.button("🚀 Cari Kafe", type="primary", use_container_width=True,
                       key="cari_btn_submit")

    if not submit:
        st.info("👆 Ketik kalimat di atas, lalu tekan **Cari Kafe**")
        return
    if not query or not query.strip():
        st.warning("Mohon isi kalimat pencarian dulu.")
        return

    with st.spinner("Memahami preferensi Anda..."):
        user_pref, mode = parse_query(query, prefer_llm=prefer_llm)
        user_pref.min_reviews = max(user_pref.min_reviews, min_reviews_override)

    with st.expander("🧠 Apa yang sistem tangkap dari kalimat Anda", expanded=True):
        st.markdown(
            f"**Mode parsing:** `{mode}` "
            + ("🤖 (LLM)" if mode == "gemini" else "🔍 (keyword)")
        )
        active_features = {f: w for f, w in user_pref.features.items() if w > 0.1}
        if active_features:
            st.markdown("**Fitur yang Anda inginkan:**")
            cols_f = st.columns(min(4, len(active_features)))
            for j, (feat, w) in enumerate(active_features.items()):
                cols_f[j % len(cols_f)].metric(FEATURE_LABELS.get(feat, feat), f"{w:.0%}")
        else:
            st.caption("(tidak terdeteksi fitur spesifik — mencari berdasarkan rating tertinggi)")

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            f"**Lokasi:** {', '.join(user_pref.lokasi) if user_pref.lokasi else '_semua wilayah_'}"
        )
        col_b.markdown(
            f"**Min Rating:** {user_pref.min_rating if user_pref.min_rating else '_tidak dibatasi_'}"
        )
        col_c.markdown(
            f"**Min Review:** {user_pref.min_reviews if user_pref.min_reviews else '_tidak dibatasi_'}"
        )

    with st.spinner("Mencari kafe terbaik..."):
        results = recommend(df, user_pref, top_n=top_n)

    st.markdown("---")
    if len(results) > 0:
        st.success(f"✅ Ditemukan **{len(results)} kafe** yang cocok!")
    _render_recommendations(results, user_pref, key_prefix="cari")


# =============================================================================
# Section: Detail Kafe
# =============================================================================
def render_detail(df: pd.DataFrame) -> None:
    section_anchor("detail")
    section_heading("📋", "Detail Kafe")

    cafe_list = sorted(df["place_name"].dropna().unique().tolist())
    default_idx = 0
    selected_default = st.session_state.get("selected_cafe")
    if selected_default and selected_default in cafe_list:
        default_idx = cafe_list.index(selected_default)

    selected_cafe = st.selectbox("Pilih kafe", cafe_list, index=default_idx,
                                 key="detail_selectbox")
    cafe_row = df[df["place_name"] == selected_cafe].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📍 Lokasi", cafe_row["lokasi"])
    col2.metric("⭐ Rating",
                f"{cafe_row['avg_rating']:.2f}" if pd.notna(cafe_row["avg_rating"]) else "—")
    col3.metric("💬 Total Review", f"{int(cafe_row['n_reviews']):,}")
    col4.metric("😊 % Positif",
                f"{cafe_row['positive_pct']*100:.1f}%"
                if pd.notna(cafe_row["positive_pct"]) else "—")

    st.markdown("---")
    st.subheader("📊 Profil Fasilitas Kafe")
    feat_df = pd.DataFrame({
        "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES],
        "Skor":  [cafe_row.get(f, 0) for f in TARGET_FEATURES],
    })
    fig = px.bar(
        feat_df, x="Fitur", y="Skor",
        text=feat_df["Skor"].apply(lambda x: f"{x:.2f}"),
        color="Skor", color_continuous_scale=TERRA_SCALE, range_y=[0, 1],
    )
    fig.update_layout(height=400, showlegend=False, **PLOTLY_LAYOUT)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💬 Ulasan Pengguna")
    reviews = load_reviews(selected_cafe or "")
    if len(reviews) == 0:
        st.info("Tidak ada ulasan untuk kafe ini.")
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        sentiment_filter = st.radio(
            "Filter sentimen",
            ["Semua", "positive", "negative", "neutral"],
            horizontal=False,
            key="detail_sentiment_radio",
        )
        st.caption(f"Total: {len(reviews)}")
        st.caption(f"Positive: {(reviews['sentiment'] == 'positive').sum()}")
        st.caption(f"Negative: {(reviews['sentiment'] == 'negative').sum()}")
        st.caption(f"Neutral: {(reviews['sentiment'] == 'neutral').sum()}")

    with col_b:
        filtered = (
            reviews if sentiment_filter == "Semua"
            else reviews[reviews["sentiment"] == sentiment_filter]
        )
        SHOW_LIMIT = 30

        for _, r in filtered.head(SHOW_LIMIT).iterrows():
            sent = r["sentiment"]
            emoji_icon = _SENT_EMOJI.get(sent, "•")
            color      = _SENT_COLOR.get(sent, "#888")
            color_name = _SENT_NAME.get(sent, "gray")

            with st.container(border=True):
                c0, c1 = st.columns([3, 1])
                with c0:
                    st.markdown(f"**{emoji_icon} {r['username'] or 'Anonim'}**")
                    st.caption(
                        f"⭐ Rating: {r['rating']}/5 — "
                        f"Sentimen: :{color_name}[{sent}] (skor: {r['sentiment_score']:.2f})"
                    )
                    st.write(r["review_raw"])
                with c1:
                    st.markdown(
                        f"<div style='font-size:24px;color:{color};text-align:center'>"
                        f"{emoji_icon}</div>",
                        unsafe_allow_html=True,
                    )

        if len(filtered) > SHOW_LIMIT:
            st.caption(f"Menampilkan {SHOW_LIMIT} dari {len(filtered)} ulasan")


# =============================================================================
# Section: Analytics
# =============================================================================
def render_analytics(df: pd.DataFrame) -> None:
    section_anchor("analytics")
    section_heading("📊", "Analytics & Insight")

    st.subheader("📈 Rata-rata Skor Fitur Across Cafes")
    avg_features = pd.DataFrame({
        "Fitur": [FEATURE_LABELS.get(f, f) for f in TARGET_FEATURES if f in df.columns],
        "Rata-rata Skor": [df[f].mean() for f in TARGET_FEATURES if f in df.columns],
    })
    fig = px.bar(
        avg_features, x="Fitur", y="Rata-rata Skor",
        text=avg_features["Rata-rata Skor"].apply(lambda x: f"{x:.2f}"),
        color="Rata-rata Skor", color_continuous_scale=TERRA_SCALE, range_y=[0, 1],
    )
    fig.update_layout(showlegend=False, height=400, **PLOTLY_LAYOUT)
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
        x="lokasi", y="avg_rating", text="avg_rating",
        color="avg_rating", color_continuous_scale=TERRA_SCALE,
    )
    fig2.update_traces(texttemplate="%{text:.2f}")
    fig2.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🏆 Top Kafe per Fitur")
    selected_feat = st.selectbox(
        "Pilih fitur", TARGET_FEATURES,
        format_func=lambda f: FEATURE_LABELS.get(f) or str(f),
        key="analytics_feat_select",
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
        color_continuous_scale=CORR_SCALE,
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig3.update_layout(height=500, **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# Section: Tentang
# =============================================================================
PIPELINE_STEPS = [
    ("01", "Preprocessing",
     "Pembersihan teks ulasan: hapus emoji/URL/karakter spesial, normalisasi slang "
     "Bahasa Indonesia, penghapusan stopwords (Sastrawi), deduplikasi."),
    ("02", "Sentiment Analysis",
     "Klasifikasi positif/netral/negatif menggunakan kombinasi lexicon-based scoring "
     "dan rating-based scoring, dengan handling negasi."),
    ("03", "Feature Extraction",
     "Ekstraksi fitur fasilitas (WiFi, stopkontak, kenyamanan, dll) via keyword matching, "
     "dengan opsional enhancement menggunakan Gemini LLM."),
    ("04", "Database",
     "Penyimpanan terstruktur SQLite dengan tiga tabel utama: cafes, reviews, features."),
    ("05", "Recommendation Engine",
     "Content-based recommendation via cosine similarity antara profil fitur kafe dan "
     "preferensi pengguna, dengan bobot tambahan dari rating & sentimen."),
    ("06", "Web App",
     "Streamlit sebagai UI interaktif dengan tema vintage retro cafe, single-page scroll, "
     "dan visualisasi Plotly."),
]

TEAM_MEMBERS = [
    ("Zahra Nur Azizah",          "Data Scraping & NLP"),
    ("Winston Lokeswara Mangori", "NLP & LLM"),
    ("Yan Andhinaya Ardika",      "Recommendation System & Database"),
    ("Yohana Lydia",              "Backend & API"),
    ("Yayang Ananda Setya",       "Frontend & UI"),
]


def render_tentang() -> None:
    section_anchor("tentang")
    section_heading("ℹ️", "Tentang Sistem Ini")

    st.markdown(
        "Sistem rekomendasi kafe di Bandung sebagai **tempat belajar**, berbasis ulasan "
        "Google Maps, dengan pendekatan **NLP** dan **Large Language Model**."
    )

    st.markdown("---")

    st.subheader("🏗️ Arsitektur Pipeline")
    col_left, col_right = st.columns(2)
    for idx, (step, title, body) in enumerate(PIPELINE_STEPS):
        target_col = col_left if idx % 2 == 0 else col_right
        with target_col:
            st.markdown(
                f'<div class="pipeline-card">'
                f'<span class="pc-step">{step}</span>'
                f'<span class="pc-title">{title}</span>'
                f'<div class="pc-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("👥 Tim Kelompok 30")
    team_cols = st.columns(len(TEAM_MEMBERS))
    for col, (name, role) in zip(team_cols, TEAM_MEMBERS):
        with col:
            st.markdown(
                f'<div class="team-card">'
                f'<div class="tc-name">{name}</div>'
                f'<div class="tc-role">{role}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # 1. CSS theme — selalu di-inject di setiap render
    st.markdown(VINTAGE_CSS, unsafe_allow_html=True)

    # 2. Persistent JS via components.html
    components.html(SCROLL_JS, height=0, scrolling=False)

    # 3. Cek apakah user sudah melewati landing page
    if not st.session_state.get("landing_done", False):
        render_landing()
        return  # Berhenti di sini — sidebar & dashboard tidak ditampilkan

    # 4. Sidebar (hanya muncul setelah landing)
    render_sidebar()

    # 5. Load data
    df = load_features()
    stats = get_stats()

    # 6. Render semua section sekaligus — single-page scroll
    render_beranda(df, stats)

    if len(df) == 0:
        st.error("Database kosong. Jalankan `python scripts/run_all.py` terlebih dahulu.")
        render_tentang()
        return

    section_spacer()
    render_cari(df)

    section_spacer()
    render_detail(df)

    section_spacer()
    render_analytics(df)

    section_spacer()
    render_tentang()


if __name__ == "__main__":
    main()
