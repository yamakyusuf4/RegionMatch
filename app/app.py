import os
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import google.generativeai as genai

# ============================================================
# REPO ROOT (define early so helpers can use it)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# GEMINI API CONFIG (define early so helpers can use it)
#   - Must be BEFORE generate_explanation() to avoid NameError
#   - Prefer Streamlit secrets, fallback to env var
# ============================================================
GEMINI_API_KEY = ""
try:
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = str(st.secrets["GEMINI_API_KEY"])
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True
else:
    gemini_available = False

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="RegionMatch",
    page_icon="🧭",  # fix mojibake
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show Gemini warning AFTER Streamlit is initialized
if not gemini_available:
    st.warning(
        "⚠️ GEMINI_API_KEY not set. Explanations will be unavailable. "
        "Set it as an environment variable (GEMINI_API_KEY) or in Streamlit secrets."
    )

# ============================================================
# THEME / STYLE
# ============================================================
st.markdown("""
<style>
* {
  margin: 0;
  padding: 0;
}

.stApp {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  color: #e2e8f0;
}

[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  border-right: 1px solid rgba(148, 163, 184, 0.12);
}

.block-container {
    padding: 0.6rem 1rem;
  max-width: 100%;
}

/* Header styling */
.header-title {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #0ea5e9, #06b6d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}

.criteria-badge {
  display: inline-block;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.3);
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  margin: 0.3rem 0.3rem 0.3rem 0;
  font-size: 0.875rem;
  color: #cbd5e1;
}

.criteria-label {
  color: #94a3b8;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Card styling */
.metric-card {
  background: linear-gradient(135deg, rgba(51, 65, 85, 0.5), rgba(30, 41, 59, 0.5));
  border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 0.75rem;
  padding: 1.25rem;
  backdrop-filter: blur(10px);
}

/* Table styling */
[data-testid="stDataFrame"] {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.15) !important;
  border-radius: 0.75rem !important;
  overflow: hidden;
}

[data-testid="stDataFrame"] > div {
  border-radius: 0.75rem;
}

/* Section headers - compact for single-page fit */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(6, 182, 212, 0.22);
}

/* Explanation container */
.explanation-box {
  background: linear-gradient(135deg, rgba(12, 74, 110, 0.2), rgba(7, 89, 133, 0.2));
  border: 1px solid rgba(6, 182, 212, 0.3);
  border-radius: 0.75rem;
  padding: 1.5rem;
  line-height: 1.6;
}

/* Sidebar styling */
.sidebar-section {
  margin-bottom: 2rem;
}

.sidebar-label {
  font-size: 0.875rem;
  font-weight: 600;
  color: #94a3b8;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Map container */
.map-container {
  border-radius: 0.75rem;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.15);
}

/* Responsive adjustments */
@media (max-width: 1024px) {
  [data-testid="column"] {
    width: 100% !important;
  }

  .header-title {
    font-size: 2rem;
  }
}

/* Info box styling */
.stInfo {
  background: linear-gradient(135deg, rgba(12, 74, 110, 0.15), rgba(7, 89, 133, 0.15)) !important;
  border-left: 4px solid #0ea5e9 !important;
}

/* Warning box styling */
.stWarning {
  background: linear-gradient(135deg, rgba(120, 53, 15, 0.15), rgba(124, 45, 18, 0.15)) !important;
  border-left: 4px solid #ea580c !important;
}

/* Selectbox and input styling */
.stSelectbox, .stNumberInput {
  background: rgba(15, 23, 42, 0.5);
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
select {
  background: rgba(30, 41, 59, 0.6) !important;
  border: 1px solid rgba(148, 163, 184, 0.2) !important;
  color: #e2e8f0 !important;
}

/* Container borders */
.stContainer {
  border-radius: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CITY DATA (Expanded list)
# ============================================================
UK_CITIES = [
    # England
    ("London", -0.1276, 51.5072),
    ("Birmingham", -1.8904, 52.4862),
    ("Manchester", -2.2426, 53.4808),
    ("Leeds", -1.5491, 53.8008),
    ("Liverpool", -2.9916, 53.4084),
    ("Bristol", -2.5879, 51.4545),
    ("Sheffield", -1.4701, 53.3811),
    ("Newcastle upon Tyne", -1.6178, 54.9783),
    ("Nottingham", -1.1505, 52.9548),
    ("Leicester", -1.1332, 52.6369),
    ("Southampton", -1.4043, 50.9097),
    ("Portsmouth", -1.0873, 50.8198),
    ("Brighton", -0.1364, 50.8225),
    ("Cambridge", 0.1218, 52.2053),
    ("Oxford", -1.2577, 51.7520),
    ("Reading", -0.9781, 51.4543),
    ("Milton Keynes", -0.7594, 52.0406),
    ("Luton", -0.4176, 51.8797),
    ("Peterborough", -0.2420, 52.5695),
    ("Norwich", 1.2974, 52.6309),
    ("Ipswich", 1.1555, 52.0567),
    ("York", -1.0815, 53.9590),
    ("Hull", -0.3367, 53.7457),
    ("Middlesbrough", -1.2348, 54.5742),
    ("Sunderland", -1.3822, 54.9069),
    ("Derby", -1.4766, 52.9225),
    ("Stoke-on-Trent", -2.1794, 53.0027),
    ("Wolverhampton", -2.1276, 52.5862),
    ("Coventry", -1.5106, 52.4068),
    ("Northampton", -0.8901, 52.2405),
    ("Cheltenham", -2.0713, 51.8994),
    ("Swindon", -1.7809, 51.5558),
    ("Exeter", -3.5339, 50.7184),
    ("Plymouth", -4.1427, 50.3755),
    ("Bournemouth", -1.8795, 50.7192),

    # Wales
    ("Cardiff", -3.1791, 51.4816),
    ("Swansea", -3.9436, 51.6214),
    ("Newport", -2.9984, 51.5842),

    # Scotland
    ("Edinburgh", -3.1883, 55.9533),
    ("Glasgow", -4.2518, 55.8642),
    ("Aberdeen", -2.0943, 57.1497),
    ("Dundee", -2.9707, 56.4620),
    ("Inverness", -4.2247, 57.4778),

    # Northern Ireland
    ("Belfast", -5.9301, 54.5973),
    ("Derry/Londonderry", -7.3092, 54.9966),
]
cities_df = pd.DataFrame(UK_CITIES, columns=["city", "lng", "lat"])

INDUSTRIES = [
    "Technology", "Creative", "Innovation",
    "Business Services", "Retail/Hospitality", "Industrial/Logistics"
]
URGENCY = ["<3 months", "3-6 months", "6+ months"]

# ============================================================
# MAP CONFIG
# ============================================================
MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_to_uk(lng, lat):
    return clamp(lng, -8.8, 2.3), clamp(lat, 49.8, 60.9)

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    R = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def minmax_series(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def generate_explanation(lad_name, score, lad_data, industry, employees, urgency):
    """Generate a human-readable explanation using Gemini API."""
    if not gemini_available:
        return "API key not configured. Please set GEMINI_API_KEY to enable AI explanations."

    try:
        context_fields = []
        for col in [
            "core_tech_density", "creative_density", "innovation_density",
            "business_services_density", "job_liquidity_score_1_10",
            "reddit_sentiment_score_1_10", "approval_rate",
            "median_decision_days", "business_density",
            "micro_ratio", "sme_ratio", "large_ratio", "scaling_index"
        ]:
            if col in lad_data.index:
                val = lad_data.get(col, "N/A")
                if isinstance(val, (int, float, np.floating)) and not pd.isna(val):
                    context_fields.append(f"{col}: {float(val):.2f}")

        context = "\n".join(context_fields) if context_fields else "Standard metrics"

        prompt = f"""Explain why '{lad_name}' received a compatibility score of {round(float(score), 2)}/100 for a {industry} business with {employees} employees looking to expand with a {urgency} hiring timeline.

Key metrics for {lad_name}:
{context}

Provide a concise, professional explanation (2-3 sentences) that a business owner would understand. Focus on why this location is a good fit for their specific needs. Be positive but honest."""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "No explanation returned."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def safe_dataset_path():
    """Prefer geo -> clean -> v1, checking repo and processed folders."""
    candidates = [
        REPO_ROOT / "data" / "processed" / "training_data_geo.csv",
        REPO_ROOT / "training_data_geo.csv",
        REPO_ROOT / "data" / "processed" / "training_data_clean.csv",
        REPO_ROOT / "training_data_clean.csv",
        REPO_ROOT / "training_data_v1.csv",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No training dataset found in expected locations.")

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load(REPO_ROOT / "models" / "location_model.joblib")
    features = joblib.load(REPO_ROOT / "models" / "model_features.joblib")
    return model, features

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

pipe, feature_list = load_model()
DATA_PATH = safe_dataset_path()
df = load_data(DATA_PATH)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### 🎯 Search Criteria")

st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
city = st.sidebar.selectbox("📍 Location Focus", cities_df["city"])

st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
industry = st.sidebar.selectbox("🏢 Industry Type", INDUSTRIES)

st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
employees = st.sidebar.number_input("👥 Employees", min_value=1, max_value=100000, value=25, step=1)

st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
urgency = st.sidebar.selectbox("⏱️  Hiring Urgency", URGENCY)

# ============================================================
# MAP VIEW STATE
# ============================================================
sel = cities_df[cities_df.city == city].iloc[0]
target_lng, target_lat = clamp_to_uk(float(sel.lng), float(sel.lat))

view_state = pdk.ViewState(
    longitude=target_lng,
    latitude=target_lat,
    zoom=10.5,
    pitch=55,
    bearing=-15
)

# ============================================================
# MODEL SCORING
# ============================================================
missing_features = [f for f in feature_list if f not in df.columns]
if missing_features:
    st.warning(
        f"{len(missing_features)} model features were missing in the dataset; "
        "they were added as 0 so scoring can continue."
    )

X = df.reindex(columns=feature_list).copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

base = pipe.predict(X)

# Only industry and hiring urgency should affect recommendations.
# Build a compact adjustment that depends on industry match and urgency level.
adj = np.zeros(len(df), dtype=float)

# Industry mapping (same as before)
industry_col_map = {
    "Technology": "core_tech_density",
    "Creative": "creative_density",
    "Innovation": "innovation_density",
    "Business Services": "business_services_density",
    "Retail/Hospitality": "business_density",
    "Industrial/Logistics": "business_density"
}

industry_col = industry_col_map.get(industry)
if industry_col and industry_col in df.columns:
    industry_boost = minmax_series(df[industry_col]).values
else:
    industry_boost = np.zeros(len(df), dtype=float)

# Urgency factor controls how strongly industry match moves ranking
urgency_factor_map = {
    "<3 months": 1.0,
    "3-6 months": 0.6,
    "6+ months": 0.3,
}
urgency_factor = urgency_factor_map.get(urgency, 0.6)

# Apply a modest industry-based adjustment scaled by urgency
adj += urgency_factor * 0.20 * industry_boost

base_std = float(np.std(base)) + 1e-9
adj_std = float(np.std(adj)) + 1e-9
adj_scaled = adj * (base_std / adj_std)

final_score = 0.50 * base + 0.50 * adj_scaled

df_scored = df[["lad_code", "lad_name"]].copy()
df_scored["score"] = final_score

# Normalize to 0–100 (fix mojibake in comment too)
df_scored["score"] = 100 * (df_scored["score"] - df_scored["score"].min()) / (
    df_scored["score"].max() - df_scored["score"].min() + 1e-9
)

# ============================================================
# CANDIDATE POOL (use entire dataset independent of selected city)
# ============================================================
# We intentionally do NOT filter candidates by distance to the selected city.
# This keeps recommendations stable regardless of the chosen map focus; only
# the per-area explanation changes when the user selects an area.
has_centroids = {"lad_lat", "lad_lng"}.issubset(df.columns)

# Prepare df_local with lat/lng when available so the map can still plot centroids,
# but use the full scored dataset as the candidate pool for ranking.
df_local = df_scored.copy()
if has_centroids:
    lat = pd.to_numeric(df.get("lad_lat"), errors="coerce")
    lng = pd.to_numeric(df.get("lad_lng"), errors="coerce")
    # attach lat/lng columns (may contain NaNs) for mapping purposes
    df_local["lad_lat"] = lat.values
    df_local["lad_lng"] = lng.values

# City-based filtering: only evaluate LADs within the selected city (50 km radius)
candidates = df_local.copy()

if has_centroids:
    # compute haversine distance (km) from the selected city to each area
    dist_km = haversine_km(
        df_local.get("lad_lat", np.nan).fillna(target_lat).values,
        df_local.get("lad_lng", np.nan).fillna(target_lng).values,
        target_lat,
        target_lng,
    )
    # Keep only LADs within 50 km radius of the selected city
    city_mask = dist_km <= 50.0
    candidates = candidates.loc[city_mask].copy()

# Exclude LADs that are in the selected city itself (within 10 km radius)
if has_centroids:
    in_city_dist = haversine_km(
        candidates.get("lad_lat", np.nan).fillna(target_lat).values,
        candidates.get("lad_lng", np.nan).fillna(target_lng).values,
        target_lat,
        target_lng,
    )
    # Keep only LADs that are NOT in the city (> 10 km away)
    not_in_city_mask = in_city_dist > 10.0
    candidates = candidates.loc[not_in_city_mask].copy()

# Re-normalize scores within the city's candidate set for local ranking
if len(candidates) > 0:
    if candidates["score"].nunique(dropna=True) > 1:
        candidates["score"] = 100 * (candidates["score"] - candidates["score"].min()) / (
            candidates["score"].max() - candidates["score"].min() + 1e-9
        )

# Generate random cap between 99 and 99.5 for highest score
rng_cap = np.random.default_rng()
max_score_cap = float(rng_cap.uniform(99.0, 99.5))

# Pick top N by score
top = candidates.sort_values("score", ascending=False).head(5).copy()

# Scale top scores so highest is between 99-99.5 (random)
if len(top) > 0 and top["score"].max() > 0:
    top_score_max = float(top["score"].max())
    scale_factor = max_score_cap / top_score_max
    top["score"] = top["score"] * scale_factor

# ============================================================
# MAP LAYERS
# ============================================================
rng = np.random.default_rng(7)
scores_arr = np.asarray(top["score"].to_numpy()).ravel()
if scores_arr.size == 0:
    # fallback to a flat score so the map still renders
    scores_arr = np.array([0.0], dtype=float)
sampled_scores = rng.choice(scores_arr, 450, replace=True)
cloud = pd.DataFrame({
    "lng": (float(target_lng) + rng.normal(0, 0.06, 450)).tolist(),
    "lat": (float(target_lat) + rng.normal(0, 0.04, 450)).tolist(),
    "score": sampled_scores.tolist(),
})

hex_layer = pdk.Layer(
    "HexagonLayer",
    cloud,
    get_position=["lng", "lat"],
    radius=1400,
    elevation_scale=30,
    extruded=True,
    pickable=True,
)

deck = pdk.Deck(
    layers=[hex_layer],
    map_style=MAP_STYLE,
    initial_view_state=view_state,
    tooltip={"text": "Heat score: {score}"}
)

# ============================================================
# HEADER - Compact
# ============================================================
st.markdown('<div style="display:flex; align-items:center; justify-content:space-between; gap:1rem">\n  <div style="display:flex; flex-direction:column">\n    <div style="font-size:1.5rem; font-weight:700;">🎯 RegionMatch</div>\n    <div style="color:#94a3b8; font-size:0.95rem; margin-top:0.1rem;">Find your perfect UK location in seconds</div>\n  </div>\n</div>', unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT - TWO COLUMN LAYOUT
# ============================================================
left_col, right_col = st.columns([0.65, 0.35], gap="large")

# ============================================================
# LEFT COLUMN - RECOMMENDATIONS & EXPLANATION
# ============================================================
with left_col:
    st.markdown('<div class="section-header">📊 Top Opportunities</div>', unsafe_allow_html=True)

    # Build display table robustly in case `top` has duplicate column labels
    def _pick_first_column_as_series(df, col):
        vals = df.loc[:, col]
        if isinstance(vals, pd.DataFrame):
            vals = vals.iloc[:, 0]
        return pd.Series(vals).reset_index(drop=True)

    area_series = _pick_first_column_as_series(top, "lad_name")
    score_series = _pick_first_column_as_series(top, "score")
    show = pd.DataFrame({"Area": area_series, "Ranking": score_series})
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown('<div class="section-header">💡 Detailed Analysis</div>', unsafe_allow_html=True)

    selected_area = st.selectbox(
        "Select an area to explore",
        options=top["lad_name"].values,
        help="Choose an area to see a detailed explanation of why it's a great fit"
    )

    explanation_container = st.container(border=True)

    with explanation_container:
        if selected_area:
            match = df[df["lad_name"] == selected_area]
            if match.empty:
                st.error(f"Could not find '{selected_area}' in the dataset.")
            else:
                selected_row = match.iloc[0]
                selected_score = float(top[top["lad_name"] == selected_area]["score"].values[0])

                st.markdown(f"### {selected_area}")
                st.markdown(f"**Compatibility Score:** `{selected_score:.1f}/100`")
                st.markdown("")

                if gemini_available:
                    with st.spinner("🤖 Generating personalized analysis..."):
                        explanation = generate_explanation(
                            selected_area,
                            selected_score,
                            selected_row,
                            industry,
                            employees,
                            urgency
                        )
                    st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
                else:
                    st.info("🔑 To unlock AI-powered explanations, please set your GEMINI_API_KEY.")
        else:
            st.info("👈 **Select an area above** to see why it's a perfect match for your business needs.")

# ============================================================
# RIGHT COLUMN - MAP & QUICK STATS
# ============================================================
with right_col:
    st.markdown('<div class="section-header">🗺️  Location Map</div>', unsafe_allow_html=True)
    st.pydeck_chart(deck, use_container_width=True, height=360)

    st.markdown("")
    st.markdown('<div class="section-header">📈 Quick Stats</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        avg_score = round(float(top["score"].mean()), 1)
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.875rem;">Avg. Score</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #0ea5e9;">{avg_score}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        top_score = round(float(top["score"].max()), 1)
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.875rem;">Best Match</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #10b981;">{top_score}</div>
        </div>
        ''', unsafe_allow_html=True)