import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agricultural Sustainability Predictor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at top left, #1a2e0f 0%, #0f1a08 40%, #0b1520 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 740px; }

/* ── Hero ── */
.hero { text-align: center; padding: 2rem 1rem 0.5rem; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(163,230,53,0.1);
    border: 1px solid rgba(163,230,53,0.28);
    color: #a3e635;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.28rem 0.85rem; border-radius: 99px;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem; font-weight: 400;
    color: #f7fee7; line-height: 1.12;
    margin-bottom: 0.7rem;
}
.hero-title em { font-style: normal; color: #a3e635; }
.hero-sub {
    color: #6b7f5a; font-size: 0.95rem; line-height: 1.65;
    max-width: 460px; margin: 0 auto 2rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #4a5c38; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.45rem;
}
.section-label::after {
    content: ''; flex: 1;
    height: 1px; background: rgba(163,230,53,0.12);
}

/* ── Input panel ── */
.input-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(163,230,53,0.1);
    border-radius: 18px; padding: 1.6rem 1.8rem 1rem;
    margin-bottom: 1.1rem;
}

/* ── Slider overrides ── */
.stSlider label { color: #a8bb90 !important; font-size: 0.88rem !important; font-weight: 500 !important; }
div[data-testid="stSlider"] > div > div > div { background: rgba(163,230,53,0.18) !important; }

/* ── Value grid ── */
.val-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 0.6rem; margin-bottom: 1.3rem; }
.val-cell {
    background: rgba(163,230,53,0.05);
    border: 1px solid rgba(163,230,53,0.1);
    border-radius: 10px; padding: 0.65rem 0.5rem; text-align: center;
}
.val-cell .lbl { font-size: 0.58rem; color: #4a5c38; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }
.val-cell .num { font-size: 1.05rem; font-weight: 700; color: #d9f99d; margin-top: 0.18rem; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4d7c0f, #a3e635);
    color: #0f1a08; font-weight: 800; font-size: 1rem;
    letter-spacing: 0.04em; border: none;
    border-radius: 12px; padding: 0.9rem 0;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 24px rgba(163,230,53,0.18);
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-2px); }

/* ── Result cards ── */
.res-card {
    border-radius: 18px; padding: 2.2rem 1.5rem 1.8rem;
    text-align: center; margin-bottom: 1rem;
}
.res-card.sustain {
    background: linear-gradient(135deg, rgba(163,230,53,0.12), rgba(74,124,15,0.08));
    border: 1px solid rgba(163,230,53,0.35);
}
.res-card.unsustain {
    background: linear-gradient(135deg, rgba(251,146,60,0.12), rgba(194,65,12,0.08));
    border: 1px solid rgba(251,146,60,0.35);
}
.res-icon { font-size: 3.2rem; margin-bottom: 0.6rem; }
.res-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem; margin-bottom: 0.3rem;
}
.res-verdict.s { color: #a3e635; }
.res-verdict.u { color: #fb923c; }
.res-conf { font-size: 0.85rem; color: #6b7f5a; margin-bottom: 1rem; }
.res-conf strong { color: #a8bb90; }

/* ── Confidence bar ── */
.bar-wrap { max-width: 280px; margin: 0 auto; }
.bar-bg { background: rgba(255,255,255,0.07); border-radius: 6px; height: 7px; overflow: hidden; }
.bar-fill-s { background: linear-gradient(90deg, #4d7c0f, #a3e635); height: 7px; border-radius: 6px; }
.bar-fill-u { background: linear-gradient(90deg, #c2410c, #fb923c); height: 7px; border-radius: 6px; }

/* ── Probability table ── */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div[data-testid="stDataFrame"] table { background: transparent !important; }

/* ── Feature importance bar ── */
.fi-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.55rem; }
.fi-name { color: #a8bb90; font-size: 0.8rem; font-weight: 500; width: 130px; flex-shrink: 0; }
.fi-bar-bg { flex: 1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; overflow: hidden; }
.fi-bar-fill { background: linear-gradient(90deg, #4d7c0f, #a3e635); height: 8px; border-radius: 4px; }
.fi-pct { color: #4a5c38; font-size: 0.75rem; width: 38px; text-align: right; flex-shrink: 0; }

/* ── Divider ── */
.hr { border: none; border-top: 1px solid rgba(163,230,53,0.08); margin: 1.5rem 0; }

/* ── Footer ── */
.footer { text-align: center; color: #2d3f1f; font-size: 0.75rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("agricultural_sustainability_model_balanced.pkl")

model = load_model()

FEATURES = ['soil_health', 'crop_yield', 'water_usage', 'carbon_footprint', 'fertilizer_use']

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🌾 Random Forest · SMOTE-Balanced</div>
    <div class="hero-title">Agricultural<br><em>Sustainability</em> Predictor</div>
    <div class="hero-sub">
        Enter your farm's key metrics below to instantly predict whether
        your agricultural practices are on a sustainable path.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-label">⚙ Farm Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    soil_health = st.slider(
        "Soil Health", 0.0, 100.0, 65.0, 0.5,
        help="Overall soil quality score (0 = degraded, 100 = excellent)"
    )
    crop_yield = st.slider(
        "Crop Yield", 0.0, 100.0, 60.0, 0.5,
        help="Normalised crop yield output (0–100)"
    )
    water_usage = st.slider(
        "Water Usage", 0.0, 100.0, 45.0, 0.5,
        help="Water consumption level (lower = more efficient)"
    )

with col2:
    carbon_footprint = st.slider(
        "Carbon Footprint", 0.0, 100.0, 40.0, 0.5,
        help="Carbon emissions from farm operations (lower = better)"
    )
    fertilizer_use = st.slider(
        "Fertilizer Use", 0.0, 100.0, 35.0, 0.5,
        help="Amount of fertilizer applied (lower = more sustainable)"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Live value grid ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="val-grid">
    <div class="val-cell"><div class="lbl">Soil Health</div><div class="num">{soil_health:.1f}</div></div>
    <div class="val-cell"><div class="lbl">Crop Yield</div><div class="num">{crop_yield:.1f}</div></div>
    <div class="val-cell"><div class="lbl">Water Usage</div><div class="num">{water_usage:.1f}</div></div>
    <div class="val-cell"><div class="lbl">Carbon Footprint</div><div class="num">{carbon_footprint:.1f}</div></div>
    <div class="val-cell"><div class="lbl">Fertilizer Use</div><div class="num">{fertilizer_use:.1f}</div></div>
</div>
""", unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────
if st.button("Predict Sustainability →"):
    input_data = np.array([[soil_health, crop_yield, water_usage, carbon_footprint, fertilizer_use]])
    prediction = model.predict(input_data)[0]
    proba      = model.predict_proba(input_data)[0]
    confidence = proba[prediction] * 100
    bar_pct    = f"{confidence:.0f}%"

    # ── Result card ──
    if prediction == 1:
        st.markdown(f"""
        <div class="res-card sustain">
            <div class="res-icon">🌿</div>
            <div class="res-verdict s">Sustainable</div>
            <div class="res-conf">Model confidence: <strong>{confidence:.1f}%</strong></div>
            <div class="bar-wrap">
                <div class="bar-bg"><div class="bar-fill-s" style="width:{bar_pct}"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Great news — your farming practices indicate a **sustainable** operation!", icon="✅")
    else:
        st.markdown(f"""
        <div class="res-card unsustain">
            <div class="res-icon">⚠️</div>
            <div class="res-verdict u">Unsustainable</div>
            <div class="res-conf">Model confidence: <strong>{confidence:.1f}%</strong></div>
            <div class="bar-wrap">
                <div class="bar-bg"><div class="bar-fill-u" style="width:{bar_pct}"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("Current metrics suggest your practices are **not yet sustainable**. Consider reducing water usage, carbon footprint, or fertilizer application.", icon="📉")

    # ── Probability breakdown ──
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Class Probabilities</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Class":            ["Unsustainable", "Sustainable"],
        "Probability (%)":  [round(proba[0]*100, 2), round(proba[1]*100, 2)],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # ── Feature importances ──
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🌱 Feature Importances</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    max_imp = fi_pairs[0][1]

    fi_html = ""
    labels = {
        'soil_health':      'Soil Health',
        'crop_yield':       'Crop Yield',
        'water_usage':      'Water Usage',
        'carbon_footprint': 'Carbon Footprint',
        'fertilizer_use':   'Fertilizer Use',
    }
    for feat, imp in fi_pairs:
        pct = imp / max_imp * 100
        fi_html += f"""
        <div class="fi-row">
            <div class="fi-name">{labels[feat]}</div>
            <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="fi-pct">{imp*100:.1f}%</div>
        </div>"""

    st.markdown(fi_html, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="hr">
<div class="footer">Random Forest Classifier · SMOTE Balanced &nbsp;|&nbsp; Agricultural Sustainability Dataset</div>
""", unsafe_allow_html=True)
