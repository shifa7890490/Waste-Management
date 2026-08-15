import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Management Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0d1117 0%, #111827 55%, #0a1a12 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 740px; }

/* ── Hero ── */
.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.28rem 0.9rem; border-radius: 99px;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.7rem; font-weight: 800;
    color: #f9fafb; line-height: 1.1; margin-bottom: 0.7rem;
}
.hero-title .accent { color: #10b981; }
.hero-sub {
    color: #6b7280; font-size: 0.95rem; line-height: 1.65;
    max-width: 460px; margin: 0 auto 0.5rem;
}

/* ── Two-column feature grid ── */
.feat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.75rem; margin-bottom: 1.2rem;
}
.feat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.1rem 1.2rem 0.5rem;
}
.feat-card-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #374151; margin-bottom: 0.5rem;
}

/* ── Slider tweaks ── */
.stSlider label {
    color: #9ca3af !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
}

/* ── Live value strip ── */
.val-strip {
    display: flex; gap: 0.6rem; flex-wrap: wrap;
    margin-bottom: 1.3rem;
}
.val-chip {
    flex: 1; min-width: 120px;
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.14);
    border-radius: 10px; padding: 0.6rem 0.9rem;
    display: flex; justify-content: space-between; align-items: center;
}
.val-chip .vc-label { font-size: 0.67rem; color: #4b5563; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.val-chip .vc-num   { font-size: 1rem; font-weight: 700; color: #6ee7b7; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #065f46, #10b981);
    color: #f0fdf4; font-weight: 800; font-size: 1.02rem;
    letter-spacing: 0.04em; border: none;
    border-radius: 12px; padding: 0.9rem 0;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 28px rgba(16,185,129,0.2);
}
.stButton > button:hover { opacity: 0.87; transform: translateY(-2px); }

/* ── Result cards ── */
.res-card {
    border-radius: 18px; padding: 2.2rem 1.5rem 1.8rem;
    text-align: center; margin-bottom: 1rem;
}
.res-recyclable {
    background: linear-gradient(135deg, rgba(16,185,129,0.13), rgba(6,95,70,0.07));
    border: 1px solid rgba(16,185,129,0.38);
}
.res-nonrecyclable {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(127,29,29,0.07));
    border: 1px solid rgba(239,68,68,0.35);
}
.res-icon { font-size: 3.5rem; margin-bottom: 0.55rem; line-height: 1; }
.res-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem;
}
.rv-green { color: #10b981; }
.rv-red   { color: #ef4444; }
.res-sub  { font-size: 0.88rem; color: #6b7280; margin-bottom: 1.1rem; line-height: 1.5; }

/* ── Decision score bar ── */
.score-wrap { max-width: 300px; margin: 0 auto 0.4rem; }
.score-label { font-size: 0.7rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; margin-bottom: 0.4rem; }
.score-bg { background: rgba(255,255,255,0.07); border-radius: 6px; height: 8px; overflow: hidden; }
.score-fill-g { background: linear-gradient(90deg, #065f46, #10b981); height: 8px; border-radius: 6px; }
.score-fill-r { background: linear-gradient(90deg, #7f1d1d, #ef4444); height: 8px; border-radius: 6px; }
.score-val { font-size: 0.78rem; color: #6b7280; margin-top: 0.35rem; }

/* ── Info table ── */
.info-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.6rem; margin-top: 1rem;
}
.info-cell {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 0.75rem 1rem;
}
.info-cell .ic-lbl { font-size: 0.65rem; color: #374151; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700; }
.info-cell .ic-val { font-size: 0.95rem; font-weight: 600; color: #d1fae5; margin-top: 0.2rem; }

/* ── Section label ── */
.sec-lbl {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #374151; margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 0.45rem;
}
.sec-lbl::after { content: ''; flex: 1; height: 1px; background: rgba(16,185,129,0.1); }

/* ── Tip card ── */
.tip-card {
    background: rgba(16,185,129,0.05);
    border: 1px solid rgba(16,185,129,0.13);
    border-radius: 12px; padding: 1rem 1.1rem;
    font-size: 0.85rem; color: #6b7280; line-height: 1.6;
    margin-top: 0.8rem;
}
.tip-card strong { color: #10b981; }

/* ── Divider ── */
.hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }
.footer { text-align: center; color: #1f2937; font-size: 0.75rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("waste_management_model.pkl")

model = load_model()


# ── Helper: map slider values to readable labels ──────────────────────────────
WASTE_TYPE_MAP = {0: "Organic", 1: "Paper", 2: "Plastic", 3: "Metal", 4: "Glass", 5: "Electronic"}
MATERIAL_MAP   = {0: "Single-material", 1: "Composite", 2: "Mixed", 3: "Contaminated", 4: "Pure"}
TOXICITY_MAP   = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Hazardous"}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">♻️ Support Vector Classifier · RBF Kernel</div>
    <div class="hero-title">Waste<br><span class="accent">Management</span> Classifier</div>
    <div class="hero-sub">
        Describe your waste sample using the parameters below.
        The model will instantly classify it as Recyclable or Non-Recyclable.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input sliders ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">⚙ Waste Sample Parameters</div>', unsafe_allow_html=True)

st.markdown('<div class="feat-grid">', unsafe_allow_html=True)

# Left column
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feat-card"><div class="feat-card-label">🗑 Waste Type</div>', unsafe_allow_html=True)
    waste_type = st.slider("Waste Type", 0, 5, 1, 1,
        help="0=Organic · 1=Paper · 2=Plastic · 3=Metal · 4=Glass · 5=Electronic")
    st.markdown(f'<div style="font-size:0.78rem;color:#10b981;margin-bottom:0.4rem;">→ {WASTE_TYPE_MAP[waste_type]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="feat-card" style="margin-top:0.75rem"><div class="feat-card-label">⚗️ Material Composition</div>', unsafe_allow_html=True)
    material_composition = st.slider("Material Composition", 0, 4, 1, 1,
        help="0=Single-material · 1=Composite · 2=Mixed · 3=Contaminated · 4=Pure")
    st.markdown(f'<div style="font-size:0.78rem;color:#10b981;margin-bottom:0.4rem;">→ {MATERIAL_MAP[material_composition]}</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feat-card"><div class="feat-card-label">♻️ Recycling Potential</div>', unsafe_allow_html=True)
    recycling_potential = st.slider("Recycling Potential", 0.0, 100.0, 65.0, 0.5,
        help="Score 0–100: how recyclable the material is (higher = better)")
    st.markdown(f'<div style="font-size:0.78rem;color:#10b981;margin-bottom:0.4rem;">→ {recycling_potential:.1f} / 100</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="feat-card" style="margin-top:0.75rem"><div class="feat-card-label">☠️ Toxicity Level</div>', unsafe_allow_html=True)
    toxicity_level = st.slider("Toxicity Level", 0, 4, 1, 1,
        help="0=None · 1=Low · 2=Moderate · 3=High · 4=Hazardous")
    st.markdown(f'<div style="font-size:0.78rem;color:#10b981;margin-bottom:0.4rem;">→ {TOXICITY_MAP[toxicity_level]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Live value strip ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="val-strip">
    <div class="val-chip">
        <span class="vc-label">Waste Type</span>
        <span class="vc-num">{WASTE_TYPE_MAP[waste_type]}</span>
    </div>
    <div class="val-chip">
        <span class="vc-label">Composition</span>
        <span class="vc-num">{MATERIAL_MAP[material_composition]}</span>
    </div>
    <div class="val-chip">
        <span class="vc-label">Recycle Potential</span>
        <span class="vc-num">{recycling_potential:.1f}</span>
    </div>
    <div class="val-chip">
        <span class="vc-label">Toxicity</span>
        <span class="vc-num">{TOXICITY_MAP[toxicity_level]}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Classify Waste Sample →"):
    input_df = pd.DataFrame([[waste_type, material_composition, recycling_potential, toxicity_level]],
                             columns=['waste_type', 'material_composition', 'recycling_potential', 'toxicity_level'])

    prediction    = model.predict(input_df)[0]
    decision_val  = model.decision_function(input_df)[0]

    # Normalise decision score to 0–100 for display
    # Clamp to [-5, 5] range then map to percentage
    clamped   = max(-5.0, min(5.0, float(decision_val)))
    norm_pct  = (clamped + 5) / 10 * 100          # 0 = strong Non-Recyclable, 100 = strong Recyclable
    bar_green = f"{norm_pct:.0f}%"
    bar_red   = f"{100 - norm_pct:.0f}%"
    dec_str   = f"{decision_val:.4f}"

    if prediction == 1:
        st.markdown(f"""
        <div class="res-card res-recyclable">
            <div class="res-icon">♻️</div>
            <div class="res-verdict rv-green">Recyclable</div>
            <div class="res-sub">This waste sample can be directed to a recycling stream.<br>
            Ensure proper sorting before processing.</div>
            <div class="score-wrap">
                <div class="score-label">SVM Decision Score</div>
                <div class="score-bg"><div class="score-fill-g" style="width:{bar_green}"></div></div>
                <div class="score-val">Raw score: {dec_str} &nbsp;(positive → Recyclable)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ Route to **recycling facility**. High recycling potential detected.", icon="♻️")

        tip = (
            f"**Tip:** {WASTE_TYPE_MAP[waste_type]} waste with "
            f"**{MATERIAL_MAP[material_composition].lower()}** composition and "
            f"**{TOXICITY_MAP[toxicity_level].lower()}** toxicity is typically accepted at "
            "standard recycling centres. Confirm local regulations before disposal."
        )
    else:
        st.markdown(f"""
        <div class="res-card res-nonrecyclable">
            <div class="res-icon">🚫</div>
            <div class="res-verdict rv-red">Non-Recyclable</div>
            <div class="res-sub">This waste sample should be directed to landfill or<br>
            specialised disposal — not standard recycling.</div>
            <div class="score-wrap">
                <div class="score-label">SVM Decision Score</div>
                <div class="score-bg"><div class="score-fill-r" style="width:{bar_red}"></div></div>
                <div class="score-val">Raw score: {dec_str} &nbsp;(negative → Non-Recyclable)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Route to **general waste or hazardous disposal** as appropriate.", icon="🗑️")

        tip = (
            f"**Tip:** {WASTE_TYPE_MAP[waste_type]} waste with "
            f"**{TOXICITY_MAP[toxicity_level].lower()}** toxicity often requires "
            "specialised handling. Check local hazardous waste guidelines before disposal."
        )

    # ── Input summary ──
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">📋 Input Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-cell"><div class="ic-lbl">Waste Type</div><div class="ic-val">{WASTE_TYPE_MAP[waste_type]} (encoded: {waste_type})</div></div>
        <div class="info-cell"><div class="ic-lbl">Material Composition</div><div class="ic-val">{MATERIAL_MAP[material_composition]} (encoded: {material_composition})</div></div>
        <div class="info-cell"><div class="ic-lbl">Recycling Potential</div><div class="ic-val">{recycling_potential:.1f} / 100</div></div>
        <div class="info-cell"><div class="ic-lbl">Toxicity Level</div><div class="ic-val">{TOXICITY_MAP[toxicity_level]} (encoded: {toxicity_level})</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disposal tip ──
    st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)


# ── How it works ──────────────────────────────────────────────────────────────
st.markdown('<hr class="hr">', unsafe_allow_html=True)
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    This classifier uses a **Support Vector Machine (SVM)** with an **RBF (Radial Basis Function)** kernel
    trained on a waste management dataset.

    | Parameter | Value |
    |-----------|-------|
    | Algorithm | SVC — Support Vector Classifier |
    | Kernel | RBF (Radial Basis Function) |
    | C (regularisation) | 1.0 |
    | Gamma | scale |
    | Classes | 0 = Non-Recyclable · 1 = Recyclable |

    **Decision score:** SVMs produce a raw decision function score rather than a probability.
    A positive score means the model leans Recyclable; negative means Non-Recyclable.
    The bar above visualises how strongly the model is pulling in either direction.
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="hr">
<div class="footer">Support Vector Classifier · RBF Kernel &nbsp;|&nbsp; Waste Management Dataset</div>
""", unsafe_allow_html=True)
