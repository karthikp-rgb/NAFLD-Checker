"""
AI-Enabled Clinical Decision Support System for Early Detection of NAFLD
Karthik P | MBA Dissertation 2026

Enter 7 routine values (optionally an ultrasound image) ->
HSI + FIB-4 + APRI scores + ML risk -> SCREEN / REFER / MONITOR.

Screening decision support only - NOT a diagnosis.
"""
import os
import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image, ImageStat, ImageFilter

st.set_page_config(page_title="NAFLD Early-Detection CDSS", page_icon="🩺", layout="wide")

MODEL_PATH, CNN_PATH = "model.pkl", "ultrasound_cnn.pt"

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1150px;}
html, body, [class*="css"] {font-family: 'Inter','Segoe UI',system-ui,sans-serif;}
.hero {background: linear-gradient(120deg,#0f766e 0%,#155e75 55%,#1e3a8a 100%);
       border-radius: 18px; padding: 26px 32px; color: #fff;
       box-shadow: 0 10px 30px rgba(15,118,110,.25); margin-bottom: 18px;}
.hero h1 {font-size: 1.7rem; font-weight: 800; margin: 0 0 4px 0; color:#fff;}
.hero p {margin: 0; opacity: .92; font-size: .96rem;}
.pill {display:inline-block; background: rgba(255,255,255,.18); padding: 3px 11px;
       border-radius: 999px; font-size: .78rem; margin-top: 10px; margin-right:6px;}
div[data-testid="stVerticalBlockBorderWrapper"]{
       border-radius: 16px !important; box-shadow: 0 4px 18px rgba(30,41,59,.07);
       border: 1px solid #eef1f5 !important; background:#fff;}
.sec {font-weight:700; font-size:1.02rem; color:#0f172a; margin:0 0 2px 0;}
.sub {color:#64748b; font-size:.82rem; margin:0 0 8px 0;}
.stButton>button {background: linear-gradient(120deg,#0f766e,#1e3a8a); color:#fff;
       border:0; border-radius: 12px; padding:.6rem 1rem; font-weight:700;
       box-shadow:0 6px 16px rgba(15,118,110,.28);}
.stButton>button:hover {filter:brightness(1.07);}
.scorecard{border-radius:14px; padding:14px 16px; background:#f8fafc; border:1px solid #eef1f5;}
.scorecard .v{font-size:1.5rem; font-weight:800; color:#0f172a; line-height:1;}
.scorecard .l{font-size:.78rem; color:#64748b; margin-top:3px;}
.reco{border-radius:16px; padding:18px 20px; color:#fff; font-weight:600;}
.reco h3{margin:0 0 4px 0; color:#fff; font-size:1.15rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 NAFLD Early-Detection CDSS</h1>
  <p>An AI decision-support tool that flags early Non-Alcoholic Fatty Liver Disease from routine tests.</p>
  <span class="pill">~38% of Indian adults affected</span>
  <span class="pill">Non-invasive · low-cost</span>
  <span class="pill">Screening aid — not a diagnosis</span>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ scores
def hsi(alt, ast, bmi, female, diabetes):
    # Hepatic Steatosis Index. >36 suggests NAFLD, <30 rules it out.
    return 8 * (alt / ast) + bmi + (2 if female else 0) + (2 if diabetes else 0)


def fib4(age, ast, alt, plt):
    return (age * ast) / (plt * math.sqrt(alt)) if plt > 0 and alt > 0 else None


def apri(ast, plt, uln=40):
    return ((ast / uln) / plt) * 100 if plt > 0 else None


@st.cache_resource
def load_tabular():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_resource
def load_cnn():
    if not os.path.exists(CNN_PATH):
        return None
    try:
        import torch, torch.nn as nn
        from torchvision import models
        ckpt = torch.load(CNN_PATH, map_location="cpu")
        net = models.mobilenet_v2()
        net.classifier[1] = nn.Linear(net.last_channel, 2)
        net.load_state_dict(ckpt["state_dict"]); net.eval()
        return {"net": net, "classes": ckpt.get("classes", ["fatty", "normal"])}
    except Exception:
        return None


def us_cnn_prob(img, cnn):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Grayscale(3), transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        p = torch.softmax(cnn["net"](tf(img).unsqueeze(0)), 1)[0]
    idx = cnn["classes"].index("fatty") if "fatty" in cnn["classes"] else 0
    return float(p[idx])


def us_heuristic_prob(img):
    g = img.convert("L").resize((256, 256))
    mean_b = ImageStat.Stat(g).mean[0] / 255.0
    texture = ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    score = 0.75 * mean_b + 0.25 * (1 - min(texture * 4, 1))
    return float(min(max((score - 0.35) / 0.4, 0), 1))


def gauge(pct, color):
    r, cx = 80, 100
    circ = 2 * math.pi * r
    filled = circ * pct
    return f"""
    <div style="text-align:center">
      <svg width="210" height="210" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="#eef1f5" stroke-width="18"/>
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="18"
          stroke-linecap="round" stroke-dasharray="{filled} {circ}"
          transform="rotate(-90 100 100)"/>
        <text x="100" y="96" text-anchor="middle" font-size="44" font-weight="800"
          fill="#0f172a">{pct*100:.0f}%</text>
        <text x="100" y="122" text-anchor="middle" font-size="15" fill="#64748b">NAFLD risk</text>
      </svg>
    </div>"""


# ------------------------------------------------------------------ inputs
tab_model, cnn = load_tabular(), load_cnn()

with st.container(border=True):
    st.markdown('<p class="sec">Patient details</p>'
                '<p class="sub">Just seven routine values — no special tests needed.</p>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    age = a.number_input("Age (years)", 18, 100, 45)
    sex = b.selectbox("Sex", ["Female", "Male"])
    diabetes = c.selectbox("Type-2 diabetes", ["No", "Yes"])
    bmi = a.number_input("BMI (kg/m²)", 12.0, 60.0, 27.0, 0.1)
    alt = b.number_input("ALT (U/L)", 5.0, 400.0, 35.0, 1.0)
    ast = c.number_input("AST (U/L)", 5.0, 400.0, 30.0, 1.0)
    platelets = a.number_input("Platelets (×10⁹/L)", 50.0, 500.0, 250.0, 1.0)

    with st.expander("➕  Add a liver ultrasound image (optional)"):
        up = st.file_uploader("B-mode ultrasound (PNG/JPG)", type=["png", "jpg", "jpeg"])
        us_img = Image.open(up) if up else None
        if us_img:
            st.image(us_img, width=260)
        st.caption(("Deep-learning CNN active" if cnn else "Image-analysis mode")
                   + " · optional, improves accuracy.")

go = st.button("🔍  Assess NAFLD risk", use_container_width=True)

# ------------------------------------------------------------------ result
if go:
    female = sex == "Female"
    dm = diabetes == "Yes"
    H = hsi(alt, ast, bmi, female, dm)
    F = fib4(age, ast, alt, platelets)
    A = apri(ast, platelets)

    if tab_model:
        row = pd.DataFrame([[age, 0 if female else 1, bmi, int(dm), alt, ast, platelets]],
                           columns=tab_model["features"])
        ml = float(tab_model["model"].predict_proba(row)[0, 1])
    else:
        ml = min(max((H - 30) / 12, 0), 1)

    hsi_like = min(max((H - 30) / 12, 0), 1)
    us_prob = None
    if 'us_img' in dir() and us_img is not None:
        us_prob = us_cnn_prob(us_img, cnn) if cnn else us_heuristic_prob(us_img)

    parts = [("Machine-learning model", ml, .55), ("Hepatic Steatosis Index", hsi_like, .45)]
    if us_prob is not None:
        parts = [("Machine-learning model", ml, .4), ("Hepatic Steatosis Index", hsi_like, .3),
                 ("Ultrasound", us_prob, .3)]
    overall = sum(p * w for _, p, w in parts) / sum(w for _, _, w in parts)

    fib_high = (F and F >= 2.67) or (A and A >= 1.0)
    fib_mid = F and 1.3 <= F < 2.67

    if overall >= 0.6 or fib_high:
        color, rc = "#dc2626", ("🔴 REFER",
            "High likelihood of NAFLD" + (" with possible fibrosis" if fib_high else "")
            + ". Refer for a confirmatory ultrasound / FibroScan and hepatology review.")
    elif overall >= 0.3 or fib_mid:
        color, rc = "#f59e0b", ("🟠 SCREEN",
            "Moderate likelihood. Arrange an abdominal ultrasound, repeat liver tests, and start lifestyle counselling.")
    else:
        color, rc = "#16a34a", ("🟢 MONITOR",
            "Low likelihood at present. Advise diet and exercise; recheck in about 12 months.")

    st.write("")
    g, d = st.columns([1, 1.25])
    with g:
        with st.container(border=True):
            st.markdown(gauge(min(max(overall, 0), 1), color), unsafe_allow_html=True)
    with d:
        with st.container(border=True):
            st.markdown(
                f'<div class="reco" style="background:{color}"><h3>{rc[0]}</h3>{rc[1]}</div>',
                unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.markdown(f'<div class="scorecard"><div class="v">{H:.0f}</div>'
                        f'<div class="l">Steatosis Index<br>&ge;36 high · &lt;30 low</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="scorecard"><div class="v">{F:.2f}</div>'
                        f'<div class="l">FIB-4<br>&ge;2.67 fibrosis risk</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="scorecard"><div class="v">{A:.2f}</div>'
                        f'<div class="l">APRI<br>&ge;1.0 sig. fibrosis</div></div>', unsafe_allow_html=True)

    with st.expander("Why this result?"):
        for name, p, w in parts:
            st.write(f"- **{name}**: {p*100:.0f}%  (weight {int(w*100)}%)")
        drivers = []
        if bmi >= 25: drivers.append(f"BMI {bmi:.0f}")
        if dm: drivers.append("type-2 diabetes")
        if alt > 40: drivers.append("raised ALT")
        if ast > 40: drivers.append("raised AST")
        st.write("**Risk factors present:** " + (", ".join(drivers) if drivers else "none major"))
        if tab_model and tab_model.get("synthetic"):
            st.caption("Bundled model trained on synthetic demo data — retrain on real data before clinical use.")
    st.caption("⚠️ Screening decision support only — not a diagnosis. Confirm with imaging and clinical judgement.")

st.caption("MBA Dissertation 2026 · Karthik P ·")

