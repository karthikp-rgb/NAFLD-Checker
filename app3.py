"""
Fatty Liver (NAFLD) Risk Checker
Karthik P | MBA Dissertation 2026

A friendly early-detection tool: enter a few routine health numbers
(optionally an ultrasound image) -> plain-language risk + what to do next.

A helper for you and your doctor - not a medical diagnosis.
"""
import os
import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image, ImageStat, ImageFilter

st.set_page_config(page_title="Fatty Liver Risk Checker", page_icon="🩺", layout="wide")

MODEL_PATH, CNN_PATH = "model.pkl", "ultrasound_cnn.pt"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
#MainMenu, header, footer {visibility: hidden;}
.stApp {background:
  radial-gradient(1100px 560px at 8% -8%, rgba(20,184,166,.16), transparent 60%),
  radial-gradient(1000px 520px at 100% 0%, rgba(99,102,241,.16), transparent 55%),
  radial-gradient(900px 500px at 50% 120%, rgba(217,70,239,.10), transparent 55%),
  #0a0e1a;}
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1120px;}
html, body, [class*="css"], .stMarkdown, p, label, div {font-family:'Plus Jakarta Sans',system-ui,sans-serif;}
@keyframes heroShift {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes fadeUp {from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none}}
.hero {position:relative; overflow:hidden; border-radius:22px; padding:30px 34px; margin-bottom:18px;
       color:#fff; animation:fadeUp .5s ease;
       background:linear-gradient(120deg,#0f766e,#155e75,#4338ca,#7c3aed,#0f766e);
       background-size:320% 320%; animation:heroShift 14s ease infinite, fadeUp .5s ease;
       box-shadow:0 20px 50px rgba(67,56,249,.28), inset 0 1px 0 rgba(255,255,255,.15);}
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.05rem; font-weight:700;
          margin:0 0 6px 0; color:#fff; letter-spacing:-.5px;}
.hero p {margin:0; opacity:.95; font-size:1.05rem; max-width:640px;}
.badge {display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,.16);
        padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600; margin-bottom:12px;}
.dot {width:8px; height:8px; border-radius:50%; background:#4ade80; box-shadow:0 0 10px #4ade80;}
.pill {display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.14);
       padding:6px 13px; border-radius:999px; font-size:.82rem; font-weight:500; margin-top:12px; margin-right:7px;}
div[data-testid="stVerticalBlockBorderWrapper"]{
   border-radius:20px !important; border:1px solid rgba(255,255,255,.09) !important;
   background:rgba(255,255,255,.035) !important; backdrop-filter:blur(10px);
   box-shadow:0 12px 40px rgba(0,0,0,.35); animation:fadeUp .5s ease; transition:transform .18s, border-color .18s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-2px); border-color:rgba(94,234,212,.35) !important;}
.sec {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem; color:#f1f5f9; margin:0;}
.sub {color:#94a3b8; font-size:.86rem; margin:2px 0 8px 0;}
.stButton>button {background:linear-gradient(120deg,#14b8a6,#4338ca); color:#fff;
   border:0; border-radius:16px; padding:.8rem 1rem; font-weight:800; font-size:1.05rem;
   letter-spacing:.2px; box-shadow:0 10px 28px rgba(67,56,249,.38); transition:transform .15s, filter .15s;}
.stButton>button:hover {transform:translateY(-2px); filter:brightness(1.12);}
.scorecard{background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09);
   border-radius:16px; padding:15px 16px;}
.scorecard .v{font-family:'Space Grotesk',sans-serif; font-size:1.85rem; font-weight:700; line-height:1;
   background:linear-gradient(120deg,#5eead4,#818cf8); -webkit-background-clip:text;
   background-clip:text; -webkit-text-fill-color:transparent;}
.scorecard .l{font-size:.73rem; color:#94a3b8; margin-top:6px; line-height:1.3;}
.reco{border-radius:18px; padding:20px 22px; color:#fff;}
.reco h3{font-family:'Space Grotesk',sans-serif; margin:0 0 6px 0; color:#fff; font-size:1.3rem; font-weight:700;}
.reco p{margin:0; color:#fff; font-size:1rem; opacity:.98; line-height:1.5;}
.tip{display:inline-block; background:rgba(255,255,255,.06); color:#cbd5e1;
   border:1px solid rgba(255,255,255,.12); padding:8px 13px; border-radius:999px;
   font-size:.83rem; margin:5px 7px 0 0; transition:background .15s;}
.tip:hover{background:rgba(255,255,255,.12);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge"><span class="dot"></span> AI-powered early check</div>
  <h1>Fatty Liver Risk Checker</h1>
  <p>Find out your chance of early fatty liver (NAFLD) in 30 seconds — using a few simple numbers from a normal health check-up.</p>
  <div>
    <span class="pill">🇮🇳 ~38% of Indian adults affected</span>
    <span class="pill">No needles · no cost</span>
    <span class="pill">Plain-language result</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------ scores
def hsi(alt, ast, bmi, sex_bonus):
    return 8 * (alt / ast) + bmi + sex_bonus  # Hepatic Steatosis Index


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
    return float(min(max((0.75 * mean_b + 0.25 * (1 - min(texture * 4, 1)) - 0.35) / 0.4, 0), 1))


def gauge(pct, c1, c2, cmain, word):
    circ = 2 * math.pi * 80
    return f"""
    <div style="text-align:center; padding:6px 0">
      <svg width="240" height="240" viewBox="0 0 200 200">
        <defs>
          <linearGradient id="arc" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>
          </linearGradient>
          <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3.4" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="16"/>
        <circle cx="100" cy="100" r="80" fill="none" stroke="url(#arc)" stroke-width="16"
          stroke-linecap="round" stroke-dasharray="{circ*pct} {circ}"
          transform="rotate(-90 100 100)" filter="url(#glow)"/>
        <text x="100" y="94" text-anchor="middle" font-size="50" font-weight="700"
          fill="{cmain}" style="font-family:'Space Grotesk',sans-serif">{pct*100:.0f}%</text>
        <text x="100" y="126" text-anchor="middle" font-size="19" font-weight="700"
          fill="{cmain}" style="font-family:'Space Grotesk',sans-serif">{word}</text>
      </svg>
    </div>"""


# ------------------------------------------------------------ inputs
tab_model, cnn = load_tabular(), load_cnn()

with st.container(border=True):
    st.markdown('<p class="sec">Your details</p>'
                '<p class="sub">Just a few numbers from a normal health check-up.</p>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    age = a.number_input("Age", 1, 100, 40, help="Your age in years")
    gender = b.selectbox("Gender", ["Female", "Male", "Other / prefer not to say"],
                         help="Used only to fine-tune the score")
    diabetes = c.selectbox("Do you have diabetes?", ["No", "Yes"], help="Type-2 diabetes")
    bmi = a.number_input("BMI", 10.0, 60.0, 25.0, 0.1,
                         help="Body Mass Index — your weight for your height. A doctor or online tool can work it out.")
    alt = b.number_input("ALT", 5.0, 400.0, 30.0, 1.0,
                         help="A liver enzyme (ALT) from a routine blood test")
    ast = c.number_input("AST", 5.0, 400.0, 28.0, 1.0,
                         help="Another liver enzyme (AST) from a blood test")
    platelets = a.number_input("Platelet count", 50.0, 500.0, 250.0, 1.0,
                               help="A number from your blood count (CBC report), in ×10⁹/L")

    with st.expander("➕  Have a liver ultrasound picture? Add it (optional)"):
        up = st.file_uploader("Upload the ultrasound image", type=["png", "jpg", "jpeg"])
        us_img = Image.open(up) if up else None
        if us_img:
            st.image(us_img, width=260)
        st.caption(("Deep-learning model active." if cnn else "Picture-analysis mode.")
                   + " This step is optional.")

go = st.button("🔍  Check my liver risk", use_container_width=True)

# ------------------------------------------------------------ result
if go:
    female = gender == "Female"
    other = gender.startswith("Other")
    sex_val = 0.0 if female else (1.0 if gender == "Male" else 0.5)
    sex_bonus = 2 if female else (1 if other else 0)
    dm = diabetes == "Yes"

    H = hsi(alt, ast, bmi, sex_bonus)
    F = fib4(age, ast, alt, platelets)
    A = apri(ast, platelets)

    if tab_model:
        row = pd.DataFrame([[age, sex_val, bmi, int(dm), alt, ast, platelets]],
                           columns=tab_model["features"])
        ml = float(tab_model["model"].predict_proba(row)[0, 1])
    else:
        ml = min(max((H - 30) / 12, 0), 1)

    hsi_like = min(max((H - 30) / 12, 0), 1)
    us_prob = us_cnn_prob(us_img, cnn) if (us_img and cnn) else (us_heuristic_prob(us_img) if us_img else None)

    parts = [("Health-number model", ml, .55), ("Fatty-liver score", hsi_like, .45)]
    if us_prob is not None:
        parts = [("Health-number model", ml, .4), ("Fatty-liver score", hsi_like, .3),
                 ("Ultrasound picture", us_prob, .3)]
    overall = sum(p * w for _, p, w in parts) / sum(w for _, _, w in parts)

    fib_high = (F and F >= 2.67) or (A and A >= 1.0)
    fib_mid = F and 1.3 <= F < 2.67

    if overall >= 0.6 or fib_high:
        c1, c2, cmain, word = "#fb7185", "#dc2626", "#fb7185", "High"
        title, msg = "🔴 Please see a doctor soon", \
            "There's a high chance of early fatty liver. Ask your doctor for a liver scan (ultrasound or FibroScan) and a check-up."
    elif overall >= 0.3 or fib_mid:
        c1, c2, cmain, word = "#fbbf24", "#f59e0b", "#fbbf24", "Moderate"
        title, msg = "🟠 Worth getting checked", \
            "There are some early signs. It's a good idea to get an ultrasound scan and speak to a doctor. Small lifestyle changes really help."
    else:
        c1, c2, cmain, word = "#34d399", "#16a34a", "#34d399", "Low"
        title, msg = "🟢 Looking good", \
            "Low chance of fatty liver right now. Keep eating well and staying active, and check again in about a year."

    st.write("")
    g, d = st.columns([1, 1.3])
    with g:
        with st.container(border=True):
            st.markdown(gauge(min(max(overall, 0), 1), c1, c2, cmain, word), unsafe_allow_html=True)
    with d:
        with st.container(border=True):
            st.markdown(f'<div class="reco" style="background:linear-gradient(120deg,{c1},{c2});'
                        f'box-shadow:0 14px 34px {c2}44"><h3>{title}</h3><p>{msg}</p></div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="margin-top:14px">'
                        '<span class="tip">🥗 More veggies, less sugar</span>'
                        '<span class="tip">🚶 Move ~30 min a day</span>'
                        '<span class="tip">💧 Skip sugary drinks</span></div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.markdown(f'<div class="scorecard"><div class="v">{H:.0f}</div>'
                        f'<div class="l">Fatty-liver score<br>(higher = more likely)</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="scorecard"><div class="v">{F:.2f}</div>'
                        f'<div class="l">Liver-scarring score<br>(higher = more concern)</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="scorecard"><div class="v">{A:.2f}</div>'
                        f'<div class="l">Scarring check<br>(higher = more concern)</div></div>', unsafe_allow_html=True)

    with st.expander("What does this mean?"):
        st.write("This check adds up a few signals to estimate your chance of **early fatty liver** — "
                 "a build-up of fat in the liver that usually has no symptoms.")
        for name, p, w in parts:
            st.write(f"- {name}: about {p*100:.0f} out of 100")
        drivers = []
        if bmi >= 25: drivers.append(f"BMI {bmi:.0f}")
        if dm: drivers.append("diabetes")
        if alt > 40: drivers.append("higher ALT")
        if ast > 40: drivers.append("higher AST")
        if drivers:
            st.write("**Things adding to your risk:** " + ", ".join(drivers))
        if tab_model and tab_model.get("synthetic"):
            st.caption("Demo model trained on example data — for a class project, not for real medical use.")

    st.caption("💙 This is a helper for you and your doctor — not a medical diagnosis. If you're worried, please see a doctor.")

st.caption("MBA Dissertation 2026 · Karthik P")
