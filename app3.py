"""
Fatty Liver (NAFLD) Risk Checker
Karthik P | MBA Dissertation 2026

A combined screening tool that FUSES two models:
  1) a tabular model trained on health/lifestyle data (the 1700-record set)
  2) an image model for a liver ultrasound (a CNN, or a built-in image analysis)
The final risk blends both when an ultrasound is uploaded.

A helper for you and your doctor - not a medical diagnosis.
"""
import os
import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image, ImageStat, ImageFilter

st.set_page_config(page_title="Fatty Liver Risk Analyzer", page_icon="🩺", layout="wide")
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
       color:#fff; background:linear-gradient(120deg,#0f766e,#155e75,#4338ca,#7c3aed,#0f766e);
       background-size:320% 320%; animation:heroShift 14s ease infinite, fadeUp .5s ease;
       box-shadow:0 20px 50px rgba(67,56,249,.28), inset 0 1px 0 rgba(255,255,255,.15);}
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.05rem; font-weight:700; margin:0 0 6px 0; color:#fff; letter-spacing:-.5px;}
.hero p {margin:0; opacity:.95; font-size:1.05rem; max-width:660px;}
.badge {display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,.16); padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600; margin-bottom:12px;}
.dot {width:8px; height:8px; border-radius:50%; background:#4ade80; box-shadow:0 0 10px #4ade80;}
.pill {display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.14); padding:6px 13px; border-radius:999px; font-size:.82rem; font-weight:500; margin-top:12px; margin-right:7px;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:20px !important; border:1px solid rgba(255,255,255,.09) !important;
   background:rgba(255,255,255,.035) !important; backdrop-filter:blur(10px); box-shadow:0 12px 40px rgba(0,0,0,.35);
   animation:fadeUp .5s ease; transition:transform .18s, border-color .18s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-2px); border-color:rgba(94,234,212,.35) !important;}
.sec {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem; color:#f1f5f9; margin:0;}
.sub {color:#94a3b8; font-size:.86rem; margin:2px 0 8px 0;}
.stButton>button {background:linear-gradient(120deg,#14b8a6,#4338ca); color:#fff; border:0; border-radius:16px;
   padding:.8rem 1rem; font-weight:800; font-size:1.05rem; box-shadow:0 10px 28px rgba(67,56,249,.38); transition:transform .15s, filter .15s;}
.stButton>button:hover {transform:translateY(-2px); filter:brightness(1.12);}
.scorecard{background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09); border-radius:16px; padding:15px 16px;}
.scorecard .v{font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; line-height:1;
   background:linear-gradient(120deg,#5eead4,#818cf8); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.scorecard .l{font-size:.73rem; color:#94a3b8; margin-top:6px; line-height:1.3;}
.reco{border-radius:18px; padding:20px 22px; color:#fff;}
.reco h3{font-family:'Space Grotesk',sans-serif; margin:0 0 6px 0; color:#fff; font-size:1.3rem; font-weight:700;}
.reco p{margin:0; color:#fff; font-size:1rem; opacity:.98; line-height:1.5;}
.tip{display:inline-block; background:rgba(255,255,255,.06); color:#cbd5e1; border:1px solid rgba(255,255,255,.12);
   padding:8px 13px; border-radius:999px; font-size:.83rem; margin:5px 7px 0 0;}
.gaugewrap svg{width:100%; max-width:240px; height:auto;}
@media (max-width: 640px){
  .block-container{padding-left:.7rem; padding-right:.7rem; padding-top:.6rem;}
  .hero{padding:22px 20px; border-radius:18px;} .hero h1{font-size:1.55rem;} .hero p{font-size:.95rem;}
  .pill{font-size:.75rem; padding:5px 11px; margin-top:8px;} .sec{font-size:1.05rem;}
  .reco{padding:16px 18px;} .reco h3{font-size:1.15rem;} .reco p{font-size:.94rem;}
  .scorecard .v{font-size:1.5rem;} .stButton>button{font-size:1rem;} .gaugewrap svg{max-width:200px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge"><span class="dot"></span></div>
  <h1>Fatty Liver Risk Analyzer</h1>
  <p>A trained model reads your health details, and — if you add a liver ultrasound — a second image model checks the scan. The two are combined into one risk estimate.</p>
  <div>
    <span class="pill">Health details + optional ultrasound</span>
    <span class="pill">Free & private</span>
  </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
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


def image_finding(p):
    # plain-language, honest verdict from the image model's score (not a detailed diagnosis)
    if 0.4 < p < 0.6:
        return "the ultrasound is borderline — hard to tell from the picture alone"
    verdict = "the ultrasound looks like a fatty liver" if p >= 0.6 else "the ultrasound looks fairly normal"
    conf = "fairly confident" if abs(p - 0.5) >= 0.3 else "not very confident"
    return f"{verdict} ({conf})"


def gauge(pct, c1, c2, cmain, word):
    circ = 2 * math.pi * 80
    return f"""
    <div class="gaugewrap" style="text-align:center; padding:6px 0">
      <svg viewBox="0 0 200 200">
        <defs>
          <linearGradient id="arc" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/></linearGradient>
          <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="16"/>
        <circle cx="100" cy="100" r="80" fill="none" stroke="url(#arc)" stroke-width="16" stroke-linecap="round"
          stroke-dasharray="{circ*pct} {circ}" transform="rotate(-90 100 100)" filter="url(#glow)"/>
        <text x="100" y="94" text-anchor="middle" font-size="50" font-weight="700" fill="{cmain}" style="font-family:'Space Grotesk',sans-serif">{pct*100:.0f}%</text>
        <text x="100" y="126" text-anchor="middle" font-size="19" font-weight="700" fill="{cmain}" style="font-family:'Space Grotesk',sans-serif">{word}</text>
      </svg>
    </div>"""


model_bundle, cnn = load_model(), load_cnn()

with st.container(border=True):
    st.markdown('<p class="sec">Your details</p>'
                '<p class="sub">Everyday health questions — no blood tests needed except the last one (optional).</p>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    age = a.number_input("Age", 1, 100, 40, help="Your age in years")
    gender = b.selectbox("Gender", ["Female", "Male", "Other / prefer not to say"], help="Used only to fine-tune the score")
    bmi = c.number_input("BMI", 10.0, 60.0, 25.0, 0.1, help="Body Mass Index — your weight for your height (an online tool can work it out)")
    activity = a.number_input("Physical activity (hours per week)", 0.0, 30.0, 3.0, 0.5, help="Hours of exercise/activity per week")
    genetic = b.selectbox("Family history of liver disease", ["None", "Some", "Strong"], help="Genetic / family risk")
    smoking = c.selectbox("Do you smoke?", ["No", "Yes"])
    diabetes = a.selectbox("Do you have diabetes?", ["No", "Yes"], help="Type-2 diabetes")
    hypertension = b.selectbox("High blood pressure?", ["No", "Yes"], help="Hypertension")

    d1, d2 = st.columns(2)
    drank = d1.selectbox("Did you drink alcohol this week?", ["No", "Yes"])
    alcohol = 0.0
    if drank == "Yes":
        alcohol = d2.number_input("How many drinks this week?", 0.0, 40.0, 3.0, 0.5,
                                  help="Roughly how many alcoholic drinks this week")

    e1, e2 = st.columns(2)
    has_lft = e1.selectbox("Do you have a liver function test score?", ["No", "Yes"],
                           help="A number from a liver blood test — leave as 'No' if you don't have one")
    lft = 40.0
    if has_lft == "Yes":
        lft = e2.number_input("Liver function test score", 0.0, 150.0, 40.0, 1.0)

    with st.expander("➕  Add a liver ultrasound image (optional — adds the image model)"):
        up = st.file_uploader("Upload a B-mode liver ultrasound (PNG/JPG)", type=["png", "jpg", "jpeg"])
        us_img = Image.open(up) if up else None
        if us_img:
            st.image(us_img, width=260)
        st.caption(("Deep-learning image model active." if cnn else "Built-in image analysis active.")
                   + " When added, the image result is blended with your health-details result.")

go = st.button("🔍  Check my liver risk", use_container_width=True)

if go:
    if model_bundle is None:
        st.error("Model file (model.pkl) not found. Add it to the app folder / repo.")
        st.stop()

    values = {
        "Age": age,
        "Gender": {"Female": 0, "Male": 1, "Other / prefer not to say": 0.5}[gender],
        "BMI": bmi,
        "AlcoholConsumption": alcohol,
        "Smoking": 1 if smoking == "Yes" else 0,
        "GeneticRisk": {"None": 0, "Some": 1, "Strong": 2}[genetic],
        "PhysicalActivity": activity,
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "LiverFunctionTest": lft,
    }
    feats = model_bundle["features"]
    row = pd.DataFrame([[values.get(f, 0) for f in feats]], columns=feats)
    tab_prob = float(model_bundle["model"].predict_proba(row)[0, 1])

    img_prob = None
    if us_img is not None:
        img_prob = us_cnn_prob(us_img, cnn) if cnn else us_heuristic_prob(us_img)

    if img_prob is not None:
        prob = 0.6 * tab_prob + 0.4 * img_prob
    else:
        prob = tab_prob

    if prob >= 0.6:
        c1, c2, cmain, word = "#fb7185", "#dc2626", "#fb7185", "High"
        title, msg = "🔴 Please see a doctor soon", \
            "The result suggests a high chance of fatty liver. Please book a check-up and ask about a liver scan (ultrasound / FibroScan)."
    elif prob >= 0.3:
        c1, c2, cmain, word = "#fbbf24", "#f59e0b", "#fbbf24", "Moderate"
        title, msg = "🟠 Worth getting checked", \
            "There are some warning signs. It's a good idea to speak to a doctor, and small lifestyle changes make a big difference."
    else:
        c1, c2, cmain, word = "#34d399", "#16a34a", "#34d399", "Low"
        title, msg = "🟢 Looking good", \
            "Low chance of fatty liver right now. Keep up the healthy habits and check again in about a year."

    st.write("")
    g, d = st.columns([1, 1.3])
    with g:
        with st.container(border=True):
            st.markdown(gauge(min(max(prob, 0), 1), c1, c2, cmain, word), unsafe_allow_html=True)
    with d:
        with st.container(border=True):
            st.markdown(f'<div class="reco" style="background:linear-gradient(120deg,{c1},{c2});'
                        f'box-shadow:0 14px 34px {c2}44"><h3>{title}</h3><p>{msg}</p></div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top:14px">'
                        '<span class="tip">🥗 More veggies, less sugar</span>'
                        '<span class="tip">🚶 Move ~30 min a day</span>'
                        '<span class="tip">🍺 Cut back on alcohol</span></div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.markdown(f'<div class="scorecard"><div class="v">{bmi:.0f}</div>'
                        f'<div class="l">Your BMI<br>(25+ raises risk)</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="scorecard"><div class="v">{alcohol:.0f}</div>'
                        f'<div class="l">Drinks / week<br>(lower is better)</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="scorecard"><div class="v">{activity:.0f}h</div>'
                        f'<div class="l">Activity / week<br>(more is better)</div></div>', unsafe_allow_html=True)
            if img_prob is not None:
                st.caption(f"🖼️ Ultrasound: {image_finding(img_prob)} — a radiologist should confirm.")

    with st.expander("Why did I get this result?"):
        if img_prob is not None:
            looked = "We looked at your answers and your ultrasound picture"
        else:
            looked = "We analysed only your answers (no ultrasound was uploaded)"
        reasons = []
        if bmi >= 25: reasons.append("your weight (BMI)")
        if diabetes == "Yes": reasons.append("diabetes")
        if hypertension == "Yes": reasons.append("high blood pressure")
        if alcohol >= 7: reasons.append("drinking alcohol")
        if smoking == "Yes": reasons.append("smoking")
        if activity < 2: reasons.append("not much exercise")
        if genetic != "None": reasons.append("family history")
        if reasons:
            st.write(f"{looked}. The main things raising your risk are: **" + ", ".join(reasons) + "**.")
        else:
            st.write(f"{looked}. Nothing major is raising your risk right now — keep it up! 👍")
        if img_prob is not None:
            st.write(f"🖼️ **From the picture:** {image_finding(img_prob)}. A radiologist should confirm.")
        st.write("Eating healthier, moving more, and cutting back on alcohol can all help lower your risk.")

    st.caption("💙 This is a helper for you and your doctor — not a medical diagnosis. If you're worried, please see a doctor.")

st.caption("MBA Dissertation 2026 · Karthik P")
