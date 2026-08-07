"""
AI-Enabled Clinical Decision Support System for Early Detection of NAFLD
Karthik P | MBA Dissertation 2026

Streamlit app: enter routine labs + anthropometry (and optionally an
ultrasound image) -> non-invasive scores (FIB-4, FLI, APRI) + ML risk ->
SCREEN / REFER / MONITOR recommendation.

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

MODEL_PATH = "model.pkl"
CNN_PATH = "ultrasound_cnn.pt"


# ----------------------------- non-invasive scores -----------------------------
def fib4(age, ast, alt, platelets):
    if platelets <= 0 or alt <= 0:
        return None
    return (age * ast) / (platelets * math.sqrt(alt))


def apri(ast, platelets, ast_uln=40):
    if platelets <= 0:
        return None
    return ((ast / ast_uln) / platelets) * 100


def fatty_liver_index(bmi, waist, triglycerides, ggt):
    # Bedogni et al. Fatty Liver Index (0-100); >=60 rules in, <30 rules out.
    L = (0.953 * math.log(triglycerides) + 0.139 * bmi +
         0.718 * math.log(ggt) + 0.053 * waist - 15.745)
    return (math.exp(L) / (1 + math.exp(L))) * 100


# ----------------------------- models -----------------------------
@st.cache_resource
def load_tabular():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_resource
def load_cnn():
    """Load the ultrasound CNN if torch + weights are available; else None."""
    if not os.path.exists(CNN_PATH):
        return None
    try:
        import torch
        from torchvision import models
        import torch.nn as nn
        ckpt = torch.load(CNN_PATH, map_location="cpu")
        net = models.mobilenet_v2()
        net.classifier[1] = nn.Linear(net.last_channel, 2)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return {"net": net, "classes": ckpt.get("classes", ["fatty", "normal"])}
    except Exception as e:  # torch missing or load error -> fall back
        st.session_state["_cnn_err"] = str(e)
        return None


def ultrasound_cnn_prob(img, cnn):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(cnn["net"](x), dim=1)[0]
    fatty_idx = cnn["classes"].index("fatty") if "fatty" in cnn["classes"] else 0
    return float(p[fatty_idx])


def ultrasound_heuristic_prob(img):
    """Medically-grounded proxy: a fatty liver is hyperechoic (brighter) with
    a smoother texture on B-mode ultrasound. We combine normalised mean
    brightness with an edge/texture measure. Approximate demo only."""
    g = img.convert("L").resize((256, 256))
    mean_b = ImageStat.Stat(g).mean[0] / 255.0          # 0..1 brightness
    edges = g.filter(ImageFilter.FIND_EDGES)
    texture = ImageStat.Stat(edges).mean[0] / 255.0     # higher = more edges
    # bright + low texture -> more likely fatty
    score = 0.75 * mean_b + 0.25 * (1 - min(texture * 4, 1))
    return float(min(max((score - 0.35) / 0.4, 0), 1))  # rescale to 0..1


# ----------------------------- UI -----------------------------
st.title("🩺 AI-Enabled CDSS for Early Detection of NAFLD")
st.caption("Non-Alcoholic Fatty Liver Disease | routine blood markers + non-invasive scores + optional ultrasound. "
           "Screening decision support — not a diagnosis.")

tab_model = load_tabular()
cnn = load_cnn()

with st.expander("ℹ️  About this tool & disclaimer"):
    st.markdown(
        "This tool estimates the **likelihood of early NAFLD** from routine, low-cost inputs and "
        "suggests a next step. It computes the **FIB-4**, **Fatty Liver Index (FLI)** and **APRI** "
        "scores, runs a machine-learning model, and (optionally) analyses an ultrasound image.\n\n"
        "It is a **screening aid for clinicians**, not a diagnostic test. Fatty liver is confirmed by "
        "ultrasound/FibroScan or biopsy and a doctor's judgement.")

left, right = st.columns([1.05, 1])

with left:
    st.subheader("1. Patient details & blood tests")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", 18, 100, 45)
    sex = c2.selectbox("Sex", ["Female", "Male"])
    diabetes = c3.selectbox("Type-2 diabetes?", ["No", "Yes"])
    bmi = c1.number_input("BMI (kg/m²)", 12.0, 60.0, 27.0, 0.1)
    waist = c2.number_input("Waist (cm)", 50.0, 180.0, 95.0, 0.5)
    glucose = c3.number_input("Fasting glucose (mg/dL)", 50.0, 400.0, 100.0, 1.0)
    alt = c1.number_input("ALT (U/L)", 5.0, 400.0, 35.0, 1.0)
    ast = c2.number_input("AST (U/L)", 5.0, 400.0, 30.0, 1.0)
    ggt = c3.number_input("GGT (U/L)", 5.0, 600.0, 40.0, 1.0)
    triglycerides = c1.number_input("Triglycerides (mg/dL)", 30.0, 800.0, 150.0, 1.0)
    hdl = c2.number_input("HDL (mg/dL)", 15.0, 120.0, 45.0, 1.0)
    platelets = c3.number_input("Platelets (x10⁹/L)", 50.0, 500.0, 250.0, 1.0)

with right:
    st.subheader("2. Ultrasound image (optional)")
    up = st.file_uploader("Upload a B-mode liver ultrasound (PNG/JPG)", type=["png", "jpg", "jpeg"])
    us_img = None
    if up is not None:
        us_img = Image.open(up)
        st.image(us_img, caption="Uploaded ultrasound", use_container_width=True)
    mode = "Deep-learning CNN" if cnn else "Image-analysis (demo)"
    st.caption(f"Ultrasound engine: **{mode}**"
               + ("" if cnn else " — train a CNN (see README) to upgrade."))

go = st.button("🔍  Assess NAFLD risk", type="primary", use_container_width=True)

if go:
    sex_v = 1 if sex == "Male" else 0
    dm_v = 1 if diabetes == "Yes" else 0

    fli = fatty_liver_index(bmi, waist, triglycerides, ggt)
    f4 = fib4(age, ast, alt, platelets)
    ap = apri(ast, platelets)

    # tabular ML probability
    if tab_model:
        row = pd.DataFrame([[age, sex_v, bmi, waist, alt, ast, ggt,
                             triglycerides, hdl, glucose, platelets, dm_v]],
                           columns=tab_model["features"])
        ml_prob = float(tab_model["model"].predict_proba(row)[0, 1])
    else:
        ml_prob = min(fli / 100, 1)  # fallback if model missing

    # ultrasound probability
    us_prob = None
    if us_img is not None:
        us_prob = ultrasound_cnn_prob(us_img, cnn) if cnn else ultrasound_heuristic_prob(us_img)

    # ---- combine into an overall steatosis (fatty-liver) risk ----
    parts = [("Machine-learning model", ml_prob, 0.5),
             ("Fatty Liver Index", min(fli / 100, 1), 0.5)]
    if us_prob is not None:
        parts = [("Machine-learning model", ml_prob, 0.35),
                 ("Fatty Liver Index", min(fli / 100, 1), 0.30),
                 ("Ultrasound", us_prob, 0.35)]
    overall = sum(p * w for _, p, w in parts) / sum(w for _, _, w in parts)

    # fibrosis (progression) flag from FIB-4 / APRI
    fib_flag = (f4 is not None and f4 >= 2.67) or (ap is not None and ap >= 1.0)
    fib_intermediate = (f4 is not None and 1.3 <= f4 < 2.67)

    st.divider()
    st.subheader("Result")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall NAFLD risk", f"{overall*100:.0f}%")
    m2.metric("Fatty Liver Index", f"{fli:.0f}", "≥60 high · <30 low")
    m3.metric("FIB-4", f"{f4:.2f}" if f4 else "—", "≥2.67 high fibrosis risk")
    m4.metric("APRI", f"{ap:.2f}" if ap else "—", "≥1.0 significant fibrosis")

    st.progress(min(max(overall, 0), 1))

    # recommendation
    if overall >= 0.6 or fib_flag:
        st.error("### 🔴 REFER\n"
                 "High likelihood of NAFLD" + (" **with possible fibrosis**" if fib_flag else "") +
                 ". Refer for a confirmatory ultrasound / FibroScan and a hepatology review.")
    elif overall >= 0.3 or fib_intermediate:
        st.warning("### 🟠 SCREEN\n"
                   "Moderate likelihood. Arrange an abdominal ultrasound and repeat liver tests; "
                   "start lifestyle counselling.")
    else:
        st.success("### 🟢 MONITOR\n"
                   "Low likelihood at present. Advise diet/exercise and recheck in ~12 months, "
                   "sooner if metabolic risk factors worsen.")

    with st.expander("Why this result? (contributing signals)"):
        for name, p, w in parts:
            st.write(f"- **{name}**: {p*100:.0f}%  (weight {int(w*100)}%)")
        drivers = []
        if bmi >= 25: drivers.append(f"BMI {bmi:.0f}")
        if waist >= 90: drivers.append(f"waist {waist:.0f} cm")
        if triglycerides >= 150: drivers.append("high triglycerides")
        if hdl < 40: drivers.append("low HDL")
        if dm_v: drivers.append("type-2 diabetes")
        if alt > 40: drivers.append("raised ALT")
        st.write("**Key risk factors present:** " + (", ".join(drivers) if drivers else "none major"))
        if tab_model and tab_model.get("synthetic"):
            st.caption("Note: the bundled ML model was trained on synthetic demo data. "
                       "Retrain on NHANES/real data (see README) before real use.")

    st.caption("⚠️  Screening decision support only — not a diagnosis. Confirm with imaging and clinical judgement.")

st.divider()
st.caption("Built for MBA Dissertation 2026 — Karthik P · AI-Enabled CDSS for Early Detection of NAFLD")

