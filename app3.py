"""

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

# AI-Enabled CDSS for Early Detection of NAFLD

A clinical decision support tool that estimates the likelihood of early
**Non-Alcoholic Fatty Liver Disease (NAFLD / MASLD)** from routine, low-cost
inputs and recommends a next step (**SCREEN / REFER / MONITOR**).

- Enter age, BMI, waist and routine blood tests → the app auto-computes the
  **FIB-4**, **Fatty Liver Index (FLI)** and **APRI** scores.
- A **machine-learning model** estimates NAFLD risk from the same inputs.
- Optionally upload a **B-mode liver ultrasound** — the app analyses it (and
  uses a deep-learning CNN if one is provided).
- Outputs an overall risk %, the individual scores, and a recommendation.

> ⚠️ Screening decision support only — **not a diagnosis**. Fatty liver is
> confirmed by ultrasound/FibroScan or biopsy and a doctor's judgement.

**MBA Dissertation 2026 — Karthik P.**

---

## Files
| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `model.pkl` | Bundled ML model (trained on synthetic demo data — retrain for real use) |
| `train_tabular.py` | Trains the ML model (synthetic, or your own CSV) |
| `train_ultrasound.py` | Optional — trains the ultrasound CNN |
| `requirements.txt` | Python dependencies |

## Run it on your computer (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy free on Streamlit Community Cloud (via GitHub)

**A. Put the code on GitHub**
1. Go to https://github.com and click **New repository**. Name it e.g. `nafld-cdss-app`, keep it **Public**, click **Create**.
2. On the new repo page click **Add file → Upload files**.
3. Drag in **all the files in this folder** (`app.py`, `model.pkl`, `train_tabular.py`, `train_ultrasound.py`, `requirements.txt`, `README.md`). Click **Commit changes**.

**B. Deploy on Streamlit**
1. Go to https://share.streamlit.io and **Sign in with GitHub** (allow access).
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose your `nafld-cdss-app` repo, branch `main`, main file `app.py`.
4. Click **Deploy**. First build takes ~2–3 minutes. You'll get a public URL like
   `https://your-name-nafld-cdss-app.streamlit.app` — put this in your dissertation.

To update the app later, just upload changed files to GitHub — Streamlit redeploys automatically.

---

## Optional: turn on the ultrasound deep-learning model
The app works immediately using a built-in image-analysis estimate. To use a
real CNN instead:
1. Download a B-mode fatty-liver ultrasound dataset from Kaggle
   (search *"B-mode fatty liver ultrasound images"*).
2. Sort images into `data/normal/` and `data/fatty/`.
3. `pip install torch torchvision` then `python train_ultrasound.py --data data`.
4. Add the produced `ultrasound_cnn.pt` to your GitHub repo, and **uncomment the
   `torch` / `torchvision` lines** in `requirements.txt`. The app auto-detects it.

## Optional: retrain the ML model on real data
Prepare a CSV with these columns plus a binary `nafld` column:
`age, sex, bmi, waist, alt, ast, ggt, triglycerides, hdl, glucose, platelets, diabetes`
```bash
python train_tabular.py --csv your_data.csv
```
Commit the new `model.pkl` to GitHub.

*(NHANES is a good free source with the blood markers, BMI and FibroScan data
needed to define fatty liver.)*

streamlit>=1.36
scikit-learn==1.7.2
pandas>=2.0
numpy>=1.26
joblib>=1.3
pillow>=10.0
# --- Optional: enable the ultrasound deep-learning CNN ---
# Uncomment the two lines below AFTER you have trained ultrasound_cnn.pt.
# Leaving them commented keeps the free deploy fast (the app falls back to
# the built-in image-analysis estimate).
# torch>=2.2
# torchvision>=0.17

"""
Train the tabular NAFLD risk model.

By default this generates a realistic SYNTHETIC dataset and trains a
RandomForest, so the app works out of the box. To train on REAL data
(e.g. an NHANES-derived CSV), pass --csv path/to/data.csv where the CSV
has the FEATURES columns below plus a binary 'nafld' column.

    python train_tabular.py                 # synthetic demo model
    python train_tabular.py --csv mydata.csv

Output: model.pkl  (dict with 'model' and 'features')
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURES = ["age", "sex", "bmi", "waist", "alt", "ast", "ggt",
            "triglycerides", "hdl", "glucose", "platelets", "diabetes"]


def make_synthetic(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 13, n).clip(18, 85)
    sex = rng.integers(0, 2, n)                       # 0 = female, 1 = male
    bmi = rng.normal(26, 4.5, n).clip(15, 45)
    waist = (bmi * 2.6 + rng.normal(20, 6, n)).clip(60, 140)
    diabetes = (rng.random(n) < (0.10 + (bmi > 27) * 0.20)).astype(int)
    triglycerides = np.exp(rng.normal(4.9, 0.4, n)).clip(40, 500)   # mg/dL
    hdl = rng.normal(48, 11, n).clip(20, 90)
    glucose = rng.normal(100, 25, n).clip(70, 260) + diabetes * 30
    alt = np.exp(rng.normal(3.1, 0.5, n)).clip(8, 200)             # U/L
    ast = (alt * rng.normal(0.9, 0.2, n)).clip(8, 200)
    ggt = np.exp(rng.normal(3.4, 0.6, n)).clip(10, 400)
    platelets = rng.normal(250, 55, n).clip(90, 450)               # x10^9/L

    # Latent NAFLD risk (steatosis) driven by known metabolic factors.
    z = (-11.0
         + 0.16 * bmi
         + 0.03 * waist
         + 0.004 * triglycerides
         - 0.03 * hdl
         + 0.012 * alt
         + 0.004 * ggt
         + 0.9 * diabetes
         + 0.008 * (glucose - 100)
         + 0.01 * (age - 45))
    p = 1 / (1 + np.exp(-z))
    nafld = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, waist=waist, alt=alt,
                           ast=ast, ggt=ggt, triglycerides=triglycerides,
                           hdl=hdl, glucose=glucose, platelets=platelets,
                           diabetes=diabetes, nafld=nafld))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"Loaded real data: {df.shape}")
    else:
        df = make_synthetic()
        print(f"Generated synthetic demo data: {df.shape}")

    X, y = df[FEATURES], df["nafld"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=20, random_state=1,
                                   class_weight="balanced")
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"Validation AUROC: {auc:.3f}")

    joblib.dump({"model": model, "features": FEATURES, "auroc": round(float(auc), 3),
                 "synthetic": args.csv is None}, "model.pkl")
    print("Saved model.pkl")


if __name__ == "__main__":
    main()

"""
OPTIONAL: train the ultrasound CNN (MobileNetV2 transfer learning).

The app works WITHOUT this — it falls back to an image-analysis
(echogenicity) estimate. Train this to enable a real deep-learning model.

1. Download a B-mode fatty-liver ultrasound dataset from Kaggle
   (search: "B-mode fatty liver ultrasound images").
2. Arrange images into two folders:
       data/normal/*.png
       data/fatty/*.png
3. Install: pip install torch torchvision
4. Run:    python train_ultrasound.py --data data
5. It saves ultrasound_cnn.pt — put it next to app.py and redeploy.
   The app auto-detects it and switches from heuristic to CNN.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

IMG = 224
CLASSES = ["fatty", "normal"]  # alphabetical -> index 0 = fatty, 1 = normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with normal/ and fatty/ subfolders")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data, transform=tf)
    print("Classes:", ds.classes)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    net.classifier[1] = nn.Linear(net.last_channel, 2)
    net = net.to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        net.train(); tot = 0; correct = 0; run = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
            run += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); tot += x.size(0)
        print(f"epoch {ep+1}: loss={run/tot:.3f} acc={correct/tot:.3f}")

    torch.save({"state_dict": net.state_dict(), "classes": ds.classes}, "ultrasound_cnn.pt")
    print("Saved ultrasound_cnn.pt")


if __name__ == "__main__":
    main()
