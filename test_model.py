import os
import streamlit as st
import pandas as pd
import requests
import base64

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Damage Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI STYLE (PERSIS KAMU) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }

    div[data-testid="stMetric"] {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #30363d;
    }

    .stExpander {
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        background-color: #0d1117 !important;
    }

    footer {visibility: hidden;}

    h1 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG ---
TEST_FOLDER = "data_test_labeled"
CSV_PATH = "labels.csv"
API_URL = "http://127.0.0.1:8000/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Settings")

    choice = st.selectbox(
        "🎯 Target Model",
        ["mix", "pothole", "crack", "korosi", "sampah"]
    )

# --- MAIN ---
st.title("AI Facility Damage Detection")
st.write("Dashboard verifikasi akurasi model YOLOv8 (API Mode)")

df = pd.read_csv(CSV_PATH)

results_list = []

in_domain_total = 0
in_domain_correct = 0

out_domain_total = 0
out_domain_correct = 0

target_focus = choice.lower()

# --- PROCESS ---
if st.button("🚀 Run Evaluation"):

    progress = st.progress(0)

    for i, row in df.iterrows():

        img_name = row["filename"]
        true_label = str(row["label"]).lower().strip()

        img_path = os.path.join(TEST_FOLDER, img_name)

        if not os.path.exists(img_path):
            continue

        with open(img_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": f},
                data={"type": choice}
            )

        result = response.json()

        detected = [d["class"].lower() for d in result["detections"]]

        # =========================
        # RUMUS ASLI KAMU (FIXED)
        # =========================

        is_match = False
        category = ""

        if target_focus == "mix":

            category = "In-Domain"

            is_match = (len(detected) == 0) if true_label == "anomali" else (true_label in detected)

            in_domain_total += 1
            if is_match:
                in_domain_correct += 1

        else:

            if true_label == target_focus:

                category = "In-Domain"
                is_match = (target_focus in detected)

                in_domain_total += 1
                if is_match:
                    in_domain_correct += 1

            else:

                category = "Out-of-Domain"
                is_match = (len(detected) == 0)

                out_domain_total += 1
                if is_match:
                    out_domain_correct += 1

        img_bytes = base64.b64decode(result["image"])

        results_list.append({
            "filename": img_name,
            "label_asli": true_label,
            "prediksi": ", ".join(detected) if detected else "None (Bersih)",
            "status": "✅ SESUAI" if is_match else "❌ TIDAK SESUAI",
            "category": category,
            "img_viz": img_bytes
        })

        progress.progress((i + 1) / len(df))

# --- METRICS ---
st.markdown("### 📊 Metrics Performance")

m1, m2, m3 = st.columns(3)

with m1:
    acc_in = (in_domain_correct / in_domain_total * 100) if in_domain_total else 0
    st.metric(
        f"Accuracy: {target_focus.capitalize()}",
        f"{in_domain_correct}/{in_domain_total}",
        f"{acc_in:.1f}%"
    )

with m2:
    acc_out = (out_domain_correct / out_domain_total * 100) if out_domain_total else 0
    st.metric(
        "Specificity (Ignore Others)",
        f"{out_domain_correct}/{out_domain_total}",
        f"{acc_out:.1f}%"
    )

with m3:
    total = in_domain_total + out_domain_total
    correct = in_domain_correct + out_domain_correct
    acc_all = (correct / total * 100) if total else 0
    st.metric("Overall Score", f"{correct}/{total}", f"{acc_all:.1f}%")

st.markdown("---")

# --- FILTER ---
st.subheader("🔍 Result Details")

f1, f2 = st.columns(2)

with f1:
    filter_stat = st.radio("Status Filter:", ["All", "Sesuai", "Tidak Sesuai"], horizontal=True)

with f2:
    filter_cat = st.radio("Domain Filter:", ["All", "In-Domain", "Out-of-Domain"], horizontal=True)

# --- GALLERY ---
cols = st.columns(2)
idx = 0

for item in results_list:

    if filter_stat == "Sesuai" and "❌" in item["status"]:
        continue
    if filter_stat == "Tidak Sesuai" and "✅" in item["status"]:
        continue

    if filter_cat != "All" and item["category"] != filter_cat:
        continue

    with cols[idx % 2]:

        with st.expander(f"{item['status']} | {item['filename']} ({item['category']})"):

            st.image(item["img_viz"])

            c1, c2 = st.columns(2)

            c1.write(f"**Target:** `{item['label_asli'].upper()}`")
            c2.write(f"**Result:** `{item['prediksi'].upper()}`")

    idx += 1