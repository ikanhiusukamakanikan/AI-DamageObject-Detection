import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# --- FIX WINERROR 1114 (URUTAN PALING ATAS) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

# Penanganan Error Import
try:
    from ultralytics import YOLO
except OSError as e:
    st.error(f"Terjadi kesalahan sistem (DLL Error): {e}")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Damage Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
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
    h1 { color: #58a6ff; font-family: 'Inter', sans-serif; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# Path Constants
MODELS_LIST = {
    "Mix (All-in-One)": "models/mix.pt",
    "Pothole Model": "models/pothole.pt",
    "Crack Model": "models/crack.pt",
    "Korosi Model": "models/korosi.pt",
    "Sampah Model": "models/sampah.pt"
}
TEST_FOLDER = "data_test_labeled"
CSV_PATH = "labels.csv"

# --- CACHED INFERENCE ENGINE ---
@st.cache_data
def run_full_inference(model_choice, conf_thresh):
    """
    Fungsi ini hanya berjalan jika Model atau Threshold diubah.
    Filter UI tidak akan memicu fungsi ini lagi.
    """
    # Load model temporary inside function for prediction
    temp_model = YOLO(MODELS_LIST[model_choice])
    df = pd.read_csv(CSV_PATH)
    
    results = []
    in_total, in_corr = 0, 0
    out_total, out_corr = 0, 0
    target_focus = model_choice.split()[0].lower()

    for _, row in df.iterrows():
        img_name, true_label = row['filename'], str(row['label']).lower().strip()
        img_path = os.path.join(TEST_FOLDER, img_name)
        
        if os.path.exists(img_path):
            # Inferensi
            res = temp_model.predict(img_path, conf=conf_thresh, device='cpu', verbose=False)[0]
            detected = list(set([temp_model.names[int(box.cls[0])].lower() for box in res.boxes]))
            
            is_match = False
            category = ""

            # Logika Evaluasi
            if target_focus == "mix":
                category = "In-Domain"
                is_match = (len(detected) == 0) if true_label == "anomali" else (true_label in detected)
                if is_match: in_corr += 1
                in_total += 1
            else:
                if true_label == target_focus:
                    category = "In-Domain"
                    is_match = (target_focus in detected)
                    in_total += 1
                    if is_match: in_corr += 1
                else:
                    category = "Out-of-Domain"
                    is_match = (len(detected) == 0)
                    out_total += 1
                    if is_match: out_corr += 1
            
            # Simpan hasil plotting sebagai array (RGB)
            res_plotted = res.plot()[:, :, ::-1] 
            
            results.append({
                "filename": img_name, 
                "label_asli": true_label, 
                "prediksi": ", ".join(detected) if detected else "None (Bersih)",
                "status": "✅ SESUAI" if is_match else "❌ TIDAK SESUAI",
                "category": category, 
                "img_viz": res_plotted 
            })
            
    return results, in_total, in_corr, out_total, out_corr

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Control Panel")
    st.markdown("---")
    choice = st.selectbox("🎯 Target Model", list(MODELS_LIST.keys()))
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    st.markdown("---")
    st.caption("Data akan di-cache secara otomatis. Filter tidak memicu loading model.")

# --- MAIN LOGIC ---
st.title("AI Facility Damage Detection")

if not os.path.exists(CSV_PATH):
    st.error(f"File `{CSV_PATH}` tidak ditemukan!")
    st.stop()

# Menjalankan inferensi (hanya jika params berubah)
with st.spinner(f"🚀 Running Inference with {choice}..."):
    results_list, in_total, in_corr, out_total, out_corr = run_full_inference(choice, conf_threshold)

# --- METRICS DISPLAY ---
st.markdown("### 📊 Metrics Performance")
m1, m2, m3 = st.columns(3)
target_focus = choice.split()[0].capitalize()

with m1:
    acc_in = (in_corr / in_total * 100) if in_total > 0 else 0
    st.metric(f"Accuracy: {target_focus}", f"{in_corr}/{in_total}", f"{acc_in:.1f}%")
with m2:
    if choice != "Mix (All-in-One)":
        acc_out = (out_corr / out_total * 100) if out_total > 0 else 0
        st.metric("Specificity (Ignore Others)", f"{out_corr}/{out_total}", f"{acc_out:.1f}%")
    else:
        st.metric("Total Sample", len(results_list))
with m3:
    total_match = in_corr + out_corr
    total_all = in_total + out_total
    overall_acc = (total_match / total_all * 100) if total_all > 0 else 0
    st.metric("Overall Match Score", f"{total_match}/{total_all}", f"{overall_acc:.1f}%")

st.markdown("---")

# --- FILTERS (TIDAK MEMICU LOADING ULANG) ---
st.subheader("🔍 Result Details")
f_col1, f_col2 = st.columns(2)
with f_col1:
    filter_stat = st.radio("Status Filter:", ["All", "Sesuai", "Tidak Sesuai"], horizontal=True)
with f_col2:
    filter_cat = st.radio("Domain Filter:", ["All Domains", "In-Domain", "Out-of-Domain"], horizontal=True)

# --- GALLERY DISPLAY ---
cols = st.columns(2)
display_idx = 0

for item in results_list:
    # Filter Status
    if filter_stat == "Sesuai" and "❌" in item["status"]: continue
    if filter_stat == "Tidak Sesuai" and "✅" in item["status"]: continue
    
    # Filter Category
    if filter_cat == "In-Domain" and item["category"] != "In-Domain": continue
    if filter_cat == "Out-of-Domain" and item["category"] != "Out-of-Domain": continue
    
    with cols[display_idx % 2]:
        with st.expander(f"{item['status']} | {item['filename']} ({item['category']})"):
            st.image(item['img_viz']) 
            c_a, c_b = st.columns(2)
            c_a.write(f"**Target:** `{item['label_asli'].upper()}`")
            c_b.write(f"**Result:** `{item['prediksi'].upper()}`")
    display_idx += 1