import streamlit as st
import requests
import base64

st.set_page_config(
    page_title="YOLO Damage Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 YOLO Damage Detection System")
st.caption("Detect corrosion, pothole, crack, and more using AI")

# sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    type_model = st.selectbox(
        "Select Detection Model",
        ["korosi", "pothole", "crack", "sampah", "mix"]
    )

    st.info("Upload image then click predict")

# main layout
col1, col2 = st.columns(2)

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    with col1:
        st.subheader("📥 Input Image")
        st.image(uploaded_file)

    if st.button("🚀 Run Detection"):

        with st.spinner("🧠 AI is analyzing image..."):

            files = {"file": uploaded_file}
            data = {"type": type_model}

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files,
                data=data
            )

            result = response.json()

        # ERROR HANDLING
        if "error" in result:
            st.error(result["error"])
        else:

            # OUTPUT IMAGE
            img_bytes = base64.b64decode(result["image"])

            with col2:
                st.subheader("📤 Detection Result")
                st.image(img_bytes)

            # STATUS SECTION
            st.divider()

            if result["has_detection"]:
                st.success("🚨 Damage Detected!")
            else:
                st.info("✅ No damage detected")

            # DETAILS
            st.subheader("📊 Detection Details")

            if result["detections"]:
                for i, det in enumerate(result["detections"]):
                    with st.expander(f"Detection {i+1} - {det['class']}"):
                        st.write(f"**Class:** {det['class']}")
                        st.write(f"**Confidence:** {det['confidence']:.2f}")
                        st.write(f"**BBox:** {det['bbox']}")

            # JSON raw (optional)
            with st.expander("🔍 Raw JSON Output"):
                st.json(result)