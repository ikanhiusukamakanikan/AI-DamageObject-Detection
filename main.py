from fastapi import FastAPI, File, UploadFile, Form
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2

app = FastAPI()

# =========================
# LOAD MODEL SEKALI SAJA
# =========================
models = {
    "korosi": YOLO("models/korosi.pt"),
    "pothole": YOLO("models/pothole.pt"),
    "crack": YOLO("models/crack.pt"),
    "sampah": YOLO("models/sampah.pt"),
    "mix": YOLO("models/mix.pt"),
}


# =========================
# ENDPOINT PREDICT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    type: str = Form(...)
):

    # VALIDASI MODEL
    if type not in models:
        return {"error": "Invalid model type"}

    model = models[type]

    # =========================
    # READ IMAGE
    # =========================
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model.predict(image, verbose=False)[0]

    # =========================
    # PARSE DETECTIONS
    # =========================
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        detections.append({
            "class": model.names[cls_id],
            "confidence": conf,
            "bbox": box.xyxy[0].tolist()
        })

    # =========================
    # DRAW RESULT IMAGE
    # =========================
    plotted = results.plot()   # ⚠️ BGR from YOLO (JANGAN DIUBAH)

    # =========================
    # ENCODE TO BASE64
    # =========================
    _, buffer = cv2.imencode(".jpg", plotted)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # =========================
    # RESPONSE
    # =========================
    return {
        "model_used": type,
        "image": img_base64,
        "detections": detections,
        "has_detection": len(detections) > 0
    }