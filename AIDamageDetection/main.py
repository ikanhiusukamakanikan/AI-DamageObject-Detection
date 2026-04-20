from fastapi import FastAPI, File, UploadFile, Form
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import base64
import numpy as np

app = FastAPI()

# Load models sekali (biar cepat)
models = {
    "korosi": YOLO("models/korosi.pt"),
    "pothole": YOLO("models/pothole.pt"),
    "crack": YOLO("models/crack.pt"),
    "sampah": YOLO("models/sampah.pt"),
    "mix": YOLO("models/mix.pt"),
}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    type: str = Form(...)
):

    if type not in models:
        return {"error": "Invalid type"}

    model = models[type]

    # read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # inference
    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    # draw bounding box image
    plotted = results[0].plot()

    # FIX COLOR (BGR -> RGB)
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # encode to base64
    _, buffer = cv2.imencode(".jpg", plotted)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "model_used": type,
        "image": img_base64,
        "detections": detections,
        "has_detection": len(detections) > 0
    }