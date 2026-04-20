from ultralytics import YOLO
from sklearn.metrics import roc_auc_score

models = {
    "korosi": {
        "model": YOLO("models/korosi.pt"),
        "data": "datakorosi.yaml",
        "val": "datasets_korosi/images/val"
    },
    "pothole": {
        "model": YOLO("models/pothole.pt"),
        "data": "datapothole.yaml",
        "val": "datasets_pothole/images/val"
    },
    "crack": {
        "model": YOLO("models/crack.pt"),
        "data": "datacrack.yaml",
        "val": "datasets_crack/images/val"
    },
    "sampah": {
        "model": YOLO("models/sampah.pt"),
        "data": "datasampah.yaml",
        "val": "datasets_sampah/images/val"
    },
    "mix": {
        "model": YOLO("models/mix.pt"),
        "data": "datagabungan.yaml",
        "val": "datasets_gabungan/images"
    }
}

def evaluate(name, cfg):

    print(f"\n===== {name.upper()} =====")

    model = cfg["model"]
    data_path = cfg["data"]
    val_images = cfg["val"]

    # YOLO evaluation
    metrics = model.val(data=data_path)

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # ROC-AUC (binary approximation)
    results = model(val_images)

    y_true = []
    y_scores = []

    for r in results:
        if len(r.boxes) > 0:
            y_true.append(1)
            y_scores.append(float(r.boxes.conf.max()))
        else:
            y_true.append(0)
            y_scores.append(0.0)

    if len(set(y_true)) < 2:
        roc_auc = None
    else:
        roc_auc = roc_auc_score(y_true, y_scores)

    print(f"mAP50     : {map50:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc if roc_auc else 'N/A'}")


# RUN ALL
for name, cfg in models.items():
    evaluate(name, cfg)