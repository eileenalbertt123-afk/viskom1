from pathlib import Path
from ultralytics import YOLO

# Kelas aplikasi (urutan dipakai untuk tabel ringkasan)
CLASSES = ["plastik", "kertas_kardus", "logam", "organik", "kaca"]

# Model teman memakai nama Inggris: glass, metal, organic, paper, plastic.
# Dipetakan lewat NAMA kelas, bukan nomor, supaya urutan tidak tertukar.
NAME_MAP = {
    "plastic": "plastik",
    "paper": "kertas_kardus",
    "cardboard": "kertas_kardus",
    "metal": "logam",
    "organic": "organik",
    "glass": "kaca",
}

MODEL_PATH = Path(__file__).parent / "models" / "waste_yolov8s_best.pt"
DEFAULT_CONF = 0.225   # confidence F1-optimal dari notebook teman
IMGSZ = 640            # sama dengan ukuran saat model dilatih

_model = None

def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {MODEL_PATH}. "
                "Taruh best_waste_yolov8s.pt di folder models/."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model

def detect(image, conf_threshold=DEFAULT_CONF):
    """image: PIL.Image -> list of {label, confidence, bbox}"""
    model = _get_model()
    result = model.predict(image, conf=conf_threshold, iou=0.6, imgsz=IMGSZ,
                           max_det=500, verbose=False)[0]
    names = result.names
    detections = []
    for box in result.boxes:
        en = str(names[int(box.cls[0])]).lower()
        label = NAME_MAP.get(en)
        if label is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "label": label,
            "confidence": round(float(box.conf[0]), 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections