from collections import Counter
from PIL import ImageDraw

COLORS = {
    "plastik": (0, 120, 255),
    "kertas_kardus": (205, 133, 63),
    "logam": (128, 128, 128),
    "organik": (34, 177, 76),
    "kaca": (0, 200, 200),
}

def resize_max(image, max_side=1280):
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))
    return image

def draw_detections(image, detections):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        color = COLORS.get(d["label"], (255, 0, 0))
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f'{d["label"]} {d["confidence"]:.2f}'
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(text), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 14)), text, fill=(255, 255, 255))
    return img

def count_per_class(detections, classes):
    c = Counter(d["label"] for d in detections)
    return [[k, c.get(k, 0)] for k in classes]