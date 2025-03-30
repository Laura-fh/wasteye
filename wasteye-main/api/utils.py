# 2º utils.py

import base64
import io
from PIL import Image
import numpy as np
import cv2
from fastapi import HTTPException # pip install fastapi[all]


def decode_base64_image(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(e)}")

def encode_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# is this part is necesary? 
def draw_boxes_on_image(image_array, results, names):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0]) #classes 
        label = names[cls_id] # plastic,metal,etc
        conf = float(box.conf[0])
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)# axes for bounding boxes.
        cv2.putText(image_array, f"{label} {conf:.2f}", (x1, y1 - 10), # it writes the class(label) on top the square
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image_array