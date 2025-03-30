from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ImageRequest
from model_loader import load_model
from utils import decode_base64_image, encode_image_to_base64, draw_boxes_on_image
import numpy as np
from PIL import Image
url= "https://londonrecycles.co.uk/wp-content/uploads/2020/08/Plastic-recycling.jpg"
app = FastAPI()

# CORS allows to access to others apps// CORS para permitir acceso desde otras apps 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# to load the model // Cargar el modelo una vez al iniciar el servidor
app.state.model = load_model()

@app.get("/predict_local_batch")
def predict_local_batch():
    model = app.state.model

    # Lista de imágenes locales (ajusta las rutas según tu estructura)
    image_paths = [url]

    results = model.predict(image_paths, conf=0.25)

    output = []
    for result, path in zip(results, image_paths):
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = list(map(float, box.xyxy[0]))

            detections.append({
                "class": label,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

        output.append({
            "image_path": path,
            "detections": detections
        })
    # print(output)
    return {"results": output}


@app.get("/")
def root():
    #return {"message": "Welcome to the YOLOv8 prediction API!"}
    return{"greeting":"hello!"}