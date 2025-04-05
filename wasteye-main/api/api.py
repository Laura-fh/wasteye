from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# CORS allows to access to others apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# to load the model
app.state.model = YOLO('yolov10n.pt', task='detect')

@app.get("/predict")
def predict(image="wasteye-main/api/test_api.jpg"):

    model = app.state.model

    results = model.predict(image, conf=0.25)

    output = []
    for result, path in zip(results, image):
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

# Add Uvicorn runner for convenience if running directly
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for development...")
    # Note: reload=True is for development, disable in production
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
