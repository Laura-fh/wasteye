from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import traceback

from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

# CORS allows to access to others apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# to load the model_
app.state.model = YOLO('wasteye-main/api/best_model_weights.pt', task='detect')

@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url=None):
    
    try: 
        if file:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
        
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image= Image.open(BytesIO(response.content))
            
        else:
            raise ValueError("No image input provided.")
        
        model = app.state.model

        results = model.predict(image, conf=0.183, iou=0.386, save=False)
        
        output = []
        for result in results:
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
                "detections": detections
            })
            # print(output)
            return {"results": output}
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)







@app.get("/")
def root():
    #return {"message": "Welcome to the YOLOv8 prediction API!"}
    return{"greeting":"hello!"}

# Add Uvicorn runner for convenience if running directly
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for development...")
    # Note: reload=True is for development, disable in production
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
