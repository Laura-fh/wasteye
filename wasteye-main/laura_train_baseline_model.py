from ultralytics import YOLO

# Load a model
# Model from scratch
model = YOLO("yolov8n.yaml")

# Use the model
results = model.train(data="laura_baseline_model.yaml", epochs=1, imgsz=416)
