from ultralytics import YOLO

from pathlib import Path

# Obtener el directorio actual del script
current_dir = Path(__file__).resolve().parent

# Subir un nivel en la estructura de carpetas
parent_dir = current_dir.parent

print("Directorio actual:", current_dir)
print("Directorio superior:", parent_dir)


# Load a model
# Model from scratch
#model = YOLO("yolov8n.yaml")

# Use the model
#results = model.train(data="laura_baseline_model.yaml", epochs=1, imgsz=416)
