# model_loader.py
from ultralytics import YOLO

def load_model():
    model = YOLO('yolov10n.pt', task='detect') #model trained (model.py)
    return model