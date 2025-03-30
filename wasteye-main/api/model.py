from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

# Define paths
#DATA_CONFIG = "gs://raw_image_data_testing/cloud_test.yaml"

def train_yolo(data_yaml, epochs=10, img_size=416): # Define validation dataset
    model = YOLO("yolov10n.pt") # Load a pretrained object detection model
    print(f"\n🚀 Training {model}...\n")
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
    print("✅ Training Complete!")
    return results

# Train models
trained_model = train_yolo(data_yaml="cloud_test.yaml")

# Load models weights
#model_beta = YOLO("path/to/best/weight")

"""def evaluate_model(
    model : model_name,
    batch_size=64
)
    print(f"\n🚀 Evaluating {model_name}...\n")




def pred(image_path, conf_level):
    model = model_beta
    pred = model_beta.predict(image_path, conf_level, save=True)

    print("\n✅ prediction done: \n")
    return pred"""

if __name__ == '__model__':
    #call_dataset() # Joel function
    train_yolo()
    #evaluate_model()
    #pred()
