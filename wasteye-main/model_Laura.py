from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
from gcs_dataset import GCSImageDataset
from torchvision import transforms
import torch


# Specify your service account file from the correct project (wasteye-ai)
#client = storage.Client.from_service_account_json('/path/to/your/service-account-key.json')

# Access the GCS bucket
#bucket = client.get_bucket('raw_image_data_testing')

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset from GCS
dataset = GCSImageDataset("cloud_test.yaml", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)


def train_yolo(data_yaml, epochs=10, img_size=416): # Define validation dataset
    model = YOLO("yolov10n.pt") # Load a pretrained object detection model
    print(f"\n🚀 Training {model}...\n")
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
    print("✅ Training Complete!")
    return results

# Train models
trained_model = train_yolo(data_yaml=dataloader, epochs=1, img_size=416)

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
#     #call_dataset() # Joel function
    train_yolo()
#     #evaluate_model()
#     #pred()
