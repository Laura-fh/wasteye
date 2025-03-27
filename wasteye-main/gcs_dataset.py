import torch
from torch.utils.data import Dataset
import gcsfs
import io
from PIL import Image
import os
import yaml


# This script loads images and labels directly from Google Cloud Storage (GCS) and prepares them for training in YOLOv10

class GCSImageDataset(Dataset):
    def __init__(self, yaml_path, transform=None):
        # Load dataset configuration from YAML
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self.fs = gcsfs.GCSFileSystem()  # Initialize GCS file system

        # Get full GCS paths for images and labels
        self.image_paths = self._load_gcs_files(f"{data['path']}/{data['train']}")
        self.label_paths = [p.replace("small_batch_images", "small_batch_labels").replace(".jpg", ".txt") for p in self.image_paths]

        self.transform = transform
        self.classes = data["names"]  # List of class names

    def _load_gcs_files(self, gcs_dir):
        """ Get a list of image file paths from a GCS directory. """
        return [f"gs://{p}" for p in self.fs.ls(gcs_dir) if p.endswith(".jpg")]

    def _load_labels(self, label_path):
        """ Load YOLO format labels from GCS. """
        if self.fs.exists(label_path):
            with self.fs.open(label_path, "r") as f:
                return [list(map(float, line.strip().split())) for line in f.readlines()]
        return []  # No labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gcs_image_path = self.image_paths[idx]
        gcs_label_path = self.label_paths[idx]

        # Read image from GCS
        with self.fs.open(gcs_image_path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read())).convert('RGB')

        # Read labels from GCS
        labels = self._load_labels(gcs_label_path)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(labels)  # Return image + labels
