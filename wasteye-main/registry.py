import os
from google.cloud import storage
from dotenv import load_dotenv
from google.auth import default

# Load GCS bucket paths from .env
load_dotenv()

# Access GCS bucket
BUCKET_NAME = "model_storage_wasteyeai"


# Authenticate
def get_gcs_client():
    # credentials, project = default ()
    # print (credentials)
    return storage.Client()

def upload_model(local_path="wasteye-main/best_model_weights.pt", gcs_path="models/best.pt"):
    """Upload trained model weights to GCS"""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded {local_path} to gs://{BUCKET_NAME}/{gcs_path}")

def download_model(gcs_path="models/best.pt", local_path="wasteye-main/best_model_weights.pt"):
    """Download model weights from GCS to local"""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"✅ Downloaded gs://{BUCKET_NAME}/{gcs_path} to {local_path}")


if __name__ == "__main__":
    #print (get_gcs_client())

    # To upload after training
    upload_model()

    # To download before inference
    # download_model()
