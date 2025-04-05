from PIL import Image
from io import BytesIO
import requests


url = 'https://wasteye-243765311449.europe-west1.run.app/predict'
image = "wasteye-main/api/test_api.jpg"


with open(image,"rb") as f:
    files = {"file": (image, f, "image/jpeg")}
    response = requests.post(url, files=files)
    
print(response.json())